from transformers import AutoTokenizer
import multiprocessing
import queue
from tqdm import tqdm
import yaml
import os
from loguru import logger
import re
import numpy as np
import time
import traceback
from typing import *


MODEL_PATH = yaml.safe_load(open("config/datagen_config.yaml"))["local_llm_path"]
CTX_LENGTH = 8192
NUM_GPUS = yaml.safe_load(open("config/datagen_config.yaml"))["num_gpus"]
GPUS_PER_MODEL = yaml.safe_load(open("config/datagen_config.yaml"))["gpu_per_model"]
BATCH_SIZE = 32 * 8
ENGINE = 'vllm'

# Max generated tokens per request. Decoupled from ctx_length so vLLM's
# scheduler doesn't over-reserve output slots (planning for 30K-token
# outputs caps batch size even when real answers are ~1 sentence).
# Description placeholders return 1 sentence (~128 tokens); TSEvol
# evolution + validation return JSON blobs (~500-1000 tokens). 2048 is
# safe for all current phases; pass a custom SamplingParams via the
# queue if a specific phase needs more.
MAX_OUTPUT_TOKENS = 2048

# Safety margins for pre-filter: prompt token count + margin must fit in
# ctx_length. Text-only worker reserves room for output generation. The TS
# worker uses a larger margin because each multi-modal timeseries input
# adds encoder tokens on top of the text prompt (not counted by the HF
# tokenizer), so we need extra headroom beyond the output reservation.
_CTX_SAFETY_MARGIN = 512       # text-only worker: reserve for output
_CTX_SAFETY_MARGIN_TS = 2048   # TS worker: reserve for output + TS encoder tokens


def _generate_and_emit(llm, inputs, args_list, sampling_params, gpu_id, output_queue):
    """Generate a batch via vLLM, recursively bisecting on any failure.

    The pre-filter in each worker catches the common context-overflow case
    cheaply (tokenize + compare). This function is the safety net for
    cases the pre-filter can't see — notably TS worker overflow from
    multi-modal encoder tokens that aren't visible to the HF tokenizer.

    Recursion: if `llm.generate` raises, split the batch in half and
    retry each half. Base case (single sample): emit empty output and
    log — main thread doesn't hang. Cost of isolating one bad sample
    in a batch of N is O(log N) extra generate calls.
    """
    if not inputs:
        return
    try:
        answers = llm.generate(inputs, sampling_params, use_tqdm=False)
        sample_n = sampling_params.n
        if sample_n > 1:
            texts = [[o.text for o in a.outputs] for a in answers]
        else:
            texts = [a.outputs[0].text for a in answers]
        for text, args in zip(texts, args_list):
            output_queue.put((text, *args))
    except Exception as e:
        if len(inputs) == 1:
            logger.warning(
                f"[worker {gpu_id}] Single-sample generation failed, "
                f"skipping: {type(e).__name__}: {e}"
            )
            output_queue.put(("", *args_list[0]))
            return
        mid = len(inputs) // 2
        logger.warning(
            f"[worker {gpu_id}] Batch of {len(inputs)} failed "
            f"({type(e).__name__}), bisecting to isolate bad sample"
        )
        _generate_and_emit(
            llm, inputs[:mid], args_list[:mid],
            sampling_params, gpu_id, output_queue,
        )
        _generate_and_emit(
            llm, inputs[mid:], args_list[mid:],
            sampling_params, gpu_id, output_queue,
        )


def worker_llama_cpp(input_queue, output_queue, gpu_id, batch_size, sample_n, finished_flag, ready_cnt, model_path=MODEL_PATH, ctx_length=CTX_LENGTH):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    try:
        from llama_cpp import Llama
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1, # Uncomment to use GPU acceleration
            n_ctx=ctx_length,
            chat_format='qwen'
        )
        print(f"[worker {gpu_id}] Initialization finished.")
        ready_cnt.value = ready_cnt.value + 1
        
        while not finished_flag.get():
            if input_queue.empty():
                time.sleep(1)
                continue

            batch_prompts = []
            batch_args = []
            for _ in range(batch_size):
                if not input_queue.empty():
                    try:
                        cur_items = input_queue.get_nowait()
                    except queue.Empty:
                        break
                    except Exception as err:
                        logger.warning("Unexpected error in queue worker", exc_info=True)
                        break
                    batch_prompts.append(cur_items[0])
                    batch_args.append(cur_items[1:])
                else:
                    break

            if batch_prompts:
                batch_generates = []
                for prompt in batch_prompts:
                    logger.debug(f"[INPUT] {prompt}")
                    cur_generate = llm(
                        prompt,
                        stop='<|im_end|>',
                        temperature=0.0,
                        top_k=10,
                        max_tokens=MAX_OUTPUT_TOKENS
                    )
                    logger.debug(f"[OUTPUT] {cur_generate['choices'][0]['text']}")
                    batch_generates.append(cur_generate['choices'][0]['text'])
                for generate, args in zip(batch_generates, batch_args):
                    output_queue.put((generate, *args))
    except Exception as err:
        logger.error(f"[worker {gpu_id}] {err}")
        time.sleep(5)

def worker_vllm(input_queue, output_queue, gpu_id, batch_size, sample_n, finished_flag, ready_cnt, model_path=MODEL_PATH, ctx_length=CTX_LENGTH, rope_scaling=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    try:
        from vllm import LLM, SamplingParams
        default_sampling_params = SamplingParams(temperature=0.5, top_p=0.95, max_tokens=MAX_OUTPUT_TOKENS, stop_token_ids=[151643, 151645], stop=['<|endoftext|>', '<|im_end|>'], n=sample_n)
        llm_kwargs = dict(model=model_path, trust_remote_code=True, max_model_len=ctx_length, tensor_parallel_size=len(gpu_id.split(',')), gpu_memory_utilization=0.95, dtype='half', enable_prefix_caching=True)
        if rope_scaling is not None:
            # Pass through hf_overrides so vLLM extends context via YaRN/linear/etc.
            llm_kwargs["hf_overrides"] = {"rope_scaling": rope_scaling}
            print(f"[worker {gpu_id}] Using rope_scaling={rope_scaling}")
        llm = LLM(**llm_kwargs)
        print(f"[worker {gpu_id}] Initialization finished.")
        ready_cnt.value = ready_cnt.value + 1

        while not finished_flag.get():
            if input_queue.empty():
                time.sleep(1)
                continue

            batch_prompts = []
            batch_args = []

            sampling_params = default_sampling_params

            for _ in range(batch_size):
                if not input_queue.empty():
                    try:
                        cur_items = input_queue.get_nowait()
                    except queue.Empty:
                        break
                    except Exception as err:
                        logger.warning("Unexpected error in queue worker", exc_info=True)
                        break
                    batch_prompts.append(cur_items[0])
                    batch_args.append(cur_items[1:])
                else:
                    break

            if batch_prompts:
                if type(batch_args[0][-1]) == SamplingParams:
                    sampling_params = batch_args[0][-1]
                    print(f"[worker {gpu_id}] Using custom sampling params: {sampling_params}")
                else:
                    sampling_params = default_sampling_params

                # Pre-filter (cheap, common case): skip prompts whose text
                # tokens alone exceed ctx_length. `_generate_and_emit`
                # below handles the remainder via bisect-on-failure, so
                # anything the pre-filter misses still gets isolated
                # rather than wiping the whole batch.
                safe_max = max(1, ctx_length - _CTX_SAFETY_MARGIN)
                tokenizer = llm.get_tokenizer()
                to_gen, to_gen_args, skipped = [], [], []
                for prompt, args in zip(batch_prompts, batch_args):
                    try:
                        tok_len = len(tokenizer.encode(prompt))
                    except Exception:
                        tok_len = -1  # permissive on tokenization failure
                    if 0 < safe_max < tok_len:
                        skipped.append((args, tok_len))
                    else:
                        to_gen.append(prompt)
                        to_gen_args.append(args)
                if skipped:
                    logger.warning(
                        f"[worker {gpu_id}] Pre-filter skipping {len(skipped)} oversize prompts "
                        f"(token lengths: {sorted({s[1] for s in skipped})}, "
                        f"ctx_length={ctx_length}, safe_max={safe_max})"
                    )
                    for args, _ in skipped:
                        output_queue.put(("", *args))

                _generate_and_emit(
                    llm, to_gen, to_gen_args,
                    sampling_params, gpu_id, output_queue,
                )
    except Exception as err:
        logger.error(f"[worker {gpu_id}] Fatal init error: {err}")
        traceback.print_exc()
        time.sleep(5)

def worker_vllm_ts(input_queue, output_queue, gpu_id, batch_size, sample_n, finished_flag, ready_cnt, model_path=MODEL_PATH, ctx_length=CTX_LENGTH):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    try:
        from vllm import LLM, SamplingParams
        import ts_vllm
        
        # 1. Define Defaults
        default_sampling_params = SamplingParams(
            temperature=0.5,
            top_p=0.95,
            max_tokens=MAX_OUTPUT_TOKENS,
            stop_token_ids=[151643, 151645],
            stop=['<|endoftext|>', '<|im_end|>'],
            n=sample_n
        )

        # 2. Initialize Model with ctx_length (allocates VRAM)
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            max_model_len=ctx_length,
            tensor_parallel_size=len(gpu_id.split(',')),
            gpu_memory_utilization=0.95,
            limit_mm_per_prompt={"timeseries": 50},
            enable_prefix_caching=True
        )
        print(f"[worker {gpu_id}] Initialization finished.")
        ready_cnt.value = ready_cnt.value + 1
        
        while not finished_flag.get():
            if input_queue.empty():
                time.sleep(0.1)
                continue

            batch_inputs = []
            batch_args = []
            
            # 3. Reset to defaults for every batch
            sampling_params = default_sampling_params

            for _ in range(batch_size):
                if not input_queue.empty():
                    try:
                        cur_items = input_queue.get_nowait()
                    except queue.Empty:
                        break
                    except Exception as err:
                        logger.warning("Unexpected error in queue worker", exc_info=True)
                        break
                    batch_inputs.append(cur_items[0])
                    batch_args.append(cur_items[1:])
                else:
                    break

            if batch_inputs:
                # 4. FIX: Check if custom SamplingParams were passed in the queue
                if batch_args and len(batch_args[0]) > 0 and isinstance(batch_args[0][-1], SamplingParams):
                    sampling_params = batch_args[0][-1]
                else:
                    sampling_params = default_sampling_params

                # Pre-filter (cheap, catches overflow from text alone).
                # The TS encoder adds multi-modal tokens that aren't visible
                # to the HF tokenizer, so anything the pre-filter misses
                # still hits `_generate_and_emit`'s bisect-on-failure path
                # to isolate the bad sample rather than wipe the batch.
                safe_max = max(1, ctx_length - _CTX_SAFETY_MARGIN_TS)
                tokenizer = llm.get_tokenizer()
                to_gen, to_gen_args, skipped = [], [], []
                for inp, args in zip(batch_inputs, batch_args):
                    prompt_text = inp['prompt'] if isinstance(inp, dict) else inp
                    try:
                        tok_len = len(tokenizer.encode(prompt_text))
                    except Exception:
                        tok_len = -1
                    if 0 < safe_max < tok_len:
                        skipped.append((args, tok_len))
                    else:
                        to_gen.append(inp)
                        to_gen_args.append(args)
                if skipped:
                    logger.warning(
                        f"[worker {gpu_id}] Pre-filter skipping {len(skipped)} oversize prompts "
                        f"(token lengths: {sorted({s[1] for s in skipped})}, "
                        f"ctx_length={ctx_length}, safe_max={safe_max})"
                    )
                    for args, _ in skipped:
                        output_queue.put(("", *args))

                _generate_and_emit(
                    llm, to_gen, to_gen_args,
                    sampling_params, gpu_id, output_queue,
                )

    except Exception as err:
        logger.error(f"[worker {gpu_id}] Fatal init error: {err}")
        traceback.print_exc()
        time.sleep(5)

def worker_dryrun(input_queue: multiprocessing.Queue, output_queue, gpu_id, batch_size, sample_n, finished_flag, ready_cnt, model_path=MODEL_PATH, ctx_length=CTX_LENGTH):
    ready_cnt.value = ready_cnt.value + 1
    try:
        while not finished_flag.get():
            if input_queue.empty():
                time.sleep(1)
                continue

            batch_inputs = []
            batch_outputs = []
            batch_args = []
            for _ in range(batch_size):
                if not input_queue.empty():
                    try:
                        cur_items = input_queue.get_nowait()
                    except queue.Empty:
                        break
                    except Exception as err:
                        logger.warning("Unexpected error in queue worker", exc_info=True)
                        break
                    batch_inputs.append(cur_items[0])
                    batch_args.append(cur_items[1:-1])
                    batch_outputs.append(cur_items[-1])
                else:
                    break
            
            if batch_inputs:
                time.sleep(0.1)

                for output, args in zip(batch_outputs, batch_args):
                    output_queue.put((output, *args))
    except Exception as err:
        logger.error(f"[worker {gpu_id}] {err}")
        traceback.print_exc()
        time.sleep(5)



class LLMClient:
    def __init__(self, model_path=MODEL_PATH, engine=ENGINE, num_gpus=NUM_GPUS, gpu_range: Optional[List[int]]=None, gpus_per_model=GPUS_PER_MODEL, batch_size=BATCH_SIZE, sample_n: int=1, chat_template: Optional[str]=None, system_prompt: str="You are a helpful assistant.", ctx_length: int=CTX_LENGTH, rope_scaling=None):
        # Create clients
        manager = multiprocessing.Manager()
        self.input_queue = manager.Queue()
        self.output_queue = manager.Queue()
        self.finished_flag = manager.Value('b', False)
        self.ready_cnt = manager.Value('i', 0)
        self.engine = engine
        self.sample_n = sample_n

        # Apply chat template
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.system_prompt = system_prompt

        if chat_template:
            self.tokenizer.chat_template = chat_template

        if gpu_range is None:
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if cuda_visible:
                # Parse "3,4,5,6" into [3, 4, 5, 6]
                gpu_range = [int(x) for x in cuda_visible.split(',')]
            else:
                # Fallback to 0..N
                gpu_range = list(range(num_gpus))

        self.processes = []
        for idx in range(0, len(gpu_range), gpus_per_model):
            gpu_id_str = ",".join(map(str, gpu_range[idx:idx+gpus_per_model]))
            print(f"[LLMClient] Starting worker {idx} on GPU {gpu_id_str}")
            if engine == 'llama':
                p = multiprocessing.Process(target=worker_llama_cpp, args=(self.input_queue, self.output_queue, gpu_id_str, batch_size, sample_n, self.finished_flag, self.ready_cnt, model_path, ctx_length))
            elif engine == 'vllm':
                p = multiprocessing.Process(target=worker_vllm, args=(self.input_queue, self.output_queue, gpu_id_str, batch_size, sample_n, self.finished_flag, self.ready_cnt, model_path, ctx_length, rope_scaling))
            elif engine == 'vllm-ts':
                p = multiprocessing.Process(target=worker_vllm_ts, args=(self.input_queue, self.output_queue, gpu_id_str, batch_size, sample_n, self.finished_flag, self.ready_cnt, model_path, ctx_length))
            elif engine == 'dryrun':
                p = multiprocessing.Process(target=worker_dryrun, args=(self.input_queue, self.output_queue, gpu_id_str, batch_size, sample_n, self.finished_flag, self.ready_cnt, model_path, ctx_length))
            else:
                raise NotImplementedError(f"Unrecognized inference engine: {engine}")
            self.processes.append(p)
            p.start()
        
        print(f"[LLMClient] {len(self.processes)} workers started.")

    def wait_for_ready(self):
        while self.ready_cnt.value < len(self.processes):
            time.sleep(1)
        print(f"[LLMClient] All workers are ready!")

    def _apply_chat_template(self, prompt: str) -> str:
        conversation = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        return self.tokenizer.decode(self.tokenizer.apply_chat_template(conversation, add_generation_prompt=True))
        
    def llm_batch_generate(self, batch_prompts: List[str], batch_timeseries: Optional[List[List[np.ndarray]]] = None, dryrun_outputs: Optional[Union[List[str], List[List[str]]]] = None, use_chat_template=True, sampling_params=None, desc: str = "Generating"):
        if batch_timeseries is not None:
            assert len(batch_prompts) == len(batch_timeseries), f"len(batch_prompts) != len(batch_timeseries): {len(batch_prompts)} != {len(batch_timeseries)}"
            assert self.engine in ['vllm-ts', 'dryrun'], f"Only vllm-ts or dryrun engine supports timeseries data."

        while not self.output_queue.empty():
            self.output_queue.get()
        self.finished_flag.set(False)

        total_cnt = 0

        if dryrun_outputs is not None:
            logger.warning(f"[llm_batch_generate] Dryrun mode. {len(batch_prompts)=}, {len(dryrun_outputs)=}")

        for i, item in enumerate(batch_prompts):
            if use_chat_template:
                inputs = self._apply_chat_template(item)
            else:
                inputs = item

            if batch_timeseries is not None:
                inputs = {
                    "prompt": inputs,
                    "multi_modal_data": {
                        "timeseries": batch_timeseries[i]
                    }
                }
            if dryrun_outputs is not None:
                self.input_queue.put((inputs, i, item, dryrun_outputs[i]))
            elif sampling_params is not None:
                self.input_queue.put((inputs, i, item, sampling_params))
            else:
                self.input_queue.put((inputs, i, item))
            total_cnt += 1

        answer_dict = {}

        with tqdm(total=total_cnt, desc=desc) as pbar:
            while len(answer_dict) < total_cnt:
                line = self.output_queue.get()
                pbar.update()

                answer_dict[line[1]] = line[0]
        
        answer_list = []
        for i in range(len(batch_prompts)):
            if i not in answer_dict:
                answer_list.append(None)
            else:
                answer_list.append(answer_dict[i])

        return answer_list

    def kill(self):
        self.finished_flag.set(True)
        print(f"[LLMClient] Killing workers...")
        time.sleep(5.0)
        for p in self.processes:
            p.join()
        print(f"[LLMClient] All workers have been killed!")


def match_metric_name(metric: str, sentence: str) -> bool:
    pattern = r'[^\u4e00-\u9fa5a-zA-Z]'
    sentence = re.sub(pattern, '', sentence).lower()
    metric = re.sub(pattern, '', metric).lower()

    return metric in sentence


def parse_llm_json(json_string, special_words=None):
    json_string = json_string.replace('```json', '').replace('```', '')
    json_string = repair_json(json_string)
    
    return json.loads(json_string)
