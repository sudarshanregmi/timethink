# TimeThink

Code for **TimeThink: Eliciting Compositional Reasoning in Time-Series Large Language Models.**

TimeThink is a time-series LLM: Qwen3-8B augmented with a time-series encoder. It is post-trained in two stages on a fully synthetic compositional time-series QA corpus. The first stage (`TimeThink-SFT`) is supervised fine-tuning on chain-of-thought targets; the second (`TimeThink-RL`) applies RL with verifiable rewards (GRPO).

## Released artifacts

Models and data are on the Hugging Face Hub:

| Artifact | Hub ID |
|---|---|
| Dataset (all splits) | [`sudarshanregmi/timethink`](https://huggingface.co/datasets/sudarshanregmi/timethink) |
| Base model (pre-SFT init) | [`sudarshanregmi/timethink-base`](https://huggingface.co/sudarshanregmi/timethink-base) |
| SFT model | [`sudarshanregmi/timethink-sft`](https://huggingface.co/sudarshanregmi/timethink-sft) |
| RL model | [`sudarshanregmi/timethink-rl`](https://huggingface.co/sudarshanregmi/timethink-rl) |

The base model is a fresh Qwen3-8B backbone with a time-series encoder ported from [ChatTS-8B](https://huggingface.co/bytedance-research/ChatTS-8B); `scripts/sft.sh` trains from it, and `scripts/build_base_model.py` shows how it was built. The models use a custom `Qwen3TSForCausalLM` architecture, so load them with `trust_remote_code=True` on `transformers>=4.55,<5`:

```python
from transformers import AutoModelForCausalLM, AutoProcessor
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("sudarshanregmi/timethink-rl", trust_remote_code=True)
proc  = AutoProcessor.from_pretrained("sudarshanregmi/timethink-rl", trust_remote_code=True)
ds    = load_dataset("sudarshanregmi/timethink")["test"]
```

## Repository layout

```
synth/             Data-generation pipeline (align/), time-series generators, inference utilities
reward/            Rule-based verifiable reward used by the RL stage
evaluation/        LLM-as-judge evaluation (eval/ + ragas/)
verl/              Vendored + customized RL training framework (GRPO)
scripts/           Training + evaluation runners and analysis utilities
config/            datagen_config.yaml + metric config
ickpt/             Tokenizer + model config/code (for training, populate with timethink-base weights)
preprocess.py      JSONL -> Parquet conversion for verl training
ts_vllm.py         Registers the custom Qwen3TS model with vLLM (imported at inference)
ts_data.py         verl custom dataset class
reward/__init__.py verl custom reward function
```

## Setup

Core dependencies (data generation, evaluation, inference) are pinned in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The RL **training** stack builds on vLLM + a customized [verl](https://github.com/volcengine/verl) (vendored under `verl/`), which additionally needs a Megatron-core / SGLang / flash-attn setup that is hardware-specific and not covered by `requirements.txt`. A convenience installer for the stack used in our runs is provided:

```bash
bash scripts/utils/install_vllm_sglang_mcore.sh
```

## Pipeline

**1. Generate the data.** The training/eval pipeline consumes locally-generated JSONL under `data/`. Producing it requires a local Qwen2.5-32B for QA authoring (set `local_llm_path` in `config/datagen_config.yaml`):

```bash
python -m synth.align      # -> data/*.jsonl (train / val / test splits)
python preprocess.py       # JSONL -> Parquet for verl training
```

The Hub dataset [`sudarshanregmi/timethink`](https://huggingface.co/datasets/sudarshanregmi/timethink) is this same corpus in Parquet for loading via `datasets`. The pipeline scripts read the local JSONL produced above.

**2. Get the base model** to train from. Download the released base into `ickpt/` (which already holds the tokenizer + config):

```bash
hf download sudarshanregmi/timethink-base --local-dir ickpt
```

(Or rebuild it from Qwen3-8B + ChatTS-8B via `python scripts/build_base_model.py` — see that script's header.)

**3. Supervised fine-tuning**, then **RL with verifiable rewards** (GRPO):

```bash
bash scripts/sft.sh    # trains from ickpt -> TimeThink-SFT
bash scripts/rl.sh     # initializes from SFT, -> TimeThink-RL
```

**4. Evaluation** (rule-based reward on the `<think>` block + LLM-judge on the response). Place the checkpoints to score in the repo root, then run — `run_eval.sh` auto-launches the LLM judge (a 72B served via vLLM; see the script header for options):

```bash
hf download sudarshanregmi/timethink-sft --local-dir sft_ckpt
hf download sudarshanregmi/timethink-rl  --local-dir rl_ckpt
bash scripts/run_eval.sh   # scores the models in $MODELS (default: "sft rl")
```

## Acknowledgements

Built on:

- **[ChatTS](https://github.com/NetManAIOps/ChatTS)** (Xie et al.)
- **[verl](https://github.com/volcengine/verl)** (Sheng et al.)

## License

Apache-2.0 (see [`LICENSE`](LICENSE)).

## Citation

```bibtex
@misc{timethink,
  title  = {TimeThink: Eliciting Compositional Reasoning in Time-Series Large Language Models},
  author = {Sudarshan Regmi and Arvind Pillai and Yu Yvonne Wu and Yuliang Chen and Bibek Panthi and Tess Z. Griffin and Michael V. Heinz and Lisa Marsch and Nicholas C. Jacobson and Andrew Campbell},
  year   = {2026}
}
```
