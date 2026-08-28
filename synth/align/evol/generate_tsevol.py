"""Time Series Evolution Dataset Generator for the align pipeline.

Evolves seed QA pairs in-memory via batched LLM rounds, returning a mapping
of seed indices to evolved data. The caller replaces seed results before
splitting and writing to disk.

Each round:
  1. Submit all pool seeds with current strategy catalog (GPU-saturating batch)
  2. Parse evolution results
  3. Submit validation prompts for parsed candidates (GPU-saturating batch)
  4. Accept first-come per strategy; seeds that picked a filled strategy,
     failed validation, or had parse errors go to the retry pool
  5. Narrow catalog to unfilled strategies
  6. Repeat with retry pool

Usage: called from __main__.py when config.dryrun is False.
"""

import json
import re
from loguru import logger
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from json_repair import repair_json

from collections import Counter

from synth.align.evol.evol_prompt import (
    EvolPrompt,
    UNIFIED_EVOL_PROMPT,
    STRATEGY_CATALOG,
    ALL_STRATEGY_NAMES,
    build_strategy_catalog,
    createComparisonEliminatorPrompt,
    createParaphrasePrompt,
    tsevol_constraints,
)
from synth.align.config import Config, SampleResult


MAX_ROUNDS = 10  # Maximum retry rounds before giving up

# Lookup: normalized (lowercase, no spaces/hyphens/underscores) → canonical name
_STRATEGY_LOOKUP: Dict[str, str] = {
    re.sub(r'[\s_-]', '', name.lower()): name
    for name in ALL_STRATEGY_NAMES
}


def _normalize_strategy(raw: str) -> str:
    """Map LLM-reported strategy name to canonical ALL_STRATEGY_NAMES entry."""
    key = re.sub(r'[\s_-]', '', raw.strip().lower())
    return _STRATEGY_LOOKUP.get(key, "unknown")


@dataclass
class EvolConfig:
    data_output_dir: str
    total_cnt: int = 100

    @classmethod
    def from_config(cls, config: "Config") -> "EvolConfig":
        """Construct EvolConfig from a Config object."""
        return cls(
            data_output_dir=config.output_base_dir,
            total_cnt=config.num_data_tsevol,
        )


class AlignEvolPrompt(EvolPrompt):
    """EvolPrompt extended with eval metadata and seed think block."""

    def __init__(
        self,
        eval_type: str,
        eval_metadata: Dict,
        source_split: str = "",
        think_block: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.eval_type = eval_type
        self.eval_metadata = eval_metadata
        self.source_split = source_split
        self.think_block = think_block
        self.strategy_catalog: str = STRATEGY_CATALOG

    def generate_prompt(self):
        seed_qa = json.dumps({
            'question': self.qa_history[-1][0],
            'answer': self.qa_history[-1][1],
        })
        return UNIFIED_EVOL_PROMPT.format(
            think_block=self.think_block,
            strategy_catalog=self.strategy_catalog,
            strategy_hint="",
            seed_qa=seed_qa,
            constraints=tsevol_constraints,
        )

    def generate_comparison_prompt(self, q: str, a: str):
        given_qa = json.dumps({
            'question': self.qa_history[-1][0],
            'answer': self.qa_history[-1][1],
        })
        generated_qa = json.dumps({
            'question': q,
            'answer': a,
        })
        return createComparisonEliminatorPrompt(
            self.think_block, given_qa, generated_qa
        )

    def generate_paraphrase_prompt(self):
        """Paraphrase-only: rewrite the question, keep the seed's rule-based answer.

        No judge validation needed — the answer is never LLM-generated, so the
        reward signal stays verifiable. Used for the RL split.
        """
        return createParaphrasePrompt(
            seed_question=self.qa_history[-1][0],
            think_block=self.think_block,
        )


def _build_prompts_from_results(results: List[SampleResult]) -> List[AlignEvolPrompt]:
    """Build AlignEvolPrompt objects from in-memory SampleResult list."""
    prompts = []
    for idx, result in enumerate(results):
        try:
            # Extract think block from answer
            raw_output = result.answers[0] if result.answers else ""
            think_match = re.search(
                r'(<think>.*?</think>)\s*', raw_output, re.DOTALL
            )
            if think_match:
                think_block = think_match.group(1)
                answer_text = raw_output[think_match.end():].strip()
            else:
                think_block = ""
                answer_text = raw_output.strip()

            eval_type = result.eval_task or (
                result.eval_tasks[0] if result.eval_tasks else "description"
            )

            prompt = AlignEvolPrompt(
                eval_type=eval_type,
                eval_metadata=dict(result.eval_metadata) if result.eval_metadata is not None else {},
                source_split="",
                think_block=think_block,
                ts_idx=idx,
                seed_q=result.questions[0],
                seed_a=answer_text,
                seed_fields=result.fields[0] if result.fields else {},
                instruction=result.base_prompt,
                timeseries=result.encoded_timeseries,
                attribute_pool=result.attributes,
                corr_pool=result.corr_pool,
                metrics=result.metrics,
            )
            prompts.append(prompt)
        except Exception as e:
            logger.warning(f"Failed to build prompt from result {idx}: {e}", exc_info=True)
            continue

    logger.info(f"Built {len(prompts)} evolution prompts from {len(results)} results")
    return prompts


def _run_generation(
    seed_prompts: List[AlignEvolPrompt],
    evol_config: EvolConfig,
    llm_client,
) -> Dict[int, Dict[str, Any]]:
    """Run batched evolution rounds until all strategy quotas are filled.

    Each round submits the entire pool at once (all GPUs saturated), then
    accepts first-come per strategy.  Seeds whose strategy was already full,
    failed validation, or had parse errors go to the retry pool for the next
    round with a narrowed catalog.
    """
    total_target = evol_config.total_cnt
    n_strategies = len(ALL_STRATEGY_NAMES)
    per_strategy_target = total_target // n_strategies

    # Assign seed IDs
    for p in seed_prompts:
        p.seed_id = p.ts_idx

    # State
    available_strategies = set(ALL_STRATEGY_NAMES)
    accepted_counter: Counter = Counter()
    strategy_counter: Counter = Counter()  # all LLM-reported strategies (diagnostics)
    evolved_results: Dict[int, Dict[str, Any]] = {}

    pool = list(seed_prompts)

    logger.info(
        f"Target: {total_target} total, {per_strategy_target} per strategy, "
        f"{len(pool)} seeds available, max {MAX_ROUNDS} rounds"
    )

    for round_num in range(1, MAX_ROUNDS + 1):
        if not pool or not available_strategies:
            break

        # Update catalog for this round (only unfilled strategies)
        catalog = build_strategy_catalog(sorted(available_strategies))
        for p in pool:
            p.strategy_catalog = catalog

        logger.info(
            f"Round {round_num}: {len(pool)} seeds, "
            f"{len(available_strategies)} strategies remaining: "
            f"{sorted(available_strategies)}"
        )

        # Phase 1: Evolution — batch submit all seeds
        evol_texts = [p.generate_prompt() for p in pool]
        evol_results = llm_client.llm_batch_generate(
            evol_texts, desc=f"[EVOL R{round_num}]",
        )

        # Parse evolution results
        candidates = []  # (prompt, question, answer, strategy)
        retry_pool = []

        for i, prompt in enumerate(pool):
            if evol_results[i] is None:
                retry_pool.append(prompt)
                continue
            try:
                qa = json.loads(repair_json(evol_results[i]))
                if not isinstance(qa, dict) or "question" not in qa or "answer" not in qa:
                    retry_pool.append(prompt)
                    continue
                strategy = _normalize_strategy(qa.get("strategy", ""))
                strategy_counter[strategy] += 1
                candidates.append((prompt, qa["question"], qa["answer"], strategy))
            except Exception as e:
                logger.warning(f"Failed to parse TSEvol response: {e}", exc_info=True)
                retry_pool.append(prompt)

        logger.info(
            f"Round {round_num} Phase 1: {len(candidates)} parsed, "
            f"{len(retry_pool)} parse failures (retryable)"
        )

        if not candidates:
            pool = retry_pool
            logger.warning(f"Round {round_num}: no candidates parsed, retrying")
            continue

        # Phase 2: Validation — batch submit all candidates
        val_texts = [
            p.generate_comparison_prompt(q, a)
            for p, q, a, _ in candidates
        ]
        val_results = llm_client.llm_batch_generate(
            val_texts, desc=f"[VALID R{round_num}]",
        )

        # Accept validated, first-come per strategy
        round_accepted = 0
        round_val_fail = 0
        round_strategy_full = 0
        for i, (prompt, q, a, strategy) in enumerate(candidates):
            sid = prompt.seed_id

            if sid in evolved_results:
                continue

            if val_results[i] is None:
                retry_pool.append(prompt)
                round_val_fail += 1
                continue

            response_lower = val_results[i].lower()
            if "valid" not in response_lower or "invalid" in response_lower:
                retry_pool.append(prompt)
                round_val_fail += 1
                continue

            # Strategy filled (either before this round or mid-batch)
            if strategy not in available_strategies or accepted_counter[strategy] >= per_strategy_target:
                retry_pool.append(prompt)
                round_strategy_full += 1
                continue

            # Accept
            full_answer = (prompt.think_block + "\n\n" + a) if prompt.think_block else a
            evolved_results[sid] = {
                "question": q,
                "answer": full_answer,
                "strategy": strategy,
            }
            accepted_counter[strategy] += 1
            round_accepted += 1

            # Narrow if strategy filled
            if accepted_counter[strategy] >= per_strategy_target:
                available_strategies.discard(strategy)
                logger.info(
                    f"Strategy '{strategy}' filled "
                    f"({accepted_counter[strategy]}/{per_strategy_target})"
                )

        logger.info(
            f"Round {round_num} complete: +{round_accepted} accepted, "
            f"{round_val_fail} validation failures, {round_strategy_full} strategy-full drops, "
            f"{len(evolved_results)} total, {len(retry_pool)} in retry pool"
        )

        if round_accepted == 0:
            logger.warning(f"Round {round_num}: no new acceptances, stopping early")
            break

        # Filter retry pool: remove already-accepted seeds
        pool = [p for p in retry_pool if p.seed_id not in evolved_results]

    # Trim to strictly equal across active strategies
    if accepted_counter:
        active_counts = {
            s: accepted_counter[s]
            for s in ALL_STRATEGY_NAMES
            if accepted_counter.get(s, 0) > 0
        }
        if active_counts:
            min_count = min(active_counts.values())
            trimmed_any = any(c > min_count for c in active_counts.values())
            if trimmed_any:
                logger.info(
                    f"Trimming to {min_count} per strategy for strict equality "
                    f"(was: {dict(accepted_counter.most_common())})"
                )
                by_strategy: Dict[str, List[int]] = {}
                for sid, data in evolved_results.items():
                    by_strategy.setdefault(data["strategy"], []).append(sid)
                trimmed = {}
                for strategy, sids in by_strategy.items():
                    for sid in sids[:min_count]:
                        trimmed[sid] = evolved_results[sid]
                evolved_results = trimmed

    # Final logging
    logger.info(f"Evolution complete. Total evolved samples: {len(evolved_results)}")
    if accepted_counter:
        total_acc = sum(accepted_counter.values())
        dist = ", ".join(
            f"{s}: {c} ({100*c/total_acc:.1f}%)"
            for s, c in accepted_counter.most_common()
        )
        logger.info(f"Pre-trim accepted distribution: {dist}")

    final_dist = Counter(d["strategy"] for d in evolved_results.values())
    if final_dist:
        dist = ", ".join(f"{s}: {c}" for s, c in sorted(final_dist.items()))
        logger.info(f"Final strategy distribution: {dist}")

    if strategy_counter:
        total_gen = sum(strategy_counter.values())
        dist = ", ".join(
            f"{s}: {c} ({100*c/total_gen:.1f}%)"
            for s, c in strategy_counter.most_common()
        )
        logger.info(f"LLM strategy picks (all rounds): {dist}")

    return evolved_results


def _run_paraphrase_generation(
    seed_prompts: List[AlignEvolPrompt],
    target_count: int,
    llm_client,
) -> Dict[int, Dict[str, Any]]:
    """Run paraphrase-only evolution.

    LLM rewrites the question; the seed's rule-based answer stays unchanged.
    No judge validation — the GT verdict never changes, so no label noise
    can be introduced. One LLM call per sample (vs. 2 for full evolution).

    The returned dict preserves seed eval_type routing (not retagged as
    "tsevol") — callers must keep the original eval_type when replacing
    seeds so rule-based reward scorers still fire correctly.
    """
    if not seed_prompts or target_count <= 0:
        return {}

    pool = list(seed_prompts)
    evolved: Dict[int, Dict[str, Any]] = {}

    for p in pool:
        p.seed_id = p.ts_idx

    logger.info(
        f"Paraphrase mode: target={target_count}, pool={len(pool)}, "
        f"max {MAX_ROUNDS} rounds"
    )

    for round_num in range(1, MAX_ROUNDS + 1):
        if not pool or len(evolved) >= target_count:
            break

        needed = target_count - len(evolved)
        batch = pool[:needed * 2]  # over-submit to absorb parse failures

        texts = [p.generate_paraphrase_prompt() for p in batch]
        results = llm_client.llm_batch_generate(
            texts, desc=f"[PARAPHRASE R{round_num}]"
        )

        retry_pool = []
        round_accepted = 0

        for i, prompt in enumerate(batch):
            if prompt.seed_id in evolved:
                continue
            if len(evolved) >= target_count:
                break

            if results[i] is None:
                retry_pool.append(prompt)
                continue

            try:
                obj = json.loads(repair_json(results[i]))
            except Exception as e:
                logger.warning(f"Paraphrase parse failed: {e}", exc_info=True)
                retry_pool.append(prompt)
                continue

            if not isinstance(obj, dict) or "question" not in obj:
                retry_pool.append(prompt)
                continue

            new_q = str(obj["question"]).strip()
            if not new_q:
                retry_pool.append(prompt)
                continue

            # Keep seed answer verbatim (think block + rule-based answer line).
            # The caller re-assembles `think_block + "\n" + answer_text` when
            # rebuilding the SampleResult, matching the seed's output shape.
            seed_answer = prompt.qa_history[-1][1]
            full_answer = (
                prompt.think_block + "\n\n" + seed_answer
                if prompt.think_block else seed_answer
            )
            evolved[prompt.seed_id] = {
                "question": new_q,
                "answer": full_answer,
                "strategy": "paraphrase",
            }
            round_accepted += 1

        logger.info(
            f"Paraphrase R{round_num}: +{round_accepted} accepted, "
            f"{len(evolved)}/{target_count} total"
        )

        if round_accepted == 0:
            logger.warning("Paraphrase: no acceptances this round, stopping")
            break

        # Drop accepted, keep unaccepted + over-submitted tail for next round
        pool = [p for p in batch[needed * 2:] + retry_pool if p.seed_id not in evolved]

    logger.info(f"Paraphrase complete: {len(evolved)} samples produced")
    return evolved


def evol_instruct(
    config: Config,
    results: List[SampleResult],
    llm_client=None,
    target_count: int = None,
) -> Dict[int, Dict[str, Any]]:
    """Evolve seed QA pairs in-memory, returning seed_id → evolved data mapping.

    Args:
        target_count: override for number of seeds to evolve. If None, uses
            config.num_data_tsevol. Pass a smaller value to run full-evolution
            on a subset of seeds (e.g., only half of the SFT split's allotment).

    Stub mode: when `llm_client is None` (e.g. `--dryrun`), skip the LLM
    calls and instead return a structural-probe result: picks `target_count`
    seeds, assigns random strategies (distributed evenly across the 7 full-
    evolution strategies), keeps Q/A unchanged. Downstream `_apply_full_evol`
    still retags `eval_type="tsevol"` and sets tsevol=True, so per-split
    counts / routing / TSEvol-RL-isolation can be validated at production
    scale without paying LLM compute.
    """
    import random
    random.seed(42)
    np.random.seed(42)

    evol_config = EvolConfig.from_config(config)
    if target_count is not None:
        evol_config.total_cnt = target_count
    n_strategies = len(ALL_STRATEGY_NAMES)
    per_strategy = evol_config.total_cnt // n_strategies

    if llm_client is None:
        # Stub mode — structural bookkeeping only, no LLM rewrite
        total = min(evol_config.total_cnt, len(results))
        if total <= 0:
            return {}
        picks = random.sample(range(len(results)), total)
        stubbed: Dict[int, Dict[str, Any]] = {}
        for k, seed_idx in enumerate(picks):
            r = results[seed_idx]
            stubbed[seed_idx] = {
                "question": r.questions[0] if r.questions else "",
                "answer": r.answers[0] if r.answers else "",
                "strategy": ALL_STRATEGY_NAMES[k % n_strategies],
            }
        logger.info(
            f"Stub-mode full evolution: picked {len(stubbed)} seeds, "
            f"~{per_strategy} per strategy × {n_strategies} strategies "
            f"(no LLM calls — questions/answers unchanged)"
        )
        return stubbed

    logger.info("Starting TS Evolution...")
    logger.info(
        f"  Target: {evol_config.total_cnt} total, "
        f"{per_strategy} per strategy x {n_strategies} strategies"
    )

    prompts = _build_prompts_from_results(results)
    if not prompts:
        logger.error("No evolution prompts built. Aborting.")
        return {}

    random.shuffle(prompts)
    logger.info(f"Total seed prompts: {len(prompts)}")

    evolved = _run_generation(prompts, evol_config, llm_client)
    logger.info(f"TS Evolution complete! {len(evolved)} seeds replaced.")
    return evolved


def evol_instruct_paraphrase(
    results: List[SampleResult],
    target_count: int,
    llm_client=None,
) -> Dict[int, Dict[str, Any]]:
    """Paraphrase-only evolution entry point.

    Rewrites the SEED question in different words; keeps the seed's rule-based
    answer byte-for-byte. Use this for the RL split so the reward signal stays
    verifiable (no LLM-generated labels) — and optionally for part of the SFT
    split for cheap phrasing diversity.

    Unlike `evol_instruct`, the returned samples retain their ORIGINAL
    eval_type — callers MUST keep the seed's eval_type when replacing seeds
    so rule-based reward scorers still route correctly.

    Stub mode: when `llm_client is None` (e.g. `--dryrun`), pick
    `target_count` seeds and tag them as paraphrased in metadata without
    actually calling the LLM. Lets the full production pipeline run at
    scale to validate per-split TSEvol routing, `eval_type` preservation,
    and RL TSEvol isolation without paying LLM compute.
    """
    if target_count <= 0:
        return {}

    import random
    random.seed(42)
    np.random.seed(42)

    if llm_client is None:
        # Stub mode — structural bookkeeping only
        total = min(target_count, len(results))
        if total <= 0:
            return {}
        picks = random.sample(range(len(results)), total)
        stubbed: Dict[int, Dict[str, Any]] = {
            seed_idx: {
                "question": results[seed_idx].questions[0] if results[seed_idx].questions else "",
                "answer": results[seed_idx].answers[0] if results[seed_idx].answers else "",
                "strategy": "paraphrase",
            }
            for seed_idx in picks
        }
        logger.info(
            f"Stub-mode paraphrase: picked {len(stubbed)} seeds "
            f"(no LLM calls — questions unchanged, tagged as paraphrased)"
        )
        return stubbed

    prompts = _build_prompts_from_results(results)
    if not prompts:
        logger.error("No paraphrase prompts built. Aborting.")
        return {}

    random.shuffle(prompts)
    logger.info(
        f"Starting paraphrase-only evolution: "
        f"target={target_count}, seed pool={len(prompts)}"
    )

    evolved = _run_paraphrase_generation(prompts, target_count, llm_client)
    logger.info(f"Paraphrase evolution complete! {len(evolved)} seeds replaced.")
    return evolved
