"""Dataset generation and management."""
import copy
from dataclasses import replace
import json
import multiprocessing as mp
import os
import concurrent.futures
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import random

from loguru import logger
from tqdm import tqdm

from synth.ts_generator.utils.common_utils import (
    NumpyEncoder,
    determine_sequence_length,
)
from synth.align.utils import timeseries_to_list
from synth.utils.llm_utils import LLMClient

from synth.align.config import (
    Config,
    Mode,
    DEFAULT_MODE_WEIGHTS,
    Difficulty,
    PromptIndexer,
    SampleResult,
)
from synth.align.factory import generate_sample
from synth.align.utils import replace_prompts_in_obj


# ---------------------------------------------------------------------------
# Debug --filter definitions
# ---------------------------------------------------------------------------

# Group aliases: --filter <name> expands to these eval_types.
# Exact eval_type names (e.g. "stat_numerical") also work.
FILTER_GROUPS: Dict[str, frozenset] = {
    'correlation':       frozenset({'correlation', 'anticorrelation'}),
    'correlation_local': frozenset({'correlation'}),
    'correlation_shape': frozenset({'correlation', 'anticorrelation'}),
    'clustering':        frozenset({'clustering', 'anticlustering'}),
    'clustering_local':  frozenset({'clustering'}),
    'clustering_shape':  frozenset({'clustering', 'anticlustering'}),
    'judgment':          frozenset({'segment_judgment', 'cross_stat_judgment',
                                    'anti_judgment'}),
    'trend':             frozenset({'segment_trend_dominance', 'cross_trend_query'}),
    'stat':              frozenset({'stat_numerical', 'periodicity',
                                    'change_point'}),
    'enumeration':       frozenset({'local_enumeration', 'segment_enumeration',
                                    'cross_metric_enumeration',
                                    'transition_enumeration',
                                    'event_segment_enumeration',
                                    'temporal_position', 'duration_proportion'}),
    'description':       frozenset({'description'}),
    'yes_no':            frozenset({'yes_no'}),
    'tsevol':            frozenset({'tsevol'}),
    'ood':               frozenset({
                                    'ood_conditional_stat', 'ood_nested_extrema',
                                    'ood_event_density', 'ood_conditional_count',
                                    'ood_trend_reversal', 'ood_range_normalized_amplitude',
                                    'ood_segment_stat_compare',
                                    # Single-metric OOD taxonomy
                                    'ood_max_before_min',
                                    'ood_std_exceeds_half_range', 'ood_max_in_highest_mean_quarter',
                                    'ood_quarter_mean_ordering',
                                    'ood_max_mean_chunk_pos', 'ood_chunk_above_proportion',
                                    'ood_symmetric_recovery', 'ood_cycle_mean_trend',
                                    'ood_trend_follows_mean', 'ood_conditional_mean_by_type',
                                    'ood_longest_type_fraction', 'ood_event_amplitude_vs_std',
                                    'ood_amplitude_vs_segment_std', 'ood_symmetric_trend_sequence',
                                    'ood_event_density_by_trend', 'ood_max_amp_in_longest_segment',
                                    # Cross-metric OOD
                                    'ood_cross_corr_count', 'ood_cross_trend_convergence',
                                    'ood_cross_extrema_alignment',
                                    'ood_cross_range_overlap', 'ood_cross_concordant_shift',
                                    'ood_cluster_singleton', 'ood_corr_transitivity',
                                    'ood_mixed_corr_anti',
                                    # Compositional probes (reverse-engineered, eval-only)
                                    'ood_peak_in_longest_segment', 'ood_duration_weighted_mean',
                                    'ood_dominant_trend_in_cluster', 'ood_cross_event_causality'}),
    'single_metric_taxonomy': frozenset({
                                    # Atomic SFT
                                    'atomic_global_mean', 'atomic_global_std',
                                    'atomic_min_value', 'atomic_max_value',
                                    'atomic_min_position', 'atomic_max_position',
                                    'atomic_percentile', 'atomic_interval_mean',
                                    'atomic_interval_std', 'atomic_chunked_means',
                                    'atomic_trend_enumeration', 'atomic_event_enumeration',
                                    'atomic_periodic_description',
                                    # RL compositions
                                    'rl_amplitude_vs_range', 'rl_amplitude_vs_std',
                                    'rl_condition_recovery', 'rl_cycle_count',
                                    'rl_dominant_trend_type', 'rl_event_count',
                                    'rl_event_count_by_type', 'rl_event_in_trend_type',
                                    'rl_event_near_extremum', 'rl_event_type_at_pos',
                                    'rl_extrema_same_half', 'rl_half_mean_compare',
                                    'rl_half_mean_diff', 'rl_has_periodicity',
                                    'rl_interval_comparison', 'rl_longest_segment',
                                    'rl_max_amplitude_event', 'rl_max_in_first_half',
                                    'rl_max_in_trend_type', 'rl_mean_shift',
                                    'rl_mean_stability', 'rl_median_mean_close',
                                    'rl_monotonic_chunks', 'rl_normalized_range',
                                    'rl_period_estimate', 'rl_range',
                                    'rl_segment_count', 'rl_segment_duration',
                                    'rl_segment_mean_compare', 'rl_segment_type_at_pos',
                                    'rl_type_duration_fraction', 'rl_type_of_longest',
                                    'rl_volatility_change'}),
    'cross_metric_taxonomy': frozenset({
                                    'atomic_cross_stat_compare', 'atomic_cross_ranking',
                                    'atomic_cross_filtering', 'atomic_cross_counting',
                                    'atomic_cross_trend_align',
                                    'rl_cross_stat_ratio', 'rl_cross_full_ordering',
                                    'rl_cross_event_sync', 'rl_cross_period_compare',
                                    'rl_cross_trend_concordance', 'rl_cross_conditional_query',
                                    'rl_cross_attribute_corr', 'rl_cross_asymmetric_behavior',
                                    'rl_cluster_count', 'rl_cluster_dominant',
                                    'rl_corr_count', 'rl_corr_conditional',
                                    'ood_cross_corr_count', 'ood_cross_trend_convergence',
                                    'ood_cross_extrema_alignment',
                                    'ood_cross_range_overlap', 'ood_cross_concordant_shift',
                                    'ood_cluster_singleton', 'ood_corr_transitivity',
                                    'ood_mixed_corr_anti',
                                    # Compositional cross-metric probes
                                    'ood_dominant_trend_in_cluster', 'ood_cross_event_causality'}),
}

# Groups that restrict output to specific modes (e.g. only local or only shape).
FILTER_MODE_RESTRICT: Dict[str, frozenset] = {
    'correlation_local': frozenset({Mode.MTS_LOCAL}),
    'correlation_shape': frozenset({Mode.MTS_SHAPE}),
    'clustering_local':  frozenset({Mode.MTS_LOCAL}),
    'clustering_shape':  frozenset({Mode.MTS_SHAPE}),
}

# Union of every known eval_type (for accepting exact names in --filter).
_ALL_EVAL_TYPES: frozenset = frozenset().union(*FILTER_GROUPS.values())

# ---------------------------------------------------------------------------
# Top-up: reverse mapping eval_type → generation strategy
# ---------------------------------------------------------------------------
# Each entry: (allowed_modes, qa_type_to_boost)

_TOPUP_STRATEGY: Dict[str, Tuple[List[Mode], str]] = {}

# QA types that map 1:1 to eval_type
_TOPUP_STRATEGY['description'] = ([Mode.UTS], 'description')
_TOPUP_STRATEGY['yes_no'] = ([Mode.UTS], 'yes_no')
_TOPUP_STRATEGY['correlation'] = ([Mode.MTS_LOCAL, Mode.MTS_SHAPE], 'correlation')
_TOPUP_STRATEGY['anticorrelation'] = ([Mode.MTS_SHAPE], 'anticorrelation')
_TOPUP_STRATEGY['clustering'] = ([Mode.MTS_SHAPE], 'clustering')
_TOPUP_STRATEGY['anticlustering'] = ([Mode.MTS_SHAPE], 'anticlustering')

# Segment-mask subtypes generated in UTS (and MTS_LOCAL per-metric)
for _et in ('stat_numerical', 'periodicity', 'change_point',
            'segment_judgment', 'segment_trend_dominance',
            'local_enumeration', 'segment_enumeration',
            'transition_enumeration', 'event_segment_enumeration',
            'temporal_position', 'duration_proportion'):
    _TOPUP_STRATEGY[_et] = ([Mode.UTS], 'segment_mask')

# Cross-metric types (MTS only)
_TOPUP_STRATEGY['cross_stat_judgment'] = ([Mode.MTS_LOCAL], 'segment_mask')
_TOPUP_STRATEGY['cross_metric_enumeration'] = ([Mode.MTS_LOCAL], 'segment_mask')
_TOPUP_STRATEGY['cross_trend_query'] = ([Mode.MTS_SHAPE], 'segment_mask')
_TOPUP_STRATEGY['anti_judgment'] = ([Mode.MTS_SHAPE], 'segment_mask')

# Types that are generated post-hoc or eval-only (skip in top-up)
_TOPUP_SKIP = frozenset(
    et for et in _ALL_EVAL_TYPES
    if et.startswith('ood_')
) | {'tsevol'}


def resolve_debug_filter(
    filter_names: List[str],
) -> Tuple[frozenset, Dict[str, frozenset]]:
    """Resolve --filter names into (allowed_eval_types, mode_restrictions).

    mode_restrictions maps eval_type -> set of allowed modes.
    If an eval_type has no entry, all modes are allowed for it.
    """
    allowed: Set[str] = set()
    mode_restrictions: Dict[str, Set[Mode]] = {}

    for name in filter_names:
        if name in FILTER_GROUPS:
            types = FILTER_GROUPS[name]
            allowed |= types
            if name in FILTER_MODE_RESTRICT:
                for t in types:
                    mode_restrictions.setdefault(t, set()).update(
                        FILTER_MODE_RESTRICT[name]
                    )
        elif name in _ALL_EVAL_TYPES:
            allowed.add(name)
        else:
            valid = sorted(set(FILTER_GROUPS.keys()) | _ALL_EVAL_TYPES)
            raise ValueError(
                f"Unknown filter '{name}'. Valid filters:\n  "
                + "\n  ".join(valid)
            )

    return frozenset(allowed), mode_restrictions


def _call_llm(config: Config, prompts: List[str], dryrun_value: str, llm_client=None) -> List[str]:
    """Shared LLM batch-generation helper.

    Args:
        config: Pipeline config (provides local_llm_path and dryrun flag).
        prompts: List of prompts to send to the LLM.
        dryrun_value: Placeholder answer returned for every prompt in dryrun mode.
        llm_client: Optional shared LLMClient. When provided, reuses it (no create/kill).

    Returns:
        List of answer strings, one per prompt.
    """
    if not prompts:
        return []
    if config.dryrun:
        return [dryrun_value] * len(prompts)
    if llm_client is not None:
        return llm_client.llm_batch_generate(prompts, use_chat_template=True)
    # Fallback: create a temporary client when no shared client is provided
    tmp_client = LLMClient(model_path=config.local_llm_path, engine='vllm', ctx_length=config.ctx_length)
    try:
        return tmp_client.llm_batch_generate(prompts, use_chat_template=True)
    finally:
        tmp_client.kill()


# --- ProcessPoolExecutor worker globals (set once per worker via initializer) ---
_worker_configs = {}


def _init_sample_worker(base_config, clustering_config, anticlustering_config):
    """Store configs in worker process memory (called once per worker, not per task)."""
    global _worker_configs
    _worker_configs = {
        'base': base_config,
        'clustering': clustering_config,
        'anticlustering': anticlustering_config,
    }


def sample_worker(args: Tuple):
    """Generate one sample. Returns SampleResult on success, None on failure.

    Catches exceptions inside the worker so one bad sample doesn't kill
    the entire imap_unordered chunk (chunksize=200 → 199 lost otherwise).
    """
    try:
        mode, config_key, seq_len, difficulty, target_cluster_size, target_anticluster_size = args
        config = _worker_configs[config_key]
        indexer = PromptIndexer()
        return generate_sample(
            mode=mode,
            config=config,
            indexer=indexer,
            seq_len=seq_len,
            difficulty=difficulty,
            target_cluster_size=target_cluster_size,
            target_anticluster_size=target_anticluster_size,
        )
    except Exception as e:
        logger.warning(f"Sample generation failed: {e}", exc_info=True)
        return None


def _debug_subtype_key(
    eval_task: str,
    eval_metadata: dict,
    mode_value: str,
) -> str:
    """Build a composite key that uniquely identifies a question sub-type.

    Discriminating fields used (from eval_metadata):
      sub_type       — stat_numerical family (10 values)
      judgment_type  — segment_judgment family (trend_local, multi_phase, …)
      variant        — yes_no family in mts_local (generic, typed, end_pos, property, param)
      end_pos        — clustering variant (bool)
      type_aware     — clustering variant (bool)

    Mode is prepended so that e.g. mts_local|correlation and mts_shape|correlation
    are kept as separate samples (different thinking formats).
    """
    parts = [mode_value, str(eval_task)]
    for field in ("sub_type", "judgment_type", "variant"):
        v = eval_metadata.get(field)
        if v is not None and v != "":
            parts.append(str(v))
    end_pos = eval_metadata.get("end_pos")
    type_aware = eval_metadata.get("type_aware")
    if end_pos is not None:
        parts.append(f"ep={end_pos}")
    if type_aware is not None:
        parts.append(f"ta={type_aware}")
    return "|".join(parts)


class DatasetGenerator:

    def __init__(self, config: Config, llm_client=None):
        self.config = config
        self.llm_client = llm_client

    def _allocate_tasks(self) -> List[Mode]:
        """Allocate seed tasks proportionally to modes using mode_weights."""
        weights = self.config.mode_weights or DEFAULT_MODE_WEIGHTS
        total = self.config.num_data

        # Normalize weights
        w_sum = sum(weights.values())
        if w_sum == 0:
            w_sum = 1.0

        # Compute counts per mode (floor), then distribute remainder
        raw = {k: v / w_sum * total for k, v in weights.items()}
        counts = {k: int(v) for k, v in raw.items()}
        remainder = total - sum(counts.values())
        # Give remainder to modes with highest fractional parts
        fracs = sorted(weights.keys(), key=lambda k: raw[k] - counts[k], reverse=True)
        for i in range(remainder):
            counts[fracs[i % len(fracs)]] += 1

        mode_map = {"uts": Mode.UTS, "mts_local": Mode.MTS_LOCAL, "mts_shape": Mode.MTS_SHAPE}
        tasks = []
        for key, count in counts.items():
            mode = mode_map.get(key)
            if mode:
                tasks.extend([mode] * count)

        return tasks

    def _flatten_llm_prompts(self, results: List[SampleResult]) -> List[str]:
        all_prompts = []
        for result in results:
            for prompt_group in result.llm_prompts:
                all_prompts.extend(prompt_group)
        return all_prompts

    def _get_llm_answers(self, prompts: List[str]) -> List[str]:
        return _call_llm(self.config, prompts, dryrun_value='This is a test answer.', llm_client=self.llm_client)

    def _replace_all_placeholders(
        self,
        results: List[SampleResult],
        llm_answers: List[str]
    ) -> None:
        global_ans_idx = 0

        for result in results:
            item_replacements = {}
            current_local_prompt_idx = 0

            for prompt_group in result.llm_prompts:
                for _ in prompt_group:
                    if global_ans_idx < len(llm_answers):
                        placeholder = f"<|prompt{current_local_prompt_idx}|>"
                        item_replacements[placeholder] = llm_answers[global_ans_idx]
                        global_ans_idx += 1
                        current_local_prompt_idx += 1

            for i in range(len(result.answers)):
                result.answers[i] = replace_prompts_in_obj(result.answers[i], item_replacements)

            for i in range(len(result.corr_pool)):
                if result.corr_pool[i]:
                    if isinstance(result.corr_pool[i], list) and len(result.corr_pool[i]) > 1:
                        result.corr_pool[i][1] = replace_prompts_in_obj(
                            result.corr_pool[i][1], item_replacements
                        )

            result.label = replace_prompts_in_obj(result.label, item_replacements)

    def _flatten_debug_results(self, raw_results: List[SampleResult]) -> List[SampleResult]:
        """Expand multi-QA SampleResults into one SampleResult per Q&A pair.

        After LLM fill-in, answers contain no placeholders, so splitting is safe.
        """
        flat = []
        for result in raw_results:
            for i in range(len(result.questions)):
                qt = result.qa_types[i] if result.qa_types and i < len(result.qa_types) else None
                et = result.eval_tasks[i] if result.eval_tasks and i < len(result.eval_tasks) else None
                em = result.eval_metadatas[i] if result.eval_metadatas and i < len(result.eval_metadatas) else {}
                fd = result.fields[i] if result.fields and i < len(result.fields) else {}

                single = SampleResult(
                    mode=result.mode,
                    original_timeseries=result.original_timeseries,
                    encoded_timeseries=result.encoded_timeseries,
                    metrics=result.metrics,
                    attributes=result.attributes,
                    base_prompt=result.base_prompt,
                    questions=[result.questions[i]],
                    answers=[result.answers[i]],
                    llm_prompts=[[]],  # already filled in
                    fields=[fd],
                    corr_pool=result.corr_pool,
                    label=result.label,
                    qa_types=[qt] if qt is not None else [],
                    eval_tasks=[et] if et is not None else [],
                    eval_metadatas=[em],
                    eval_task=et,
                    eval_metadata=em,
                    situation=result.situation,
                )
                flat.append(single)
        return flat

    @staticmethod
    def _log_distribution_stats(results: List[SampleResult]) -> None:
        """Print mode / qa_type / eval_type distribution stats."""
        from collections import Counter
        total = len(results)
        if total == 0:
            print("[STATS] No results to report.")
            return

        mode_counts = Counter(r.mode.value for r in results)
        qa_counts = Counter(
            r.qa_types[0] if r.qa_types else "unknown" for r in results
        )
        eval_counts = Counter(
            r.eval_task if r.eval_task else (r.eval_tasks[0] if r.eval_tasks else "unknown")
            for r in results
        )

        print(f"\n{'='*60}")
        print(f"  Distribution Stats ({total} samples)")
        print(f"{'='*60}")

        print("\n  Mode distribution:")
        for mode in sorted(mode_counts.keys()):
            c = mode_counts[mode]
            print(f"    {mode:15s}: {c:5d}  ({100*c/total:.1f}%)")

        print("\n  QA type distribution:")
        for qt in sorted(qa_counts.keys()):
            c = qa_counts[qt]
            print(f"    {qt:25s}: {c:5d}  ({100*c/total:.1f}%)")

        print("\n  Eval type distribution:")
        for et in sorted(eval_counts.keys()):
            c = eval_counts[et]
            print(f"    {et:35s}: {c:5d}  ({100*c/total:.1f}%)")

        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Top-up: fill deficit eval_types to minimum sample count
    # ------------------------------------------------------------------

    def _topup_deficit_types(self, results: List[SampleResult]) -> None:
        """Generate extra samples for eval_types below min_samples_per_eval_type.

        Appends directly to *results*. Runs after the main pool generation
        but before LLM prompt collection, so top-up samples get their
        placeholders filled alongside the originals.
        """
        min_count = self.config.min_samples_per_eval_type
        if min_count <= 0:
            return

        from collections import Counter, defaultdict

        # Current distribution
        eval_counts = Counter(
            r.eval_task or (r.eval_tasks[0] if r.eval_tasks else None)
            for r in results
        )

        # Identify deficit types (skip OOD / tsevol — generated separately)
        deficits: Dict[str, int] = {}
        for et in _ALL_EVAL_TYPES - _TOPUP_SKIP:
            needed = min_count - eval_counts.get(et, 0)
            if needed > 0 and et in _TOPUP_STRATEGY:
                deficits[et] = needed

        if not deficits:
            print(f"[TOP-UP] All eval_types >= {min_count} samples. No top-up needed.")
            return

        total_deficit = sum(deficits.values())
        print(f"\n[TOP-UP] {len(deficits)} eval_types below minimum ({min_count}), "
              f"total deficit: {total_deficit}")
        for et in sorted(deficits):
            print(f"  {et:35s}: have {eval_counts.get(et, 0):5d}, need +{deficits[et]}")

        # Group deficits by generation strategy for batch efficiency.
        # Key: (tuple(mode_values), qa_type)
        groups: Dict[tuple, Dict[str, int]] = defaultdict(dict)
        for et, needed in deficits.items():
            modes, qa_type = _TOPUP_STRATEGY[et]
            key = (tuple(sorted(m.value for m in modes)), qa_type)
            groups[key][et] = needed

        difficulties = list(Difficulty)

        for (mode_vals, qa_type), group_deficits in groups.items():
            modes = [Mode(v) for v in mode_vals]
            label = f"{'+'.join(mode_vals)}/{qa_type}"
            total_needed = sum(group_deficits.values())

            # Build targeted config: heavy weights toward deficit types
            overrides = {
                'eval_type_weights': {et: 100.0 for et in group_deficits},
                'qa_type_weights': {qa_type: 100.0},
            }

            target_config = replace(self.config, **overrides)

            remaining = dict(group_deficits)
            max_attempts = total_needed * 10
            generated = 0

            for attempt in range(max_attempts):
                mode = random.choice(modes)
                seq_len = determine_sequence_length(max_len=self.config.main_pool_max_seq_len)
                difficulty = random.choice(difficulties)
                indexer = PromptIndexer()
                try:
                    result = generate_sample(
                        mode=mode,
                        config=target_config,
                        indexer=indexer,
                        seq_len=seq_len,
                        difficulty=difficulty,
                    )
                    et = result.eval_task
                    if et in remaining and remaining[et] > 0:
                        results.append(result)
                        remaining[et] -= 1
                        generated += 1
                        if all(v <= 0 for v in remaining.values()):
                            break
                except Exception as e:
                    logger.warning(f"[TOP-UP] {mode.value} sample failed: {e}", exc_info=True)
                    continue

            still_short = {et: n for et, n in remaining.items() if n > 0}
            msg = f"  [{label}] +{generated} samples in {attempt + 1} attempts"
            if still_short:
                msg += f" (still short: {still_short})"
            print(msg)

        new_total = len(results)
        print(f"[TOP-UP] Done. {new_total - sum(eval_counts.values())} extra samples "
              f"added (total: {new_total})")

    def _debug_pass(
        self,
        modes: List[Mode],
        seen_keys: Set[str],
        kept: List[SampleResult],
        config_override: Config = None,
        seq_len_override: int = None,
    ) -> int:
        """Run one debug pass: generate samples, flatten, collect new sub-types.

        Returns the number of new sub-types found.
        """
        cfg = config_override or self.config
        raw = []
        failed_modes = []
        for mode in modes:
            seq_len = seq_len_override or determine_sequence_length(
                max_len=cfg.main_pool_max_seq_len
            )
            indexer = PromptIndexer()
            try:
                result = generate_sample(
                    mode=mode,
                    config=cfg,
                    indexer=indexer,
                    seq_len=seq_len,
                    difficulty=Difficulty.EASY,
                    debug=True,
                )
                raw.append(result)
            except Exception as e:
                failed_modes.append(mode.value)
                logger.warning(f"[DEBUG] {mode.value} generation failed (seq_len={seq_len}): {e}", exc_info=True)
        if failed_modes:
            print(f"  [DEBUG] WARNING: failed modes this pass: {failed_modes}")

        prompts = self._flatten_llm_prompts(raw)
        answers = self._get_llm_answers(prompts)
        self._replace_all_placeholders(raw, answers)

        flat = self._flatten_debug_results(raw)
        new_count = 0
        for item in flat:
            et = item.eval_tasks[0] if item.eval_tasks else (item.qa_types[0] if item.qa_types else "unknown")
            em = item.eval_metadatas[0] if item.eval_metadatas else {}
            key = _debug_subtype_key(et, em, item.mode.value)
            if key not in seen_keys:
                seen_keys.add(key)
                kept.append(item)
                new_count += 1
        return new_count

    def _debug_generate(self) -> List[SampleResult]:
        """Debug mode: exactly one sample per unique question sub-type.

        Exhaustively generates samples across all modes, flattens to individual
        QA pairs, and deduplicates by (mode, eval_task, discriminators).  Stops
        when no new sub-type is found for PATIENCE consecutive passes or
        MAX_PASSES total.

        When ``config.debug_filter`` is set, only matching eval_types are kept
        and generation is scoped to the modes that can produce them.
        """
        MAX_PASSES = 100
        PATIENCE = 10

        # ── resolve filter ──────────────────────────────────────────────
        allowed_types: frozenset = None          # None = no filter, keep all
        mode_restrictions: Dict[str, frozenset] = {}

        if self.config.debug_filter:
            allowed_types, mode_restrictions = resolve_debug_filter(
                self.config.debug_filter
            )
            print(f"[DEBUG] Filter active — keeping eval_types: {sorted(allowed_types)}")

        print("[DEBUG] Generating one sample per unique question sub-type.")

        debug_modes = [Mode.UTS, Mode.MTS_LOCAL, Mode.MTS_SHAPE]
        seen_keys: set = set()
        kept: List[SampleResult] = []

        # ── main exhaustive loop ────────────────────────────────────────
        no_new_streak = 0
        for pass_num in range(1, MAX_PASSES + 1):
            new_count = self._debug_pass(debug_modes, seen_keys, kept)

            print(f"  [DEBUG] Pass {pass_num}: +{new_count} new sub-types (total: {len(seen_keys)})")
            if new_count == 0:
                no_new_streak += 1
                if no_new_streak >= PATIENCE:
                    print(f"  [DEBUG] No new sub-types for {PATIENCE} passes. Done.")
                    break
            else:
                no_new_streak = 0

        # ── apply filter ────────────────────────────────────────────────
        if allowed_types is not None:
            before = len(kept)
            filtered = []
            for item in kept:
                et = item.eval_tasks[0] if item.eval_tasks else None
                if et not in allowed_types:
                    continue
                if et in mode_restrictions and item.mode not in mode_restrictions[et]:
                    continue
                filtered.append(item)
            kept = filtered
            print(f"[DEBUG] Filter: {before} → {len(kept)} samples")

        print(f"[DEBUG] Sub-types found ({len(kept)}):")
        for item in sorted(kept, key=lambda r: _debug_subtype_key(
            r.eval_tasks[0] if r.eval_tasks else "unknown",
            r.eval_metadatas[0] if r.eval_metadatas else {},
            r.mode.value,
        )):
            et = item.eval_tasks[0] if item.eval_tasks else "unknown"
            em = item.eval_metadatas[0] if item.eval_metadatas else {}
            print(f"  {_debug_subtype_key(et, em, item.mode.value)}")
        print(f"[DEBUG] Total output samples: {len(kept)}")
        return kept

    def generate(self) -> List[SampleResult]:
        if self.config.debug:
            results = self._debug_generate()
            self._log_distribution_stats(results)
            return results

        results = []

        allocated = self._allocate_tasks()
        mode_summary = {}
        for mode in allocated:
            mode_summary[mode.value] = mode_summary.get(mode.value, 0) + 1
        print(f"Task allocation ({len(allocated)} total):")
        for m, c in sorted(mode_summary.items()):
            print(f"  {m}: {c} ({100*c/len(allocated):.1f}%)")

        difficulties = list(Difficulty)

        # --- Clustering GT-length balancing ---
        # Reserve ~30% of MTS_SHAPE tasks for clustering with controlled
        # positive-cluster sizes (1..11 → GT Len 0..10).
        shape_indices = [i for i, m in enumerate(allocated) if m == Mode.MTS_SHAPE]
        n_clustering = len(shape_indices) * 3 // 10
        random.shuffle(shape_indices)
        clustering_indices = set(shape_indices[:n_clustering])

        clustering_config = replace(self.config, qa_type_weights={"clustering": 1.0})

        cluster_sizes = list(range(1, 12))  # 1..11 → GT Len 0..10
        cs_idx = 0

        # --- Anticlustering GT-length balancing ---
        # Reserve ~20% of remaining MTS_SHAPE tasks for anticlustering with
        # controlled anti-trend counts (0..5 → GT Len 0..5).
        remaining_shape = [i for i in shape_indices if i not in clustering_indices]
        n_anticlustering = len(remaining_shape) * 2 // 10
        anticlustering_indices = set(remaining_shape[:n_anticlustering])

        anticlustering_config = replace(self.config, qa_type_weights={"anticlustering": 1.0})

        anticluster_sizes = list(range(0, 6))  # 0..5 anti-trend metrics
        acs_idx = 0

        tasks = []
        for idx, mode in enumerate(allocated):
            difficulty = difficulties[idx % len(difficulties)]
            seq_len = determine_sequence_length(max_len=self.config.main_pool_max_seq_len)
            if idx in clustering_indices:
                tcs = cluster_sizes[cs_idx % len(cluster_sizes)]
                cs_idx += 1
                tasks.append((mode, 'clustering', seq_len, difficulty, tcs, None))
            elif idx in anticlustering_indices:
                tacs = anticluster_sizes[acs_idx % len(anticluster_sizes)]
                acs_idx += 1
                tasks.append((mode, 'anticlustering', seq_len, difficulty, None, tacs))
            else:
                tasks.append((mode, 'base', seq_len, difficulty, None, None))

        max_workers = os.cpu_count()
        print(f"Spawning {max_workers} workers for {len(tasks)} tasks...")

        # Use 'spawn' context to avoid CUDA fork deadlock — the parent
        # process has vLLM loaded (CUDA initialized), and fork() with
        # active CUDA contexts corrupts the multiprocessing queue.
        # Config is passed via initializer (once per worker) instead of
        # per-task to avoid pickling it 250K times.
        # imap_unordered with chunksize batches tasks (250K/200 = 1250
        # queue messages instead of 250K), avoiding pipe saturation.
        ctx = mp.get_context('spawn')
        with tqdm(total=len(tasks), desc="Generating samples") as pbar:
            with ctx.Pool(
                processes=max_workers,
                initializer=_init_sample_worker,
                initargs=(self.config, clustering_config, anticlustering_config),
            ) as pool:
                for result in pool.imap_unordered(sample_worker, tasks, chunksize=200):
                    if result is not None:
                        results.append(result)
                    pbar.update(1)

        # Top-up: ensure every eval_type meets the minimum sample count
        self._topup_deficit_types(results)

        print("Collecting LLM prompts...")
        all_prompts = self._flatten_llm_prompts(results)
        print(f"Total LLM prompts: {len(all_prompts)}")

        print("Getting LLM answers...")
        llm_answers = self._get_llm_answers(all_prompts)

        print("Replacing placeholders...")
        self._replace_all_placeholders(results, llm_answers)

        self._log_distribution_stats(results)
        return results


_BINARY_VERDICT_TYPES = frozenset({
    'yes_no', 'anti_judgment', 'segment_judgment', 'cross_stat_judgment',
    'correlation', 'anticorrelation',
    # OOD binary verdict types (eval-only)
    'ood_trend_reversal', 'ood_range_normalized_amplitude', 'ood_segment_stat_compare',
    'ood_max_before_min', 'ood_std_exceeds_half_range',
    'ood_max_in_highest_mean_quarter', 'ood_quarter_mean_ordering',
    'ood_symmetric_recovery', 'ood_cycle_mean_trend', 'ood_event_amplitude_vs_std',
    'ood_amplitude_vs_segment_std', 'ood_symmetric_trend_sequence', 'ood_trend_follows_mean',
    # RL single-metric binary types
    'rl_amplitude_vs_range', 'rl_amplitude_vs_std', 'rl_condition_recovery',
    'rl_event_in_trend_type', 'rl_event_near_extremum', 'rl_extrema_same_half',
    'rl_half_mean_compare', 'rl_has_periodicity', 'rl_interval_comparison',
    'rl_max_in_first_half', 'rl_max_in_trend_type', 'rl_mean_shift',
    'rl_mean_stability', 'rl_median_mean_close', 'rl_monotonic_chunks',
    'rl_segment_mean_compare', 'rl_volatility_change',
    # Cross-metric taxonomy binary types
    'atomic_cross_stat_compare', 'atomic_cross_trend_align',
    'rl_cross_stat_ratio', 'rl_cross_event_sync', 'rl_cross_period_compare',
    'rl_cross_attribute_corr',
    'ood_cross_extrema_alignment', 'ood_cross_range_overlap',
})


def _get_result_verdict(r: SampleResult) -> str:
    """Extract verdict string from a SampleResult, normalized to yes/no for binary types."""
    import re
    et = r.eval_task or (r.eval_tasks[0] if r.eval_tasks else '')
    meta = r.eval_metadata or (r.eval_metadatas[0] if r.eval_metadatas else {})
    verdict = str(meta.get('verdict', '')).lower() if isinstance(meta, dict) else ''
    if not verdict and et == 'yes_no':
        output = r.answers[0] if r.answers else ''
        m = re.search(r'answer:\s*(yes|no)', output, re.IGNORECASE)
        verdict = m.group(1).lower() if m else ''
    # Normalize native verdicts to yes/no for binary balancing
    if et == 'correlation':
        if verdict == 'similar':
            verdict = 'yes'
        elif verdict == 'different':
            verdict = 'no'
    elif et == 'anticorrelation':
        if verdict == 'opposite':
            verdict = 'yes'
        elif verdict == 'not_opposite':
            verdict = 'no'
    return verdict


def balance_binary_results(results: List[SampleResult]) -> List[SampleResult]:
    """Hard 50/50 balance for binary verdict eval_types at the source.

    Downsamples the majority verdict to match the minority for each
    binary eval_type. Operates on SampleResults directly — no data
    is written to disk unbalanced.
    """
    from collections import defaultdict

    # Group by (eval_type, verdict)
    buckets = defaultdict(list)
    non_binary = []

    for r in results:
        et = r.eval_task or (r.eval_tasks[0] if r.eval_tasks else '')
        if et in _BINARY_VERDICT_TYPES:
            verdict = _get_result_verdict(r)
            if verdict:
                buckets[(et, verdict)].append(r)
            else:
                non_binary.append(r)
        else:
            non_binary.append(r)

    # For each binary eval_type, downsample majority to match minority
    balanced = list(non_binary)
    eval_types_seen = set(et for et, _ in buckets.keys())

    total_dropped = 0
    for et in sorted(eval_types_seen):
        verdicts = {v: buckets[(et, v)] for _, v in buckets.keys() if _ == et}
        if len(verdicts) < 2:
            # Only one verdict present — keep all (nothing to balance against)
            for v_list in verdicts.values():
                balanced.extend(v_list)
            continue

        counts = {v: len(items) for v, items in verdicts.items()}
        min_count = min(counts.values())

        for v, items in verdicts.items():
            if len(items) > min_count:
                random.shuffle(items)
                balanced.extend(items[:min_count])
                total_dropped += len(items) - min_count
                print(f"  Balance {et}/{v}: {len(items)} → {min_count} (dropped {len(items) - min_count})")
            else:
                balanced.extend(items)

    if total_dropped > 0:
        print(f"  Binary verdict balancing: dropped {total_dropped} majority samples, {len(balanced)} remaining")
    else:
        print(f"  Binary verdict balancing: all types already balanced ({len(balanced)} samples)")

    random.shuffle(balanced)
    return balanced


def write_split_files(
    config: Config,
    split_name: str,
    results: List[SampleResult]
) -> None:
    output_path = config.output_path.parent / f"{split_name}{config.output_path.suffix}"

    print(f"Writing {len(results)} samples to {split_name} split...")

    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            ts_len = len(result.original_timeseries[0]) if result.original_timeseries else None
            eval_meta = dict(result.eval_metadata) if result.eval_metadata is not None else {}
            if ts_len is not None and 'length' not in eval_meta:
                eval_meta['length'] = ts_len

            raw_input = f"{result.base_prompt.rstrip(';.')}. {result.questions[0]}"
            if config.strip:
                last_ts = raw_input.rfind('<ts>')
                if last_ts != -1:
                    raw_input = raw_input[last_ts + len('<ts>'):].lstrip('. ;')
            record = {
                'input': raw_input,
                'output': result.answers[0],
                'timeseries': timeseries_to_list(result.encoded_timeseries),
                'eval_type': result.eval_task if result.eval_task else result.qa_types[0],
                'eval_metadata': eval_meta,
            }
            f.write(json.dumps(record, ensure_ascii=False, cls=NumpyEncoder) + '\n')

    print(f"  - Output: {output_path}")


def _is_force_sft(result: SampleResult) -> bool:
    """Check if a result must be routed to SFT (never RL).

    Force-SFT only for types that provide no useful RL gradient:
    - Bridge (bridge=True in metadata): decomposition demos are a
      pedagogical SFT signal; reusing them in RL defeats their purpose.
    - description / tsevol: free-text; reward returns neutral 0.5 so
      there is no gradient to optimize.

    Everything else — atomic_*, legacy (correlation, clustering,
    yes_no, segment_*, stat_numerical, periodicity, change_point, …),
    and rl_* — flows through the normal partition. Each has a
    verdict-only scorer in ``reward/__init__.py`` routing on
    ``answer: …`` extraction, so RL can train on all of them. This
    preserves atomic skills under RL by including them in the reward
    distribution rather than letting the policy drift away from them.
    """
    meta = result.eval_metadata
    if meta and meta.get('bridge'):
        return True
    task = result.eval_task or ''
    if task in ('description', 'tsevol'):
        return True
    return False


def split_results(
    results: List[SampleResult],
    train_ratio: float = 0.94,
    val_ratio: float = 0.013,
    sft_ratio: float = 0.667,
) -> Tuple[List[SampleResult], List[SampleResult], List[SampleResult], List[SampleResult], List[SampleResult]]:
    """Split results into train_sft / train_rl / val / test plus a back-compat
    train-union value.

    First splits into train/val/test (94/1.3/4.7), then splits train
    into sft/rl (66.7/33.3). The 5th return value (train_union) is kept for
    signature back-compat with older callers but is no longer used by the
    pipeline: under Option A the SFT-stage checkpoint (trained on train_sft
    alone) serves directly as the SFT-only baseline, so no separate
    `train_sft_only.parquet` is emitted. See __main__.py for the rationale.

    Samples with eval_metadata['bridge'] = True are force-routed to
    SFT (never RL). They still appear in val/test.

    Targets at num_data≈200K (with ~10-20% filter loss → ~160-180K effective):
      train_sft ≈ 100K   (SFT phase of SFT→RL pipeline, labeled)
      train_rl  ≈  50K   (RL phase, reward-only; labels withheld from model)
      val       ≈ 2K raw → stratified to 1K in preprocess.py
      test      ≈ 7-9K non-OOD, plus up to `max_ood_samples` OOD (targeting 12K total)
    """
    total = len(results)
    indices = list(range(total))
    random.shuffle(indices)
    shuffled = [results[i] for i in indices]

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_data = shuffled[:train_end]
    val_data = shuffled[train_end:val_end]
    test_data = shuffled[val_end:]

    # Separate force-SFT samples before the sft/rl split
    force_sft = [r for r in train_data if _is_force_sft(r)]
    rest = [r for r in train_data if not _is_force_sft(r)]

    sft_end = int(len(train_data) * sft_ratio)
    # Fill SFT quota: force_sft first, then random from rest
    remaining_sft_slots = max(0, sft_end - len(force_sft))
    sft_data = force_sft + rest[:remaining_sft_slots]
    rl_data = rest[remaining_sft_slots:]

    return sft_data, rl_data, val_data, test_data, train_data
