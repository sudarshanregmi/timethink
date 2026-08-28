"""MTS Local (Multiple Time Series with Local Event Focus) QA Generator."""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from synth.ts_generator.utils.common_utils import (
    article,
    has_local_event_near,
    has_local_event_end_near,
    get_local_event_positions,
    get_local_event_end_positions,
    find_point_far_from_all_events,
    sanitize_attributes_for_sync,
    DEFAULT_THRESHOLD,
)
from synth.ts_generator.utils.thinking import build_local_verification_thought, ThoughtBuilder, get_threshold_note, format_float
from synth.ts_generator.utils.probability_utils import QAType
from synth.align.config import Config, Difficulty, PromptIndexer
from synth.align.base_generator import BaseQAGenerator
from synth.align.templates.prompts import _fix_article
from synth.align.templates import PromptRegistry
from synth.align.types import GenerationResult


# Common local event types used when selecting a random "asked type" for typed questions.
_COMMON_CHANGE_TYPES = [
    "upward spike", "downward spike",
    "continuous upward spike", "continuous downward spike",
    "shake",
    "upward convex", "downward convex",
    "sudden increase", "sudden decrease",
    "rapid rise followed by slow decline",
    "rapid decline followed by slow rise",
    "wide upward spike", "wide downward spike",
]


def _pick_different_type(event_type: str) -> str:
    """Pick a random change type that differs from event_type."""
    others = [t for t in _COMMON_CHANGE_TYPES if t != event_type]
    return random.choice(others) if others else _COMMON_CHANGE_TYPES[0]


def _get_event_direction(event_type: str) -> Optional[str]:
    """Extract 'upward' or 'downward' from an event type string, or None."""
    if not event_type:
        return None
    lower = event_type.lower()
    if any(w in lower for w in ('upward', 'rise', 'increase')):
        return 'upward'
    if any(w in lower for w in ('downward', 'decline', 'decrease')):
        return 'downward'
    return None


class MTSLocalQAGenerator(BaseQAGenerator):
    """QA Generator for multiple time series with local event focus."""

    def __init__(
        self,
        config: Config,
        indexer: PromptIndexer,
        situation: str,
        metrics: List[str],
        attributes: List[Dict],
        cluster_indices: List[Optional[int]],
        positive_clusters: List[List[str]],
        change_positions: List[int],
        metric_to_cluster: Dict[str, str],
        original_timeseries: List[np.ndarray],
        threshold: int = DEFAULT_THRESHOLD,
        threshold_provided: bool = False,
        seq_len: int = 0,
        difficulty: Difficulty = Difficulty.EASY,
    ):
        super().__init__(
            config=config,
            indexer=indexer,
            threshold=threshold,
            threshold_provided=threshold_provided,
            seq_len=seq_len,
            difficulty=difficulty,
        )
        self.situation = situation
        self.metrics = metrics
        self.attributes = attributes
        self.cluster_indices = cluster_indices
        self.positive_clusters = positive_clusters
        self.change_positions = change_positions
        self.metric_to_cluster = metric_to_cluster
        self.original_timeseries = original_timeseries
        
        # Additional tracking for MTS
        self.correlations: List[Dict] = []
        self.clusters: List[Dict] = []

    def _get_anchor_for_metric(self, i: int) -> int:
        """Return the change_position for metric i's cluster."""
        cluster_id = self.cluster_indices[i]
        if cluster_id is not None and cluster_id < len(self.change_positions):
            return self.change_positions[cluster_id]
        return self.change_positions[0] if self.change_positions else 0

    def generate_all_qa(self) -> GenerationResult:
        corr_pool_list = [None] * len(self.metrics)

        for i in range(len(self.metrics)):
            # Generate description QA using base class method
            self._generate_description_qa(
                timeseries=self.original_timeseries[i],
                attributes=self.attributes[i],
                metric=self.metrics[i],
                situation=self.situation,
                series_index=i
            )

            # Generate comparison QAs — boost same-cluster pairs
            comparison_generated = False
            for j in range(len(self.metrics)):
                if i == j:
                    continue
                keep_prob = 0.2
                if (self.cluster_indices[i] is not None and
                        self.cluster_indices[i] == self.cluster_indices[j]):
                    keep_prob = 1.0
                if random.random() < keep_prob:
                    self._generate_comparison_qa(i, j)
                    comparison_generated = True
            # Debug: guarantee at least one comparison pair per metric
            if self.config.debug and not comparison_generated:
                candidates = [j for j in range(len(self.metrics)) if j != i]
                if candidates:
                    self._generate_comparison_qa(i, random.choice(candidates))

            # Generate yes/no QAs
            self._generate_yes_no_qa(i)

            # Generate param QA (~30% per metric, always in debug)
            if self.config.debug or random.random() < 0.30:
                self._generate_param_qa_for_mts(i)

            # Generate similarity QA
            result = self._generate_similarity_qa(i)
            if result:
                corr_pool_list[i] = [list(result[0]), result[1]]

            # Per-metric base QA families (inherited from BaseQAGenerator)
            self._generate_compound_judgment_qa(
                attributes=self.attributes[i],
                metric=self.metrics[i],
                series_index=i,
            )
            self._generate_stat_numerical_qa(
                attributes=self.attributes[i],
                metric=self.metrics[i],
                series_index=i,
                timeseries=self.original_timeseries[i] if i < len(self.original_timeseries) else None,
            )
            self._generate_local_enumeration_qa(
                attributes=self.attributes[i],
                metric=self.metrics[i],
                series_index=i,
            )
            self._generate_periodicity_qa(
                attributes=self.attributes[i],
                metric=self.metrics[i],
                series_index=i,
            )
            self._generate_segment_enumeration_qa(
                attributes=self.attributes[i],
                metric=self.metrics[i],
                series_index=i,
            )
            self._generate_segment_trend_dominance_qa(
                attributes=self.attributes[i],
                metric=self.metrics[i],
                series_index=i,
            )
            self._generate_transition_enumeration_qa(
                attributes=self.attributes[i],
                metric=self.metrics[i],
                series_index=i,
            )
            self._generate_event_segment_enumeration_qa(
                attributes=self.attributes[i],
                metric=self.metrics[i],
                series_index=i,
            )
            self._generate_temporal_position_qa(
                attributes=self.attributes[i],
                metric=self.metrics[i],
                series_index=i,
            )
            self._generate_duration_proportion_qa(
                attributes=self.attributes[i],
                metric=self.metrics[i],
                series_index=i,
            )
            self._generate_change_point_qa(
                attributes=self.attributes[i],
                metric=self.metrics[i],
                series_index=i,
            )

            # New taxonomy: atomic SFT + RL compositions + bridge
            self._generate_taxonomy_qa(
                timeseries=self.original_timeseries[i],
                attributes=self.attributes[i],
                metric=self.metrics[i],
                series_index=i,
            )

        self._generate_compound_description_qa(
            timeseries_list=self.original_timeseries,
            attributes_list=self.attributes,
            metrics=self.metrics,
            situation=self.situation,
            series_indices=list(range(len(self.metrics))),
        )

        self._generate_cross_metric_enumeration_qa(
            attributes_list=self.attributes,
            metrics=self.metrics,
        )

        self._generate_cross_stat_judgment_qa(
            attributes_list=self.attributes,
            metrics=self.metrics,
        )

        # Cross-metric taxonomy QAs (Category H: atomic + RL + bridge)
        self._generate_cross_metric_taxonomy_qa(
            timeseries_list=self.original_timeseries,
            metrics=self.metrics,
            attributes_list=self.attributes,
            seq_len=self.seq_len,
        )

        return GenerationResult(
            questions=self.questions,
            answers=self.answers,
            llm_prompts=self.llm_prompts,
            fields=self.fields,
            qa_types=self.qa_types,
            eval_tasks=self.eval_tasks,
            eval_metadatas=self.eval_metadatas,
            correlations=self.correlations,
            clusters=self.clusters,
            corr_pool_list=corr_pool_list,
        )

    def _generate_comparison_qa(self, i: int, j: int) -> None:
        pos = self._get_anchor_for_metric(i)
        tolerance_text = (
            f", considering fluctuations within {self.threshold} timesteps as synchronized"
            if self.threshold_provided else ""
        )

        # Choose question variant: position-only, explicit framing, type-aware, or end-position.
        variant = random.choices(
            ['position', 'explicit', 'typed', 'end_pos'],
            weights=[0.40, 0.25, 0.15, 0.20]
        )[0]

        # For end_pos: use the end position of the event in series i closest to the anchor.
        if variant == 'end_pos':
            end_positions_i = get_local_event_end_positions(self.attributes[i])
            if end_positions_i:
                query_pos = min(end_positions_i, key=lambda p: abs(p - pos))
            else:
                variant = 'position'
                query_pos = pos
        else:
            query_pos = pos

        if variant == 'end_pos':
            sub_key = "local_corr_end"
        elif variant == 'explicit':
            sub_key = "local_corr_explicit"
        elif variant == 'typed':
            sub_key = "local_corr_typed"
        else:
            sub_key = None

        corr_ctx = {
            "metric_a": self.metrics[i],
            "metric_b": self.metrics[j],
            # 0-indexed: query_pos is position_start or position_end as-is.
            "point": query_pos,
            "tolerance": tolerance_text,
        }

        question = PromptRegistry.get_prompt(
            QAType.CORRELATION,
            context=corr_ctx,
            sub_key=sub_key,
            augment=True,
        )

        self.questions.append(_fix_article(question))
        self.fields.append({"local": [i, j]})
        self.qa_types.append(QAType.CORRELATION)
        self.eval_tasks.append(QAType.CORRELATION)
        # Content-based event detection — use end positions for end_pos variant.
        if variant == 'end_pos':
            event_i = has_local_event_end_near(self.attributes[i], query_pos, self.threshold)
            event_j = has_local_event_end_near(self.attributes[j], query_pos, self.threshold)
        else:
            event_i = has_local_event_near(self.attributes[i], pos, self.threshold)
            event_j = has_local_event_near(self.attributes[j], pos, self.threshold)

        corr_verdict = "similar" if (event_i and event_j) else "different"
        self.eval_metadatas.append({
            "metric_a": self.metrics[i],
            "metric_b": self.metrics[j],
            "position": query_pos,
            "verdict": corr_verdict,
            "length": self.seq_len,
        })

        # Build thought block with sparse attributes.
        sparse_i = sanitize_attributes_for_sync(self.attributes[i])
        sparse_j = sanitize_attributes_for_sync(self.attributes[j])

        if variant == 'end_pos':
            # Strip position_start — only ending matters; add computation via local_end mode.
            for ev in sparse_i.get('local', []):
                ev.pop('position_start', None)
            for ev in sparse_j.get('local', []):
                ev.pop('position_start', None)
            thought_block, _ = build_local_verification_thought(
                [sparse_i, sparse_j],
                [self.metrics[i], self.metrics[j]],
                include_attributes=['local'],
                anchor_point=query_pos,
                mode='local_end',
                show_verdict=True,
                threshold=self.threshold,
                threshold_provided=self.threshold_provided,
            )
        else:
            # Strip position_end — start-pos questions only need starting position.
            for ev in sparse_i.get('local', []):
                ev.pop('position_end', None)
            for ev in sparse_j.get('local', []):
                ev.pop('position_end', None)
            thought_block, _ = build_local_verification_thought(
                [sparse_i, sparse_j],
                [self.metrics[i], self.metrics[j]],
                include_attributes=['local'],
                anchor_point=pos,
                mode='local',
                show_verdict=True,
                threshold=self.threshold,
                threshold_provided=self.threshold_provided,
            )

        base_answer = thought_block
        prompt_idx = self.indexer.next()

        if variant == 'typed':
            type_match = (
                event_i is not None and
                event_j is not None and
                event_i == event_j
            )
            answer, llm_prompt = self._handle_typed_correlation(
                i, j, pos, base_answer, prompt_idx, event_i, event_j, type_match
            )
        elif variant == 'end_pos':
            answer, llm_prompt = self._handle_end_pos_correlation(
                i, j, query_pos, base_answer, prompt_idx, event_i, event_j
            )
        else:
            # Position / explicit variants (position-only matching).
            if event_i:
                if event_j:
                    answer, llm_prompt = self._handle_positive_correlation(
                        i, j, pos, base_answer, prompt_idx
                    )
                else:
                    answer, llm_prompt = self._handle_partial_correlation(
                        i, j, event_i, event_j, pos, base_answer, prompt_idx
                    )
            else:
                if not event_j:
                    if (self.cluster_indices[i] is not None and
                            self.cluster_indices[i] == self.cluster_indices[j]):
                        answer, llm_prompt = self._handle_same_cluster_no_correlation(
                            i, j, pos, base_answer, prompt_idx
                        )
                    else:
                        answer, llm_prompt = self._handle_no_correlation(
                            i, j, pos, base_answer, prompt_idx
                        )
                else:
                    answer, llm_prompt = self._handle_partial_correlation(
                        i, j, event_i, event_j, pos, base_answer, prompt_idx
                    )

        self.answers.append(answer)
        self.llm_prompts.append([llm_prompt] if llm_prompt else [])

    def _handle_positive_correlation(
        self,
        i: int,
        j: int,
        position: int,
        base_answer: str,
        prompt_idx: int
    ) -> Tuple[str, str]:
        answer = (
            f"{base_answer} {self.metrics[i]} and {self.metrics[j]} both show local fluctuations around point {position}, "
            f"indicating a correlation in their behavior. <|prompt{prompt_idx}|>"
        )
        
        self.correlations.append({
            "pair": [self.metrics[i], self.metrics[j]],
            "explain": f"<|prompt{prompt_idx}|>",
            "label": True
        })
        
        llm_prompt = (
            f"In {article(self.situation)} {self.situation} system, there are many monitoring metrics. "
            f"Near a timestamp (maybe during a failure), we found there are fluctuations "
            f"in {self.metrics[i]} and {self.metrics[j]} that happens together. "
            f"Please explain why they fluctuates together in one sentence."
        )
        
        if self.metric_to_cluster.get(self.metrics[i]) == self.metric_to_cluster.get(self.metrics[j]):
            llm_prompt += f" (Hint: These two metrics are both {self.metric_to_cluster[self.metrics[i]]}-related.)"
        
        return answer, llm_prompt

    def _handle_same_cluster_no_correlation(
        self,
        i: int,
        j: int,
        position: int,
        base_answer: str,
        prompt_idx: int
    ) -> Tuple[str, str]:
        answer = (
            f"{base_answer} Neither metric shows significant fluctuation around point {position}. "
            f"There is no fluctuation correlation at this specific point. <|prompt{prompt_idx}|>"
        )
        
        self.correlations.append({
            "pair": [self.metrics[i], self.metrics[j]],
            "explain": f"<|prompt{prompt_idx}|>",
            "label": False
        })
        
        llm_prompt = (
            f"In {article(self.situation)} {self.situation} system, we found there are **no** fluctuations in both "
            f"{self.metrics[i]} and {self.metrics[j]} now, but they fluctuated "
            f"together in another time. Please explain why they are not fluctuating together "
            f"at this time in one sentence."
        )
        
        if self.metric_to_cluster.get(self.metrics[i]) == self.metric_to_cluster.get(self.metrics[j]):
            llm_prompt += f" (Hint: These two metrics are both {self.metric_to_cluster[self.metrics[i]]}-related.)"
        
        return answer, llm_prompt

    def _handle_partial_correlation(
        self,
        i: int,
        j: int,
        event_i: Optional[str],
        event_j: Optional[str],
        position: int,
        base_answer: str,
        prompt_idx: int
    ) -> Tuple[str, str]:
        if not event_i:
            answer = (
                f"{base_answer} {self.metrics[i]} does not show significant fluctuation around point {position}, "
                f"whereas {self.metrics[j]} shows a fluctuation. "
                f"Therefore, they are not correlated. <|prompt{prompt_idx}|>"
            )
        else:
            answer = (
                f"{base_answer} {self.metrics[i]} shows a fluctuation around point {position}, "
                f"while {self.metrics[j]} does not. "
                f"Thus, there is no correlation. <|prompt{prompt_idx}|>"
            )
        
        self.correlations.append({
            "pair": [self.metrics[i], self.metrics[j]],
            "explain": f"<|prompt{prompt_idx}|>",
            "label": False
        })
        
        a, b = (i, j) if event_i else (j, i)
        llm_prompt = (
            f"In {article(self.situation)} {self.situation} system, near a timestamp, we found there are fluctuations "
            f"in {self.metrics[a]}, but no fluctuations in {self.metrics[b]}. "
            f"Please explain why they are **not** fluctuating together in one simple sentence."
        )
        
        return answer, llm_prompt

    def _handle_no_correlation(
        self,
        i: int,
        j: int,
        position: int,
        base_answer: str,
        prompt_idx: int
    ) -> Tuple[str, str]:
        answer = (
            f"{base_answer} {self.metrics[i]} does not show significant fluctuation around point {position}. "
            f"Therefore, it is not possible to establish a correlation of fluctuation with "
            f"{self.metrics[j]} at this location. <|prompt{prompt_idx}|>"
        )

        self.correlations.append({
            "pair": [self.metrics[i], self.metrics[j]],
            "explain": f"<|prompt{prompt_idx}|>",
            "label": False
        })

        llm_prompt = (
            f"In {article(self.situation)} {self.situation} system, near a timestamp, we found no fluctuations "
            f"in both {self.metrics[i]} and {self.metrics[j]}. "
            f"Please explain why they are **not** fluctuating in one simple sentence."
        )

        return answer, llm_prompt

    def _handle_typed_correlation(
        self,
        i: int,
        j: int,
        position: int,
        base_answer: str,
        prompt_idx: int,
        event_i: Optional[str],
        event_j: Optional[str],
        type_match: bool,
    ) -> Tuple[str, Optional[str]]:
        """Handle answer for type-aware (same type + same position) correlation variant."""
        mi, mj = self.metrics[i], self.metrics[j]
        pt = position

        if type_match:
            answer = (
                f"{base_answer} Both {mi} and {mj} show a {event_i} near point {pt}, "
                f"matching in both timing and event type. They are type-correlated. <|prompt{prompt_idx}|>"
            )
            self.correlations.append({
                "pair": [mi, mj],
                "explain": f"<|prompt{prompt_idx}|>",
                "label": True
            })
            llm_prompt = (
                f"In {article(self.situation)} {self.situation} system, both {mi} and {mj} show a {event_i} "
                f"near the same timestamp. Please explain why they exhibit the same kind "
                f"of fluctuation in one sentence."
            )
        elif event_i and event_j:
            # Both have events but different types
            answer = (
                f"{base_answer} {mi} shows a {event_i} near point {pt}, while {mj} shows "
                f"a {event_j}. The event types differ, so they are not type-correlated. "
                f"<|prompt{prompt_idx}|>"
            )
            self.correlations.append({
                "pair": [mi, mj],
                "explain": f"<|prompt{prompt_idx}|>",
                "label": False
            })
            llm_prompt = (
                f"In {article(self.situation)} {self.situation} system, {mi} has {article(event_i)} {event_i} while {mj} has {article(event_j)} "
                f"{event_j} near the same point. Please explain in one sentence why their "
                f"local events differ."
            )
        elif event_i:
            answer = (
                f"{base_answer} {mi} shows {article(event_i)} {event_i} near point {pt}, but {mj} has no "
                f"local event there. They are not type-correlated. <|prompt{prompt_idx}|>"
            )
            self.correlations.append({
                "pair": [mi, mj],
                "explain": f"<|prompt{prompt_idx}|>",
                "label": False
            })
            llm_prompt = (
                f"In {article(self.situation)} {self.situation} system, {mi} has {article(event_i)} {event_i} but {mj} does not "
                f"fluctuate near the same timestamp. Explain why in one sentence."
            )
        else:
            answer = (
                f"{base_answer} Neither {mi} nor {mj} shows a local event near point {pt}. "
                f"There is no type-correlated behavior. <|prompt{prompt_idx}|>"
            )
            self.correlations.append({
                "pair": [mi, mj],
                "explain": f"<|prompt{prompt_idx}|>",
                "label": False
            })
            llm_prompt = (
                f"In {article(self.situation)} {self.situation} system, neither {mi} nor {mj} shows a local "
                f"fluctuation near this timestamp. Explain why in one sentence."
            )

        return answer, llm_prompt

    def _handle_end_pos_correlation(
        self,
        i: int,
        j: int,
        end_pos: int,
        base_answer: str,
        prompt_idx: int,
        event_i: Optional[str],
        event_j: Optional[str],
    ) -> Tuple[str, Optional[str]]:
        """Handle answer for end-position correlation variant."""
        mi, mj = self.metrics[i], self.metrics[j]
        pt = end_pos  # position_end (exclusive endpoint), 0-indexed

        if event_i and event_j:
            answer = (
                f"{base_answer} Both {mi} and {mj} have local events ending near "
                f"point {pt}, indicating correlated behavior at their event endpoints. "
                f"<|prompt{prompt_idx}|>"
            )
            self.correlations.append({
                "pair": [mi, mj], "explain": f"<|prompt{prompt_idx}|>", "label": True
            })
            llm_prompt = (
                f"In {article(self.situation)} {self.situation} system, both {mi} and {mj} have local "
                f"fluctuations ending near the same timestamp. Explain why in one sentence."
            )
            if self.metric_to_cluster.get(mi) == self.metric_to_cluster.get(mj):
                llm_prompt += f" (Hint: They are both {self.metric_to_cluster[mi]}-related.)"
        elif event_i:
            answer = (
                f"{base_answer} {mi} has a local event ({event_i}) ending near point {pt}, "
                f"but {mj} does not have an event ending there. "
                f"Not correlated by end position. <|prompt{prompt_idx}|>"
            )
            self.correlations.append({
                "pair": [mi, mj], "explain": f"<|prompt{prompt_idx}|>", "label": False
            })
            llm_prompt = (
                f"In {article(self.situation)} {self.situation} system, {mi} has a local event ending "
                f"near a timestamp but {mj} does not. Explain why in one sentence."
            )
        elif event_j:
            answer = (
                f"{base_answer} {mj} has a local event ({event_j}) ending near point {pt}, "
                f"but {mi} does not have an event ending there. "
                f"Not correlated by end position. <|prompt{prompt_idx}|>"
            )
            self.correlations.append({
                "pair": [mi, mj], "explain": f"<|prompt{prompt_idx}|>", "label": False
            })
            llm_prompt = (
                f"In {article(self.situation)} {self.situation} system, {mj} has a local event ending "
                f"near a timestamp but {mi} does not. Explain why in one sentence."
            )
        else:
            answer = (
                f"{base_answer} Neither {mi} nor {mj} has a local event ending near "
                f"point {pt}. There is no end-position correlation between them. "
                f"<|prompt{prompt_idx}|>"
            )
            self.correlations.append({
                "pair": [mi, mj], "explain": f"<|prompt{prompt_idx}|>", "label": False
            })
            llm_prompt = (
                f"In {article(self.situation)} {self.situation} system, neither {mi} nor {mj} has a local "
                f"event ending near this timestamp. Explain why in one sentence."
            )

        return answer, llm_prompt

    def _generate_similarity_qa(self, i: int) -> Optional[Tuple[List[int], str]]:
        pos = self._get_anchor_for_metric(i)
        target_name = self.metrics[i]

        tolerance_text = (
            f", where fluctuations within {self.threshold} timesteps are considered synchronized"
            if self.threshold_provided else ""
        )

        # Choose variant: position-only, type-aware, or end-position.
        variant_choice = random.choices(
            ['position', 'typed', 'end_pos'],
            weights=[0.73, 0.20, 0.07]
        )[0]
        use_typed = (variant_choice == 'typed')
        use_end_pos = (variant_choice == 'end_pos')

        # For end_pos: use the end position of the event in series i closest to the anchor.
        if use_end_pos:
            end_positions_i = get_local_event_end_positions(self.attributes[i])
            if end_positions_i:
                cluster_pos = min(end_positions_i, key=lambda p: abs(p - pos))
            else:
                use_end_pos = False
                cluster_pos = pos
        else:
            cluster_pos = pos

        if use_end_pos:
            sub_key = "local_cluster_end"
        elif use_typed:
            sub_key = "local_cluster_typed"
        else:
            sub_key = None

        question = PromptRegistry.get_prompt(
            QAType.CLUSTERING,
            context={
                "metric": target_name,
                # 0-indexed: cluster_pos is position_start or position_end as-is.
                "point": cluster_pos,
                "tolerance": tolerance_text,
            },
            sub_key=sub_key,
            augment=True,
        )

        eval_meta = {
            "target_metric": target_name,
            "position": cluster_pos,
            "type_aware": use_typed,
            "end_pos": use_end_pos,
            "length": self.seq_len,
        }

        # Build thought block with sparse attributes.
        target_attr = sanitize_attributes_for_sync(self.attributes[i])
        others_indices = [idx for idx in range(len(self.metrics)) if idx != i]
        ordered_metrics = [target_name] + [self.metrics[idx] for idx in others_indices]
        ordered_attrs = [target_attr] + [
            sanitize_attributes_for_sync(self.attributes[idx]) for idx in others_indices
        ]

        if use_end_pos:
            # Strip position_start from all attrs — only ending matters for end_pos questions.
            for attr in ordered_attrs:
                for ev in attr.get('local', []):
                    ev.pop('position_start', None)
            thought_block, cluster_metrics = build_local_verification_thought(
                ordered_attrs,
                ordered_metrics,
                include_attributes=['local'],
                anchor_point=cluster_pos,
                mode='local_end',
                show_cluster=True,
                threshold=self.threshold,
                threshold_provided=self.threshold_provided,
            )
        else:
            # For type-aware clustering, identify the target event type before attribute stripping.
            typed_event_type = has_local_event_near(self.attributes[i], pos, self.threshold) if use_typed else None

            # Strip position_end — start-pos clustering only needs starting position for proximity math.
            for attr in ordered_attrs:
                for ev in attr.get('local', []):
                    ev.pop('position_end', None)
            thought_block, cluster_metrics = build_local_verification_thought(
                ordered_attrs,
                ordered_metrics,
                include_attributes=['local'],
                anchor_point=pos,
                mode='local',
                show_cluster=True,
                show_verdict=False,
                threshold=self.threshold,
                threshold_provided=self.threshold_provided,
                target_event_type=typed_event_type,
            )

        # Convert cluster metric names to indices.
        metric_to_idx = {name: idx for idx, name in enumerate(self.metrics)}
        calculated_related_indices = [metric_to_idx[name] for name in cluster_metrics]

        # Pre-compute end event type for use_end_pos branches (grounded from anchor data).
        target_end_type = None
        if use_end_pos and cluster_metrics:
            anchor_local = ordered_attrs[0].get('local', [])
            best_dist = float('inf')
            for ev in anchor_local:
                pos_end = ev.get('position_end')
                if pos_end is None:
                    continue
                dist = abs(pos_end - cluster_pos)
                if dist <= self.threshold and dist < best_dist:
                    best_dist = dist
                    e_type = ev.get('type', 'change')
                    if isinstance(e_type, list):
                        e_type = e_type[0]
                    target_end_type = e_type
            if target_end_type is None:
                return None  # No grounded event type — skip QA

        # Generate answer based on cluster result.
        if not cluster_metrics:
            if random.random() > 0.25:
                return None
            if use_end_pos:
                answer = (
                    f"{thought_block}{target_name} has no local event "
                    f"ending near point {cluster_pos}. "
                    f"Cannot find related time series based on event endpoints."
                )
            else:
                answer = (
                    f"{thought_block}{target_name} does not show "
                    f"significant fluctuation around point {cluster_pos}. "
                    f"Hence, I cannot find other time series related to it in terms of fluctuation."
                )
            self.questions.append(_fix_article(question))
            self.qa_types.append(QAType.CLUSTERING)
            self.eval_tasks.append(QAType.CLUSTERING)
            self.eval_metadatas.append(eval_meta)
            self.answers.append(answer)
            self.fields.append({"local": [i]})
            self.llm_prompts.append([])
            return None
        elif len(cluster_metrics) <= 1:
            if random.random() > 0.25:
                return None
            if use_end_pos:
                answer = (
                    f"{thought_block}{target_name} has a "
                    f"{target_end_type} ending near point {cluster_pos}, "
                    f"but no other time series shows a local event ending near this point."
                )
            else:
                answer = (
                    f"{thought_block}{target_name} fluctuates around "
                    f"point {cluster_pos}, "
                    f"but I did not find any other time series that show a similar fluctuation at this point."
                )
            self.questions.append(_fix_article(question))
            self.qa_types.append(QAType.CLUSTERING)
            self.eval_tasks.append(QAType.CLUSTERING)
            self.eval_metadatas.append(eval_meta)
            self.answers.append(answer)
            self.fields.append({"local": [i]})
            self.llm_prompts.append([])
            return None

        # Matches found.
        self.questions.append(_fix_article(question))
        self.qa_types.append(QAType.CLUSTERING)
        self.eval_tasks.append(QAType.CLUSTERING)
        self.eval_metadatas.append(eval_meta)
        prompt_idx = self.indexer.next()
        others_names = [name for name in cluster_metrics if name != target_name]

        if use_end_pos:
            answer = (
                f"{thought_block}Based on local events ending near point {cluster_pos}, "
                f"I found the following related time series: "
            )
            answer += ", ".join(others_names) + ". "
            answer += (
                f"Their local events all end near point {cluster_pos}, "
                f"similar to {target_name}'s {target_end_type} endpoint. "
                f"<|prompt{prompt_idx}|>"
            )
        else:
            answer = (
                f"{thought_block}Based on the fluctuations around point {cluster_pos}, "
                f"I found the following related time series: "
            )
            answer += ", ".join(others_names) + ". "
            if use_typed:
                target_type = has_local_event_near(self.attributes[i], pos, self.threshold)
                similarity_desc = (
                    f"the same type of local event ({target_type}) starting near this point"
                    if target_type else "similar local fluctuations near this point"
                )
                answer += (
                    f"They all show {similarity_desc}, matching {target_name} in both "
                    f"timing and event type. <|prompt{prompt_idx}|>"
                )
            else:
                answer += (
                    f"They all exhibit fluctuations starting near this point, similar to {target_name}. "
                    f"<|prompt{prompt_idx}|>"
                )

        self.answers.append(answer)
        self.fields.append({"local": calculated_related_indices})

        self.clusters.append({
            'col_idx': [
                [int(j), has_local_event_near(self.attributes[j], pos, self.threshold)]
                for j in calculated_related_indices
            ],
            'cols': list(cluster_metrics),
            'explain': f"<|prompt{prompt_idx}|>"
        })

        related_str = ', '.join(cluster_metrics)
        llm_prompt = (
            f"In {article(self.situation)} {self.situation} system, we found there are fluctuations in {related_str}. "
            f"Please explain their relationship in physical meaning and describe what's "
            f"may happening in 1 sentence."
        )

        self.llm_prompts.append([llm_prompt])

        return calculated_related_indices, answer

    def _select_confusing_mts_point(self, i: int, want_yes: bool) -> Optional[int]:
        """For HARD MTS: select a point that's confusing in multi-metric context.

        want_yes: point where metric i AND other metrics have events nearby.
        want_no:  point where OTHER metrics have events but metric i does NOT.
        """
        my_positions = get_local_event_positions(self.attributes[i])
        other_positions = []
        for j in range(len(self.metrics)):
            if j != i:
                other_positions.extend(get_local_event_positions(self.attributes[j]))
        if not other_positions:
            return None

        if want_yes:
            # Find a point near metric i's event where other metrics also have events
            if not my_positions:
                return None
            candidates = []
            for pos in my_positions:
                if any(abs(pos - op) <= self.threshold for op in other_positions):
                    candidates.append(pos)
            if not candidates:
                return None
            event_pos = random.choice(candidates)
            lo, hi = self._yes_offset_range(Difficulty.HARD, self.threshold)
            if lo > hi:
                lo, hi = 0, max(self.threshold // 3, 1)
            offset = random.randint(lo, hi) * random.choice([-1, 1])
            return max(0, min(event_pos + offset, self.seq_len - 1))
        else:
            # Find a point near OTHER metrics' events where metric i has NO event
            random.shuffle(other_positions)
            for op in other_positions:
                if my_positions and all(abs(op - mp) > self.threshold for mp in my_positions):
                    # op is near other metric but far from metric i — confusing "no"
                    return max(0, min(op, self.seq_len - 1))
            return None

    def _generate_yes_no_qa(self, i: int) -> None:
        # In debug mode: generate one question per variant (exhaustive)
        if self.config.debug:
            local_positions = get_local_event_positions(self.attributes[i])
            has_events = len(local_positions) > 0
            for forced_variant in ['generic', 'typed', 'property', 'end_pos']:
                want_yes = random.random() < 0.5
                qp = self._select_yes_no_query_point(local_positions, has_events, want_yes=want_yes)
                # Skip if desired verdict unachievable — never fallback to opposite.
                if qp is not None:
                    self._generate_single_yes_no_for_mts(i, qp, forced_variant=forced_variant)
            return

        num_questions = random.randint(0, 3)
        if num_questions == 0:
            return

        local_positions = get_local_event_positions(self.attributes[i])
        has_events = len(local_positions) > 0

        for _ in range(num_questions):
            # 50/50 verdict balancing
            want_yes = random.random() < 0.5
            query_point = None

            # HARD MTS: try confusing multi-metric placement first
            if self.difficulty == Difficulty.HARD:
                query_point = self._select_confusing_mts_point(i, want_yes)

            # Standard difficulty-aware placement (fallback chain built-in)
            if query_point is None:
                query_point = self._select_yes_no_query_point(
                    local_positions, has_events, want_yes=want_yes
                )
            # Skip if desired verdict unachievable — never fallback to opposite.
            if query_point is None:
                continue

            if query_point is not None:
                self._generate_single_yes_no_for_mts(i, query_point)

    def _generate_single_yes_no_for_mts(
        self, i: int, query_point: int, forced_variant: Optional[str] = None,
    ) -> None:
        metric_name = self.metrics[i]

        tolerance_text = (
            f" within the tolerance of {self.threshold} timestep{'s' if self.threshold != 1 else ''}"
            if self.threshold_provided else ""
        )

        # Pre-detect actual event type at the query point.
        event_type = has_local_event_near(self.attributes[i], query_point, self.threshold)

        # Choose variant: generic / typed / property / end-position.
        if forced_variant is not None:
            variant = forced_variant
        else:
            variant = random.choices(
                ['generic', 'typed', 'property', 'end_pos'],
                weights=[0.40, 0.25, 0.15, 0.20]
            )[0]

        if variant == 'property':
            self._generate_local_property_for_mts(i, query_point, event_type)
            return

        # For end_pos: ask about an event's end position rather than its start.
        if variant == 'end_pos':
            end_positions = get_local_event_end_positions(self.attributes[i])
            if end_positions:
                # 70% pick a real end position (exclusive endpoint),
                # 30% use query_point (start pos) as near-miss for negative examples.
                if random.random() < 0.70:
                    end_pt = random.choice(end_positions)
                else:
                    end_pt = query_point
                end_event_type = has_local_event_end_near(
                    self.attributes[i], end_pt, self.threshold
                )
                # end_pt = position_end (exclusive endpoint), 0-indexed.
                question = PromptRegistry.get_prompt(
                    QAType.YES_NO,
                    context={"point": end_pt, "metric": metric_name, "tolerance": tolerance_text},
                    sub_key="local_yes_no_end",
                    augment=True,
                )
                sparse_attr = sanitize_attributes_for_sync(self.attributes[i])
                # Strip position_start — for end-pos questions only the ending matters.
                for ev in sparse_attr.get('local', []):
                    ev.pop('position_start', None)
                thought, _ = build_local_verification_thought(
                    [sparse_attr],
                    [metric_name],
                    include_attributes=['local'],
                    anchor_point=end_pt,
                    mode='local_end',
                    threshold=self.threshold,
                    threshold_provided=self.threshold_provided,
                )
                verdict = "yes" if end_event_type else "no"
                thought = thought.replace("\n</think>", f"\nanswer: {verdict}\n</think>", 1)
                if end_event_type:
                    answer = (
                        f"{thought}Yes, {metric_name} has {article(end_event_type)} {end_event_type} ending "
                        f"near point {end_pt}."
                    )
                else:
                    answer = (
                        f"{thought}No, there is no local event ending around "
                        f"point {end_pt} in {metric_name}."
                    )
                self.questions.append(_fix_article(question))
                self.answers.append(answer)
                self.llm_prompts.append([])
                self.fields.append({"local": [i]})
                self.qa_types.append(QAType.YES_NO)
                self.eval_tasks.append(QAType.YES_NO)
                self.eval_metadatas.append({
                    "target_point": end_pt,
                    "metric": metric_name,
                    "threshold": self.threshold,
                    "variant": "end_pos",
                    "verdict": verdict,
                    "length": self.seq_len,
                })
                return
            else:
                variant = 'generic'  # fall back if no end positions available

        # --- Build question ---
        base_ctx = {"point": query_point, "metric": metric_name, "tolerance": tolerance_text}

        if variant == 'typed':
            # 70% ask about the actual type (yes answer if present), 30% ask about a different type.
            if event_type and random.random() < 0.70:
                asked_type = event_type
            elif event_type:
                asked_type = _pick_different_type(event_type)
            else:
                asked_type = random.choice(_COMMON_CHANGE_TYPES)
            question = PromptRegistry.get_prompt(
                QAType.YES_NO,
                context={**base_ctx, "event_type": asked_type},
                sub_key="local_yes_no_typed",
                augment=True,
            )
        else:
            asked_type = None
            question = PromptRegistry.get_prompt(
                QAType.YES_NO,
                context=base_ctx,
                augment=True,
            )

        # --- Build thought block ---
        sparse_attr = sanitize_attributes_for_sync(self.attributes[i])
        # Strip position_end — start-pos questions only need starting position for the proximity math.
        for ev in sparse_attr.get('local', []):
            ev.pop('position_end', None)
        thought, _ = build_local_verification_thought(
            [sparse_attr],
            [metric_name],
            include_attributes=['local'],
            anchor_point=query_point,
            mode='local',
            threshold=self.threshold,
            threshold_provided=self.threshold_provided,
        )

        # --- Inject answer verdict into thought block ---
        if variant == 'typed':
            verdict = "yes" if (event_type and event_type == asked_type) else "no"
        else:
            verdict = "yes" if event_type else "no"
        thought = thought.replace("\n</think>", f"\nanswer: {verdict}\n</think>", 1)

        # --- Build answer ---
        if variant == 'typed':
            if event_type and event_type == asked_type:
                answer = (
                    f"{thought}Yes, {metric_name} has {article(asked_type)} {asked_type} around point {query_point}."
                )
            elif event_type and event_type != asked_type:
                answer = (
                    f"{thought}No, there is no {asked_type} in {metric_name} around "
                    f"point {query_point}. The local event detected there is {article(event_type)} {event_type}."
                )
            else:
                answer = (
                    f"{thought}No, there is no {asked_type} near point {query_point} "
                    f"in {metric_name}."
                )
        else:
            if event_type:
                answer = f"{thought}Yes, {metric_name} has a fluctuation around point {query_point}."
            else:
                answer = (
                    f"{thought}I did not find any local event "
                    f"starting around point {query_point} in {metric_name}."
                )

        self.questions.append(_fix_article(question))
        self.answers.append(answer)
        self.llm_prompts.append([])
        self.fields.append({"local": [i]})
        self.qa_types.append(QAType.YES_NO)
        self.eval_tasks.append(QAType.YES_NO)
        self.eval_metadatas.append({
            "target_point": query_point,
            "metric": metric_name,
            "threshold": self.threshold,
            "variant": variant,
            "verdict": verdict,
            "length": self.seq_len,
        })

    def _generate_local_property_for_mts(
        self, i: int, query_point: int, event_type: Optional[str]
    ) -> None:
        """Generate a property-style question about local event characteristics."""
        metric_name = self.metrics[i]
        local_events = self.attributes[i].get('local', [])
        count = len(local_events)
        tolerance_text = (
            f", within tolerance of {self.threshold} timestep{'s' if self.threshold != 1 else ''}"
            if self.threshold_provided else ""
        )
        ctx = {"metric": metric_name, "point": query_point, "tolerance": tolerance_text}

        # Choose sub-variant: count or direction/type at query_point.
        # Only offer direction/type sub-variants when there is a detected event.
        if event_type and random.random() < 0.60:
            sub = random.choice(['direction', 'type_id'])
        else:
            sub = 'count'

        sparse_attr = sanitize_attributes_for_sync(self.attributes[i])

        # Compute direction early (needed for compute_text in direction sub-variant).
        direction = _get_event_direction(event_type or '') if event_type else None

        # For count: only event types matter — strip all position/value fields.
        # For direction/type_id: only start matters for proximity, strip end.
        if sub == 'count':
            for ev in sparse_attr.get('local', []):
                ev.pop('position_start', None)
                ev.pop('position_end', None)
                ev.pop('value_start', None)
                ev.pop('value_end', None)
        else:
            for ev in sparse_attr.get('local', []):
                ev.pop('position_end', None)

        # --- Build compute_text (Part 2 derivation) ---
        if sub == 'count':
            compute_text = f"count = {count} events."
        else:
            near_event = None
            for ev in sparse_attr.get('local', []):
                if abs(ev.get('position_start', 0) - query_point) <= self.threshold:
                    near_event = ev
                    break
            if near_event:
                ev_pos = near_event['position_start']
                diff = abs(ev_pos - query_point)
                if sub == 'direction':
                    compute_text = (
                        f"|{ev_pos} - {query_point}| = {diff} <= {self.threshold}. "
                        f"found: type={event_type}, direction={direction if direction else 'none'}."
                    )
                else:  # type_id
                    compute_text = (
                        f"|{ev_pos} - {query_point}| = {diff} <= {self.threshold}. "
                        f"found: type={event_type}."
                    )
            else:
                # Can't show distance proof — skip this QA
                return

        tb = ThoughtBuilder()
        tb.metric(metric_name, sparse_attr, ['local'])
        tb.line(compute_text)
        thought = tb.build()[0]

        if sub == 'count':
            question = PromptRegistry.get_prompt(
                QAType.YES_NO, context=ctx, sub_key="local_property_count", augment=True,
            )
            if count == 0:
                answer = f"{thought}{metric_name} has no local fluctuations."
            elif count == 1:
                e = local_events[0]
                answer = (
                    f"{thought}{metric_name} has 1 local fluctuation: "
                    f"a {e['type']}."
                )
            else:
                descs = [f"a {e['type']}" for e in local_events[:5]]
                suffix = " (and more)" if count > 5 else ""
                answer = (
                    f"{thought}{metric_name} has {count} local fluctuations: "
                    f"{', '.join(descs)}{suffix}."
                )

        elif sub == 'direction':
            # direction already computed above
            question = PromptRegistry.get_prompt(
                QAType.YES_NO, context=ctx, sub_key="local_property_direction", augment=True,
            )
            if direction:
                answer = (
                    f"{thought}The local fluctuation in {metric_name} near point "
                    f"{query_point} is {direction} (it is {article(event_type)} {event_type})."
                )
            else:
                answer = (
                    f"{thought}The local event in {metric_name} near point {query_point} "
                    f"is {article(event_type)} {event_type}, which does not have a clear upward/downward direction."
                )

        else:  # type_id
            question = PromptRegistry.get_prompt(
                QAType.YES_NO, context=ctx, sub_key="local_property_type_id", augment=True,
            )
            answer = (
                f"{thought}The local event in {metric_name} near point {query_point} "
                f"is {article(event_type)} {event_type}."
            )

        self.questions.append(_fix_article(question))
        self.answers.append(answer)
        self.llm_prompts.append([])
        self.fields.append({"local": [i]})
        self.qa_types.append(QAType.DESCRIPTION)
        self.eval_tasks.append(QAType.DESCRIPTION)
        self.eval_metadatas.append({
            "target_point": query_point,
            "metric": metric_name,
            "threshold": self.threshold,
            "variant": "property",
            "sub": sub,
            "length": self.seq_len,
        })

    # Mapping from params key → human-readable phase name for timing questions.
    _TIMING_KEY_TO_PHASE = {
        'rise_length': 'rise',
        'fall_length': 'fall',
        'plateau_length': 'plateau',
        'drop_length': 'drop',
        'trough_length': 'trough',
        'change_length': 'change',
        'spike_length': 'spike',
        'followup_length': 'followup',
        'duration': 'event',
    }

    def _generate_param_qa_for_mts(self, i: int) -> None:
        """Generate a question about a structural parameter stored in a local event's params dict."""
        local_events = self.attributes[i].get('local', [])
        events_with_params = [e for e in local_events if e.get('params')]
        if not events_with_params:
            return

        event = random.choice(events_with_params)
        params = event['params']
        metric_name = self.metrics[i]
        pos = event.get('position_start', 0)
        point = pos

        # Collect available question types for this event.
        available: List[str] = []

        timing_pairs = [
            (k, label) for k, label in self._TIMING_KEY_TO_PHASE.items() if k in params
        ]
        if timing_pairs:
            available.append('timing')
        if 'followup_type' in params:
            available.append('followup')
        if 'num_spikes' in params:
            available.append('count')
        event_amplitude = event.get('amplitude')
        if event_amplitude is not None:
            available.append('amplitude')

        if not available:
            return

        param_type = random.choice(available)
        tolerance_text = (
            f", within tolerance of {self.threshold} timestep{'s' if self.threshold != 1 else ''}"
            if self.threshold_provided else ""
        )
        ctx = {"metric": metric_name, "point": point, "tolerance": tolerance_text}

        if param_type == 'timing':
            key, phase_name = random.choice(timing_pairs)
            value = params[key]
            question = PromptRegistry.get_prompt(
                QAType.YES_NO,
                context={**ctx, "phase_name": phase_name},
                sub_key="local_param_timing",
                augment=True,
            )
            answer_suffix = (
                f"The {phase_name} phase of the local event near point {point} "
                f"in {metric_name} lasts {value} timesteps."
            )

        elif param_type == 'followup':
            followup = params['followup_type']
            question = PromptRegistry.get_prompt(
                QAType.YES_NO, context=ctx, sub_key="local_param_followup", augment=True,
            )
            answer_suffix = (
                f"After the spike near point {point} in {metric_name}, "
                f"the followup is {article(followup)} {followup}."
            )

        elif param_type == 'count':
            n = params['num_spikes']
            question = PromptRegistry.get_prompt(
                QAType.YES_NO, context=ctx, sub_key="local_param_count", augment=True,
            )
            answer_suffix = (
                f"The local event near point {point} in {metric_name} "
                f"contains {n} spike{'s' if n != 1 else ''}."
            )

        else:  # amplitude
            question = PromptRegistry.get_prompt(
                QAType.YES_NO, context=ctx, sub_key="local_param_amplitude", augment=True,
            )
            answer_suffix = (
                f"The amplitude of the local event near point {point} "
                f"in {metric_name} is {format_float(event_amplitude, 2)}."
            )

        # Build minimal event dict: only the fields needed to answer this question.
        e_type = event.get('type', '')
        if isinstance(e_type, list):
            e_type = e_type[0] if e_type else ''
        event_for_cot: Dict[str, Any] = {'type': e_type, 'position_start': pos,
                                          'value_start': event['value_start']}
        if param_type == 'amplitude':
            event_for_cot['amplitude'] = event_amplitude
        elif param_type == 'timing':
            event_for_cot['params'] = {key: value}
        elif param_type == 'followup':
            event_for_cot['params'] = {'followup_type': followup}
        elif param_type == 'count':
            event_for_cot['params'] = {'num_spikes': n}
        cot_attr = {'local': [event_for_cot]}

        # Build compute_text (Part 2 derivation). Point is exact so diff is always 0.
        threshold_note = get_threshold_note(self.threshold_provided, self.threshold)
        proximity = f"{threshold_note} anchor={point}. |{point}-{point}|=0 <= {self.threshold}."
        if param_type == 'timing':
            compute_text = f"{proximity} {phase_name} phase lasts {value} timesteps."
        elif param_type == 'followup':
            compute_text = f"{proximity} followup type = {followup}."
        elif param_type == 'count':
            compute_text = f"{proximity} spike count = {n}."
        else:  # amplitude
            compute_text = f"{proximity} amplitude = {format_float(event_amplitude, 2)}."

        tb = ThoughtBuilder()
        tb.metric(metric_name, cot_attr, ['local'])
        tb.line(compute_text)
        thought = tb.build()[0]

        self.questions.append(_fix_article(question))
        self.answers.append(f"{thought}{answer_suffix}")
        self.llm_prompts.append([])
        self.fields.append({"local": [i]})
        self.qa_types.append(QAType.DESCRIPTION)
        self.eval_tasks.append(QAType.DESCRIPTION)
        self.eval_metadatas.append({
            "target_point": point,
            "metric": metric_name,
            "threshold": self.threshold,
            "variant": "param",
            "param_type": param_type,
            "length": self.seq_len,
        })
