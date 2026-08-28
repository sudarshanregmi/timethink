"""MTS Shape (Multiple Time Series with Trend/Shape Focus) QA Generator."""

import random
from typing import Dict, List, Optional, Set, Tuple

from synth.ts_generator.utils.common_utils import article, DEFAULT_THRESHOLD, format_float
from synth.align.templates.prompts import _fix_article
from synth.ts_generator.utils.trend_utils import generate_trend_prompt
from synth.ts_generator.utils.thinking import (
    build_trend_verification_thought,
    check_trend_match,
    ThoughtBuilder,
    build_anti_judgment_thought,
    build_cross_trend_thought,
    _anti_trend_type,
)
from synth.ts_generator.utils.probability_utils import QAType
from synth.align.config import Config, Difficulty, PromptIndexer
from synth.align.base_generator import BaseQAGenerator
from synth.align.templates import PromptRegistry
from synth.align.types import GenerationResult
from synth.align.templates.mts_questions import (
    CROSS_TREND_QUESTIONS,
    TREND_TYPE_VERBS,
    ANTI_JUDGMENT_LABELS,
    ANTI_JUDGMENT_QUESTIONS,
)


class MTSShapeQAGenerator(BaseQAGenerator):

    def __init__(
        self,
        config: Config,
        indexer: PromptIndexer,
        situation: str,
        metrics: List[str],
        attributes: List[Dict],
        cluster_indices: List[Optional[int]],
        points_list: List[List],
        metric_to_cluster: Dict[str, str],
        seq_len: int,
        threshold: int = DEFAULT_THRESHOLD,
        threshold_provided: bool = False,
        difficulty: Difficulty = Difficulty.EASY,
        original_timeseries: Optional[List] = None,
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
        self.points_list = points_list
        self.metric_to_cluster = metric_to_cluster
        self.original_timeseries = original_timeseries or []
        
        # Additional tracking for MTS
        self.correlations: List[Dict] = []
        self.clusters: List[Dict] = []
        self.appended_clusters: Set[Tuple] = set()
        self.appended_anticlusters: Set[Tuple] = set()

    def generate_all_qa(self) -> GenerationResult:
        corr_pool_list = [None] * len(self.metrics)

        for i in range(len(self.metrics)):
            # Trend question
            self._generate_trend_question(i)

            # Correlation and anticorrelation questions
            for j in range(len(self.metrics)):
                if i == j:
                    continue
                self._generate_correlation_question(i, j)
                self._generate_anticorrelation_question(i, j)

            # Cluster question
            result = self._generate_cluster_question(i)
            if result:
                corr_pool_list[i] = result

            # Anticluster question
            self._generate_anticluster_question(i)

            # Anti-trend noise judgment (multi-metric compound)
            for j in range(len(self.metrics)):
                if i == j:
                    continue
                self._generate_anti_judgment_qa(i, j)
                self._generate_cross_trend_qa(i, j)

            # Per-metric segment QAs (inherited from BaseQAGenerator)
            self._generate_compound_judgment_qa(
                attributes=self.attributes[i],
                metric=self.metrics[i],
                series_index=i,
            )
            self._generate_segment_trend_dominance_qa(
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
            self._generate_transition_enumeration_qa(
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
            if i < len(self.original_timeseries):
                self._generate_taxonomy_qa(
                    timeseries=self.original_timeseries[i],
                    attributes=self.attributes[i],
                    metric=self.metrics[i],
                    series_index=i,
                )

        self._generate_compound_trend_description_qa()

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
            correlations=self.correlations,
            clusters=self.clusters,
            corr_pool_list=corr_pool_list,
            eval_tasks=self.eval_tasks,
            eval_metadatas=self.eval_metadatas,
        )

    def _generate_trend_question(self, i: int) -> None:
        question = PromptRegistry.get_prompt(
            QAType.DESCRIPTION,
            context={"metric": self.metrics[i]},
            augment=True,
            sub_key="trend",
        )
        
        tb = ThoughtBuilder()
        tb.metric(self.metrics[i], self.attributes[i], ['trend'])
        thought = tb.build()[0]
        
        overall_trend = self.attributes[i].get('trend', {}).get('detail', '')
        if not overall_trend:
            raise ValueError(
                f"trend.detail missing for metric {self.metrics[i]}; "
                f"shape-mode attributes must always have trend.detail set by derive_trend_info"
            )
        
        answer = thought + overall_trend
        
        self.questions.append(_fix_article(question))
        self.answers.append(answer)
        self.llm_prompts.append([])
        self.fields.append({'trend': [i]})
        self.qa_types.append(QAType.DESCRIPTION)
        self.eval_tasks.append(QAType.DESCRIPTION)
        self.eval_metadatas.append({"perspectives": ["trend"], "metric_index": i, "length": self.seq_len})

    def _generate_correlation_question(self, i: int, j: int) -> None:
        is_similar_pre = check_trend_match(
            self.attributes[i], self.attributes[j],
            mode='trend', threshold=self.threshold,
        )

        thought, _, verdict = build_trend_verification_thought(
            [self.attributes[i], self.attributes[j]],
            [self.metrics[i], self.metrics[j]],
            include_attributes=['trend_list'],
            mode='trend',
            show_verdict=True,
            threshold=self.threshold,
            threshold_provided=self.threshold_provided,
        )
        is_similar = verdict == "similar"

        prompt_idx = self.indexer.next()

        tolerance_text = (
            f" Trend transitions occurring within {self.threshold} timesteps of each other "
            f"should be considered synchronized."
            if self.threshold_provided else ""
        )

        question = PromptRegistry.get_prompt(
            QAType.CORRELATION,
            context={
                "metric_a": self.metrics[i],
                "metric_b": self.metrics[j],
                "tolerance": tolerance_text,
            },
            augment=True,
            sub_key="shape_correlation",
        )

        if is_similar:
            answer = (
                f"{thought}Both time series are showing similar trends, "
                f"indicating a correlation in terms of trend: "
                f"{generate_trend_prompt(self.points_list[i], seq_len=None)} "
                f"<|prompt{prompt_idx}|>"
            )
            llm_prompt = (
                f"In {article(self.situation)} {self.situation} system, {self.metrics[i]} and {self.metrics[j]} have similar trends. "
                f"Explain why in one sentence."
            )
            if self.metric_to_cluster.get(self.metrics[i]) == self.metric_to_cluster.get(self.metrics[j]):
                llm_prompt += f" (Hint: Both {self.metric_to_cluster[self.metrics[i]]}-related.)"

            self.correlations.append({
                "pair": [self.metrics[i], self.metrics[j]],
                "explain": f"<|prompt{prompt_idx}|>",
                "label": True
            })
        else:
            a, b = (i, j) if self.cluster_indices[i] is not None else (j, i)
            answer = (
                f"{thought}{self.metrics[b]} shows a different trend pattern from {self.metrics[a]}. "
                f"{self.metrics[a]}: {generate_trend_prompt(self.points_list[a], seq_len=None)} "
                f"{self.metrics[b]}: {generate_trend_prompt(self.points_list[b], seq_len=None)} "
                f"<|prompt{prompt_idx}|>"
            )
            llm_prompt = (
                f"In {article(self.situation)} {self.situation} system, {self.metrics[i]} and {self.metrics[j]} do not have similar trends. "
                f"Explain why in one sentence."
            )

            self.correlations.append({
                "pair": [self.metrics[i], self.metrics[j]],
                "explain": f"<|prompt{prompt_idx}|>",
                "label": False
            })

        self.questions.append(_fix_article(question))
        self.answers.append(answer)
        self.llm_prompts.append([llm_prompt])
        self.fields.append({'trend': [i, j]})
        self.qa_types.append(QAType.CORRELATION)
        self.eval_tasks.append(QAType.CORRELATION)
        self.eval_metadatas.append({
            "metric_a": self.metrics[i],
            "metric_b": self.metrics[j],
            "mode": "shape",
            "length": self.seq_len,
            "verdict": "similar" if is_similar else "different",
        })

    def _generate_cluster_question(self, i: int) -> Optional[List]:
        # Lightweight pre-check: count matching metrics before expensive thought construction.
        others = [idx for idx in range(len(self.metrics)) if idx != i]
        pre_match_count = 1  # anchor always included
        for j in others:
            if check_trend_match(self.attributes[i], self.attributes[j],
                                 mode='trend', threshold=self.threshold):
                pre_match_count += 1
        if pre_match_count <= 1 and random.random() > 0.25:
            return None

        tolerance_text = (
            f" Trend transitions occurring within {self.threshold} timesteps of each other "
            f"should be considered synchronized."
            if self.threshold_provided else ""
        )

        question = PromptRegistry.get_prompt(
            QAType.CLUSTERING,
            context={
                "metric": self.metrics[i],
                "tolerance": tolerance_text,
            },
            augment=True,
            sub_key="shape_clustering",
        )

        # 1 vs N reasoning — cluster is determined by thought block
        ordered_indices = [i] + others
        ordered_metrics = [self.metrics[idx] for idx in ordered_indices]
        ordered_attrs = [self.attributes[idx] for idx in ordered_indices]

        thought, cluster_metrics, _ = build_trend_verification_thought(
            ordered_attrs,
            ordered_metrics,
            include_attributes=['trend_list'],
            mode='trend',
            show_cluster=True,
            threshold=self.threshold,
            threshold_provided=self.threshold_provided,
        )

        # Convert cluster metric names to indices
        metric_to_idx = {name: idx for idx, name in enumerate(self.metrics)}
        related_indices = sorted(metric_to_idx[name] for name in cluster_metrics)
        related_metrics = [self.metrics[j] for j in related_indices]

        # No cluster — skip 50% of negative answers
        if len(related_indices) <= 1:
            if random.random() > 0.25:
                return None
            self.questions.append(_fix_article(question))
            self.answers.append(f"{thought}No related time series found.")
            self.llm_prompts.append([])
            self.fields.append({'trend': [i]})
            self.qa_types.append(QAType.CLUSTERING)
            self.eval_tasks.append(QAType.CLUSTERING)
            self.eval_metadatas.append({"target_metric": self.metrics[i], "mode": "shape", "length": self.seq_len})
            return None

        prompt_idx = self.indexer.next()

        answer = (
            f"{thought}Related: {', '.join(related_metrics)}. "
            f"Similar trends: {generate_trend_prompt(self.points_list[i], seq_len=None)} "
            f"<|prompt{prompt_idx}|>"
        )

        cluster_key = tuple(related_indices)
        if cluster_key not in self.appended_clusters:
            self.clusters.append({
                'col_idx': related_indices,
                'cols': related_metrics,
                'explain': f"<|prompt{prompt_idx}|>"
            })
            self.appended_clusters.add(cluster_key)

        llm_prompt = (
            f"In {article(self.situation)} {self.situation} system, {', '.join(related_metrics)} have similar trends. "
            f"Explain in 1 sentence."
        )

        self.questions.append(_fix_article(question))
        self.answers.append(answer)
        self.llm_prompts.append([llm_prompt])
        self.fields.append({'trend': related_indices})
        self.qa_types.append(QAType.CLUSTERING)
        self.eval_tasks.append(QAType.CLUSTERING)
        self.eval_metadatas.append({"target_metric": self.metrics[i], "mode": "shape", "length": self.seq_len})

        corr_pool = [related_indices, answer]
        return corr_pool

    def _generate_anticorrelation_question(self, i: int, j: int) -> None:
        is_anti_pre = check_trend_match(
            self.attributes[i], self.attributes[j],
            mode='anti_trend', threshold=self.threshold,
        )

        thought, _, verdict = build_trend_verification_thought(
            [self.attributes[i], self.attributes[j]],
            [self.metrics[i], self.metrics[j]],
            include_attributes=['trend_list'],
            mode='anti_trend',
            show_verdict=True,
            threshold=self.threshold,
            threshold_provided=self.threshold_provided,
        )

        is_anticorrelated = (verdict == "opposite")

        prompt_idx = self.indexer.next()

        tolerance_text = (
            f" Trend transitions occurring within {self.threshold} timesteps of each other "
            f"should be considered synchronized."
            if self.threshold_provided else ""
        )

        question = PromptRegistry.get_prompt(
            QAType.ANTICORRELATION,
            context={
                "metric_a": self.metrics[i],
                "metric_b": self.metrics[j],
                "tolerance": tolerance_text,
            },
            augment=True,
            sub_key="shape_anticorrelation",
        )

        if is_anticorrelated:
            answer = (
                f"{thought}{self.metrics[i]} and {self.metrics[j]} show opposite trend patterns (anticorrelated). "
                f"{self.metrics[i]}: {generate_trend_prompt(self.points_list[i], seq_len=None)} "
                f"{self.metrics[j]}: {generate_trend_prompt(self.points_list[j], seq_len=None)} "
                f"<|prompt{prompt_idx}|>"
            )
            llm_prompt = (
                f"In {article(self.situation)} {self.situation} system, {self.metrics[i]} and {self.metrics[j]} "
                f"have opposite trends (anticorrelated). Explain why in one sentence."
            )
            self.correlations.append({
                "pair": [self.metrics[i], self.metrics[j]],
                "explain": f"<|prompt{prompt_idx}|>",
                "label": True,
                "type": "anticorrelation",
            })
        else:
            a, b = (i, j) if self.cluster_indices[i] is not None else (j, i)
            answer = (
                f"{thought}{self.metrics[i]} and {self.metrics[j]} do not have opposite trends. "
                f"{self.metrics[a]}: {generate_trend_prompt(self.points_list[a], seq_len=None)} "
                f"{self.metrics[b]}: {generate_trend_prompt(self.points_list[b], seq_len=None)} "
                f"<|prompt{prompt_idx}|>"
            )
            llm_prompt = (
                f"In {article(self.situation)} {self.situation} system, {self.metrics[i]} and {self.metrics[j]} "
                f"do not have opposite trends. Explain why in one sentence."
            )
            self.correlations.append({
                "pair": [self.metrics[i], self.metrics[j]],
                "explain": f"<|prompt{prompt_idx}|>",
                "label": False,
                "type": "anticorrelation",
            })

        self.questions.append(_fix_article(question))
        self.answers.append(answer)
        self.llm_prompts.append([llm_prompt])
        self.fields.append({'trend': [i, j]})
        self.qa_types.append(QAType.ANTICORRELATION)
        self.eval_tasks.append(QAType.ANTICORRELATION)
        self.eval_metadatas.append({
            "metric_a": self.metrics[i],
            "metric_b": self.metrics[j],
            "mode": "anti_shape",
            "length": self.seq_len,
            "verdict": "opposite" if is_anticorrelated else "not_opposite",
        })

    def _generate_anticluster_question(self, i: int) -> None:
        # Lightweight pre-check: count anti-matching metrics before expensive thought construction.
        others = [idx for idx in range(len(self.metrics)) if idx != i]
        pre_match_count = 1  # anchor always included
        for j in others:
            if check_trend_match(self.attributes[i], self.attributes[j],
                                 mode='anti_trend', threshold=self.threshold):
                pre_match_count += 1
        if pre_match_count <= 1 and random.random() > 0.25:
            return

        tolerance_text = (
            f" Trend transitions occurring within {self.threshold} timesteps of each other "
            f"should be considered synchronized."
            if self.threshold_provided else ""
        )

        question = PromptRegistry.get_prompt(
            QAType.ANTICLUSTERING,
            context={
                "metric": self.metrics[i],
                "tolerance": tolerance_text,
            },
            augment=True,
            sub_key="shape_anticlustering",
        )

        ordered_indices = [i] + others
        ordered_metrics = [self.metrics[idx] for idx in ordered_indices]
        ordered_attrs = [self.attributes[idx] for idx in ordered_indices]

        thought, anticluster_metrics, _ = build_trend_verification_thought(
            ordered_attrs,
            ordered_metrics,
            include_attributes=['trend_list'],
            mode='anti_trend',
            show_cluster=True,
            threshold=self.threshold,
            threshold_provided=self.threshold_provided,
        )

        metric_to_idx = {name: idx for idx, name in enumerate(self.metrics)}
        related_indices = sorted(metric_to_idx[name] for name in anticluster_metrics)
        related_metrics = [self.metrics[j] for j in related_indices]
        # Filter anchor out of answer text — "opposite trends to X" must not list X itself.
        others_metrics = [name for name in related_metrics if name != self.metrics[i]]

        if len(related_indices) <= 1:
            if random.random() > 0.25:
                return
            self.questions.append(_fix_article(question))
            self.answers.append(f"{thought}No time series show opposite trends to {self.metrics[i]}.")
            self.llm_prompts.append([])
            self.fields.append({'trend': [i]})
            self.qa_types.append(QAType.ANTICLUSTERING)
            self.eval_tasks.append(QAType.ANTICLUSTERING)
            self.eval_metadatas.append({"target_metric": self.metrics[i], "mode": "anti_shape", "length": self.seq_len})
            return

        prompt_idx = self.indexer.next()

        answer = (
            f"{thought}The following time series show opposite trends to {self.metrics[i]}: "
            f"{', '.join(others_metrics)}. "
            f"{self.metrics[i]}: {generate_trend_prompt(self.points_list[i], seq_len=None)} "
            f"<|prompt{prompt_idx}|>"
        )

        anticluster_key = tuple(related_indices)
        if anticluster_key not in self.appended_anticlusters:
            self.clusters.append({
                'col_idx': related_indices,
                'cols': related_metrics,
                'explain': f"<|prompt{prompt_idx}|>",
                'type': 'anticlustering',
            })
            self.appended_anticlusters.add(anticluster_key)

        llm_prompt = (
            f"In {article(self.situation)} {self.situation} system, {', '.join(others_metrics)} have opposite trends "
            f"to {self.metrics[i]}. Explain in 1 sentence."
        )

        self.questions.append(_fix_article(question))
        self.answers.append(answer)
        self.llm_prompts.append([llm_prompt])
        self.fields.append({'trend': related_indices})
        self.qa_types.append(QAType.ANTICLUSTERING)
        self.eval_tasks.append(QAType.ANTICLUSTERING)
        self.eval_metadatas.append({"target_metric": self.metrics[i], "mode": "anti_shape", "length": self.seq_len})

    # ------------------------------------------------------------------ #
    # Anti-trend noise compound judgment (multi-metric Family 7 analog)   #
    # ------------------------------------------------------------------ #
    def _generate_anti_judgment_qa(self, i: int, j: int) -> None:
        tl_a = [s for s in self.attributes[i].get('trend_list', []) if isinstance(s, (list, tuple)) and len(s) >= 3]
        tl_b = [s for s in self.attributes[j].get('trend_list', []) if isinstance(s, (list, tuple)) and len(s) >= 3]
        if not tl_a or not tl_b:
            return

        ns_b = self.attributes[j].get('noise', {})
        if not isinstance(ns_b, dict) or 'strength' not in ns_b:
            return
        act_noise_amp = round(ns_b['strength'], 2)
        act_noise_type = ns_b['type']

        # Check anticorrelation
        anti_match = (len(tl_a) == len(tl_b)) and all(
            sb[0] == _anti_trend_type(sa[0]) for sa, sb in zip(tl_a, tl_b)
        )

        # Only generate anti_judgment for actually anti-correlated pairs.
        # Non-anti-correlated pairs would produce 100% "no" — skip them.
        if not anti_match:
            return

        # Check what verdicts are achievable
        can_yes = act_noise_amp > 0.0  # anti_match already guaranteed
        can_no = True  # always achievable (high threshold)

        # Coin flip only among achievable verdicts
        if can_yes and can_no:
            desired_yes = random.random() < 0.5
        elif can_yes:
            desired_yes = True
        else:
            desired_yes = False

        if desired_yes:
            threshold = round(act_noise_amp * random.uniform(0.3, 0.85), 2)
            c2_met = round(act_noise_amp, 2) > threshold
            verdict = 'yes' if c2_met else 'no'
        else:
            if not anti_match:
                threshold = round(random.uniform(0.03, 0.12), 2)
                verdict = 'no'
            elif act_noise_amp == 0.0:
                threshold = round(random.uniform(0.03, 0.12), 2)
                verdict = 'no'
            else:
                threshold = round(act_noise_amp * random.uniform(1.1, 2.5) + 1e-8, 2)
                c2_met = round(act_noise_amp, 2) > threshold
                verdict = 'yes' if c2_met else 'no'

        label = random.choice(ANTI_JUDGMENT_LABELS)

        params = {
            'label': label,
            'threshold': threshold,
            'noise_category': 'noisy' if 'noisy' in act_noise_type else 'smooth',
            'actual_noise_amp': act_noise_amp,
            'actual_noise_type': act_noise_type,
            'verdict': verdict,
        }

        think_str, _ = build_anti_judgment_thought(
            self.attributes[i], self.attributes[j],
            self.metrics[i], self.metrics[j],
            'anti_trend_noise', params
        )

        question = random.choice(ANTI_JUDGMENT_QUESTIONS).format(
            label=label,
            metric_a=self.metrics[i],
            metric_b=self.metrics[j],
            threshold=threshold,
        )
        if verdict == 'yes':
            answer_text = (
                f"Yes. {self.metrics[i]} and {self.metrics[j]} show anticorrelated trends, "
                f"and {self.metrics[j]}'s noise strength ({format_float(act_noise_amp, 2)}) "
                f"exceeds the threshold ({format_float(threshold, 2)})."
            )
        else:
            answer_text = (
                f"No. {self.metrics[i]} and {self.metrics[j]} do not jointly satisfy "
                f"the anticorrelation and noise conditions."
            )

        self.questions.append(_fix_article(question))
        self.answers.append(think_str + answer_text)
        self.llm_prompts.append([])
        self.fields.append({'trend': [i, j]})
        self.qa_types.append(QAType.SEGMENT_MASK)
        self.eval_tasks.append("anti_judgment")
        self.eval_metadatas.append({
            'judgment_type': 'anti_trend_noise',
            'verdict': verdict,
            'length': self.seq_len,
        })

    # ------------------------------------------------------------------ #
    # Cross-trend range query (multi-metric, hardest)                     #
    # ------------------------------------------------------------------ #
    def _generate_cross_trend_qa(self, i: int, j: int) -> None:
        tl_a = [s for s in self.attributes[i].get('trend_list', []) if isinstance(s, (list, tuple)) and len(s) >= 3]
        tl_b = [s for s in self.attributes[j].get('trend_list', []) if isinstance(s, (list, tuple)) and len(s) >= 3]
        if not tl_a or not tl_b:
            return

        # Pick a trend type that A has
        types_in_a = list({s[0] for s in tl_a})
        if not types_in_a:
            return
        target_type = random.choice(types_in_a)

        think_str, verdict = build_cross_trend_thought(
            self.attributes[i], self.attributes[j],
            self.metrics[i], self.metrics[j],
            target_type
        )

        if verdict == 'none':
            return  # skip if A doesn't actually have the target type (shouldn't happen)

        trend_verb = TREND_TYPE_VERBS.get(target_type, target_type)
        question = random.choice(CROSS_TREND_QUESTIONS).format(
            metric_a=self.metrics[i],
            metric_b=self.metrics[j],
            trend_type=target_type,
            trend_type_verb=trend_verb,
        )

        direction_text = {
            'increase': 'an increasing (upward) trend',
            'decrease': 'a decreasing (downward) trend',
            'keep steady': 'a steady (flat) trend',
            'equal': 'equal coverage of multiple trend types',
        }.get(verdict, verdict)
        answer = (
            think_str +
            f"When {self.metrics[i]} is {target_type}, {self.metrics[j]} predominantly shows {direction_text}."
        )

        self.questions.append(_fix_article(question))
        self.answers.append(answer)
        self.llm_prompts.append([])
        self.fields.append({'trend': [i, j]})
        self.qa_types.append(QAType.SEGMENT_MASK)
        self.eval_tasks.append("cross_trend_query")
        self.eval_metadatas.append({
            'target_metric_a': self.metrics[i],
            'target_trend_type': target_type,
            'verdict': verdict,
            'length': self.seq_len,
        })

    def _generate_compound_trend_description_qa(self) -> None:
        """Generate a compound description QA asking about trend across multiple metrics."""
        n = len(self.metrics)
        if n < 2:
            return

        k = random.randint(2, min(n, 4))
        chosen_idx = random.sample(range(n), k)

        chosen_metrics = [self.metrics[i] for i in chosen_idx]
        chosen_attrs = [self.attributes[i] for i in chosen_idx]

        tb = ThoughtBuilder()
        for j, (name, attr) in enumerate(zip(chosen_metrics, chosen_attrs)):
            if j > 0:
                tb.metric_separator()
            tb.metric(name, attr or {}, ['trend'])
        thought_block = tb.build()[0]

        metrics_str = ", ".join(chosen_metrics[:-1]) + f" and {chosen_metrics[-1]}"
        question = f"Describe the trend of {metrics_str}."

        answer_parts = []
        for metric, attrs in zip(chosen_metrics, chosen_attrs):
            detail = attrs['trend']['detail']
            answer_parts.append(f"{metric}: {detail}")

        full_answer = thought_block + " ".join(answer_parts)

        self.questions.append(_fix_article(question))
        self.answers.append(full_answer)
        self.llm_prompts.append([])
        self.fields.append({'trend': list(chosen_idx)})
        self.qa_types.append(QAType.DESCRIPTION)
        self.eval_tasks.append(QAType.DESCRIPTION)
        self.eval_metadatas.append({
            "perspectives": {m: ["trend"] for m in chosen_metrics},
            "length": self.seq_len,
        })
