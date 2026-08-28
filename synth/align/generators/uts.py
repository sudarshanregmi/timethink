"""UTS (Univariate Time Series) QA Generator."""

import random
from typing import Dict, List

import numpy as np

from synth.ts_generator.utils.common_utils import (
    get_local_event_positions,
    DEFAULT_THRESHOLD,
)
from synth.ts_generator.utils.probability_utils import QAType
from synth.align.config import Config, Difficulty, PromptIndexer
from synth.align.base_generator import BaseQAGenerator
from synth.align.types import GenerationResult


class UTSQAGenerator(BaseQAGenerator):

    def __init__(
        self,
        config: Config,
        indexer: PromptIndexer,
        timeseries: np.ndarray,
        attributes: Dict,
        metric: str,
        category: str,
        seq_len: int,
        threshold: int = DEFAULT_THRESHOLD,
        threshold_provided: bool = False,
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
        self.timeseries = timeseries
        self.attributes = attributes
        self.metric = metric
        self.category = category

    def generate_all_qa(self) -> GenerationResult:
        self._generate_description_qa(
            timeseries=self.timeseries,
            attributes=self.attributes,
            metric=self.metric,
            situation=self.category,
            series_index=None
        )

        self._generate_yes_no_qa()

        self._generate_compound_judgment_qa(
            attributes=self.attributes,
            metric=self.metric,
            series_index=None,
        )

        self._generate_segment_trend_dominance_qa(
            attributes=self.attributes,
            metric=self.metric,
            series_index=None,
        )

        self._generate_stat_numerical_qa(
            attributes=self.attributes,
            metric=self.metric,
            series_index=None,
            timeseries=self.timeseries,
        )

        self._generate_periodicity_qa(
            attributes=self.attributes,
            metric=self.metric,
            series_index=None,
        )

        self._generate_local_enumeration_qa(
            attributes=self.attributes,
            metric=self.metric,
            series_index=None,
        )

        self._generate_segment_enumeration_qa(
            attributes=self.attributes,
            metric=self.metric,
            series_index=None,
        )

        self._generate_transition_enumeration_qa(
            attributes=self.attributes,
            metric=self.metric,
            series_index=None,
        )

        self._generate_event_segment_enumeration_qa(
            attributes=self.attributes,
            metric=self.metric,
            series_index=None,
        )

        self._generate_temporal_position_qa(
            attributes=self.attributes,
            metric=self.metric,
            series_index=None,
        )

        self._generate_duration_proportion_qa(
            attributes=self.attributes,
            metric=self.metric,
            series_index=None,
        )

        self._generate_change_point_qa(
            attributes=self.attributes,
            metric=self.metric,
            series_index=None,
        )

        # New taxonomy: atomic SFT + RL compositions + bridge
        self._generate_taxonomy_qa(
            timeseries=self.timeseries,
            attributes=self.attributes,
            metric=self.metric,
            series_index=None,
        )

        return GenerationResult(
            questions=self.questions,
            answers=self.answers,
            llm_prompts=self.llm_prompts,
            fields=self.fields,
            qa_types=self.qa_types,
            eval_tasks=self.eval_tasks,
            eval_metadatas=self.eval_metadatas,
        )

    def _generate_yes_no_qa(self) -> None:
        num_questions = random.randint(0, 3)
        if self.config.debug:
            num_questions = max(1, num_questions)
        if num_questions == 0:
            return
        
        local_positions = get_local_event_positions(self.attributes)
        has_events = len(local_positions) > 0
        
        for _ in range(num_questions):
            # 50/50 verdict balancing
            want_yes = random.random() < 0.5
            query_point = self._select_yes_no_query_point(
                local_positions, has_events, want_yes=want_yes
            )
            # Skip if desired verdict is unachievable — never fallback to
            # opposite verdict, as this causes class imbalance drift.
            if query_point is None:
                continue

            if query_point is not None:
                self._generate_single_yes_no_qa(
                    attributes=self.attributes,
                    metric=self.metric,
                    query_point=query_point,
                    series_index=None
                )
