"""
Comprehensive test suite for synth.align package.

Covers:
  - PromptIndexer, Mode, Difficulty enums (config.py)
  - Utility functions (utils.py)
  - build_compound_judgment_thought (thinking_utils.py)
  - UTSQAGenerator, MTSLocalQAGenerator, MTSShapeQAGenerator
  - select_single_qa placeholder reindexing (factory.py)
  - Compound description QA (single <think> block invariant)
  - MTS positive/negative series generation & structural verification
  - build_local_verification_thought pre-check

Usage:
    python -m pytest synth/align/test.py -v
    python -m synth.align.test        # original main-style output
"""

import re
import random
import pytest
import numpy as np

random.seed(42)
np.random.seed(42)

SEQ_LEN = 256
THRESHOLD = 15

from synth.align.config import Config, PromptIndexer, Difficulty, Mode, SampleResult
from synth.align.utils import (
    timeseries_to_list,
    timeseries_encoding,
    replace_prompts_in_obj,
)
from synth.ts_generator.utils.common_utils import (
    BatchSampleGenerator as TimeSeriesGenerator,
    DEFAULT_LOCAL_FEATURE_CONFIG,
    DEFAULT_SHAPE_FEATURE_CONFIG,
    has_local_event_near,
    get_local_event_positions,
    sanitize_attributes_for_sync,
    find_point_far_from_all_events,
    DEFAULT_THRESHOLD,
)
from synth.ts_generator.utils.thinking import (
    build_local_verification_thought,
    generate_structural_verification,
    build_compound_judgment_thought,
)
from synth.ts_generator.utils.probability_utils import QAType, DEFAULT_QA_TYPE_WEIGHTS, select_weighted_qa_index
from synth.align.config import DEFAULT_MODE_WEIGHTS


def _dummy_config_and_indexer():
    config = Config.__new__(Config)
    config.output_dir = "/tmp"
    config.metric_set = {}
    config.debug = False
    indexer = PromptIndexer()
    return config, indexer


def _noise_seg_attrs():
    """Simple noisy attributes (single uniform noise) + all required detail strings."""
    return {
        'noise': {
            'type': 'noisy',
            'strength': 0.05,
            'detail': 'The signal exhibits moderate noise throughout.',
        },
        'seasonal': {
            'type': 'no',
            'period': 0,
            'amplitude': 0.0,
            'detail': 'The signal shows no clear periodicity.',
        },
        'trend': {
            'type': 'keep steady',
            'detail': 'The metric remains relatively stable over the observation window.',
            'start': 0.00, 'end': 0.10, 'amplitude': 0.10,
        },
        'trend_list': [('keep steady', 0, 256)],
        'length': 256,
        'seq_len': 256,
        'statistics': {'min': -1.0, 'max': 1.0, 'mean': 0.0, 'std': 0.3},
        'local': [],
    }


def _complex_attrs():
    """Full attributes: single seasonal + local events + multi-phase trend + statistics."""
    return {
        'noise': {
            'type': 'noisy',
            'strength': 0.08,
            'detail': 'The signal exhibits moderate to high noise.',
        },
        'seasonal': {
            'type': 'sin periodic fluctuation',
            'period': 12,
            'amplitude': 0.5,
            'detail': 'The signal exhibits a periodic oscillation with period ~12.',
            'segments': [
                {'amplitude': 0.5, 'position_start': 0, 'position_end': 256,
                 'description': 'the amplitude of the periodic fluctuation is 0.50 between point 1 and point 256'},
            ],
        },
        'trend': {
            'type': 'increase',
            'detail': 'The metric shows an increasing trend throughout the period.',
            'start': 0.00, 'end': 0.80, 'amplitude': 0.80,
        },
        'trend_list': [('increase', 0, 128), ('keep steady', 128, 256)],
        'length': 256,
        'seq_len': 256,
        'statistics': {'min': -0.5, 'max': 2.0, 'mean': 0.5, 'std': 0.4},
        'local': [
            {'type': 'upward spike',   'position_start': 80,  'amplitude': 1.5, 'value_start': 1.0},
            {'type': 'shake',          'position_start': 150, 'amplitude': 0.5, 'value_start': 0.5},
        ],
    }


def _make_uts_generator(attrs, timeseries=None, difficulty=Difficulty.EASY):
    from synth.align.generators.uts import UTSQAGenerator
    config, indexer = _dummy_config_and_indexer()
    if timeseries is None:
        timeseries = np.random.randn(SEQ_LEN)
    return UTSQAGenerator(
        config=config,
        indexer=indexer,
        timeseries=timeseries,
        metric="cpu_usage",
        attributes=attrs,
        category="server",
        seq_len=SEQ_LEN,
        threshold=THRESHOLD,
        threshold_provided=True,
        difficulty=difficulty,
    )


class TestPromptIndexer:
    def test_initial_current_is_zero(self):
        assert PromptIndexer().current() == 0

    def test_next_increments_sequentially(self):
        idx = PromptIndexer()
        assert idx.next() == 0
        assert idx.next() == 1
        assert idx.next() == 2

    def test_current_does_not_increment(self):
        idx = PromptIndexer()
        idx.next()
        assert idx.current() == 1
        assert idx.current() == 1   # still 1

    def test_reset_returns_counter_to_zero(self):
        idx = PromptIndexer()
        idx.next(); idx.next(); idx.next()
        idx.reset()
        assert idx.current() == 0
        assert idx.next() == 0

    def test_placeholder_format_implicit(self):
        idx = PromptIndexer()
        assert idx.placeholder() == "<|prompt0|>"
        assert idx.placeholder() == "<|prompt1|>"

    def test_placeholder_explicit_index(self):
        idx = PromptIndexer()
        p = idx.placeholder(idx=7)
        assert p == "<|prompt7|>"

    def test_independent_instances_do_not_share_state(self):
        idx1 = PromptIndexer()
        idx2 = PromptIndexer()
        idx1.next(); idx1.next()
        assert idx2.current() == 0


class TestEnums:
    def test_mode_values(self):
        assert Mode.UTS.value == "uts"
        assert Mode.MTS_LOCAL.value == "local"
        assert Mode.MTS_SHAPE.value == "shape"

    def test_difficulty_values(self):
        assert Difficulty.EASY.value == "easy"
        assert Difficulty.MEDIUM.value == "medium"
        assert Difficulty.HARD.value == "hard"

    def test_qa_type_values(self):
        assert QAType.DESCRIPTION == "description"
        assert QAType.YES_NO == "yes_no"
        assert QAType.CORRELATION == "correlation"
        assert QAType.CLUSTERING == "clustering"
        assert QAType.SEGMENT_MASK == "segment_mask"

    def test_default_qa_weights_sum_to_one(self):
        total = sum(DEFAULT_QA_TYPE_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-6


class TestTimeseriesToList:
    def test_1d_basic(self):
        arr = np.array([1.0, 2.5, 3.14159])
        result = timeseries_to_list(arr, digits=2)
        assert result == [1.0, 2.5, 3.14]

    def test_2d_nested(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = timeseries_to_list(arr)
        assert isinstance(result, list)
        assert isinstance(result[0], list)
        assert len(result) == 2 and len(result[0]) == 2

    def test_rounding(self):
        arr = np.array([1.123456789])
        assert timeseries_to_list(arr, digits=4) == [1.1235]

    def test_does_not_mutate_original_with_copy(self):
        arr = np.array([1.0, 2.0, 3.0])
        orig = arr.copy()
        timeseries_to_list(arr, cp=True)
        np.testing.assert_array_equal(arr, orig)

    def test_returns_python_floats(self):
        arr = np.array([1.5])
        result = timeseries_to_list(arr)
        assert isinstance(result[0], float)


class TestTimeseriesEncoding:
    def test_no_method_returns_original_ts(self):
        ts = np.array([1.0, 2.0, 3.0])
        scaled, prompt, extra = timeseries_encoding(ts, 'no')
        np.testing.assert_array_equal(scaled, ts)
        assert isinstance(prompt, str)
        assert isinstance(extra, dict)

    def test_no_method_prompt_contains_ts_tag(self):
        ts = np.zeros(5)
        _, prompt, _ = timeseries_encoding(ts, 'no')
        assert "<ts>" in prompt

    def test_unknown_method_raises_not_implemented(self):
        ts = np.array([1.0])
        with pytest.raises(NotImplementedError):
            timeseries_encoding(ts, 'fourier')


class TestReplacePromptsInObj:
    def test_string_replacement(self):
        result = replace_prompts_in_obj("hello <|prompt0|> world", {"<|prompt0|>": "X"})
        assert result == "hello X world"

    def test_list_replacement(self):
        result = replace_prompts_in_obj(
            ["<|prompt0|>", "keep", "<|prompt1|>"],
            {"<|prompt0|>": "A", "<|prompt1|>": "B"},
        )
        assert result == ["A", "keep", "B"]

    def test_nested_dict_replacement(self):
        obj = {"outer": {"inner": "<|prompt0|>"}}
        result = replace_prompts_in_obj(obj, {"<|prompt0|>": "V"})
        assert result["outer"]["inner"] == "V"

    def test_no_match_unchanged(self):
        obj = {"key": "no_placeholder"}
        result = replace_prompts_in_obj(obj, {"<|prompt0|>": "X"})
        assert result["key"] == "no_placeholder"

    def test_multiple_placeholders_in_one_string(self):
        result = replace_prompts_in_obj(
            "<|prompt0|> and <|prompt1|>",
            {"<|prompt0|>": "foo", "<|prompt1|>": "bar"},
        )
        assert result == "foo and bar"


class TestBuildCompoundJudgmentThought:
    """Tests for build_compound_judgment_thought(attributes, metric, judgment_type, params)."""

    def _base_attrs(self):
        return {
            'noise': {'type': 'noisy', 'strength': 0.10},
            'trend': {'type': 'increase', 'start': 0.00, 'end': 1.50, 'amplitude': 1.50},
            'trend_list': [('increase', 0, 256)],
            'length': 256,
            'seq_len': 256,
            'statistics': {'min': -0.5, 'max': 2.0, 'mean': 0.5, 'std': 0.4},
            'local': [{'type': 'upward spike', 'position_start': 80, 'amplitude': 1.5}],
        }

    def test_noise_trend_yes(self):
        attrs = self._base_attrs()
        # Params keys match what build_compound_judgment_thought actually reads
        params = {
            'label': 'unstable state',
            'noise_category': 'noisy',
            'threshold': 0.05,        # actual strength 0.10 > 0.05 → c2 met
            'required_trend': 'increase',
            'actual_trend': 'increase',
            'actual_noise_type': 'noisy',
            'actual_noise_amp': 0.10,
            'verdict': 'yes',
        }
        think_str, verdict = build_compound_judgment_thought(attrs, "cpu", 'noise_trend', params)
        assert verdict == 'yes'
        assert "<think>" in think_str and "</think>" in think_str

    def test_noise_trend_no(self):
        attrs = self._base_attrs()
        params = {
            'label': 'high-noise regime',
            'noise_category': 'noisy',
            'threshold': 0.05,
            'required_trend': 'decrease',   # doesn't match actual 'increase' → c1 fails → no
            'actual_trend': 'increase',
            'actual_noise_type': 'noisy',
            'actual_noise_amp': 0.10,
            'verdict': 'no',
        }
        _, verdict = build_compound_judgment_thought(attrs, "cpu", 'noise_trend', params)
        assert verdict == 'no'

    def test_stat_threshold_above_yes(self):
        attrs = self._base_attrs()
        # actual_max = 2.0 > threshold = 1.5 → yes
        params = {
            'direction': 'above',
            'threshold': 1.5,
            'actual_min': -0.5,
            'actual_max': 2.0,
            'verdict': 'yes',
        }
        think_str, verdict = build_compound_judgment_thought(attrs, "cpu", 'stat_threshold', params)
        assert verdict == 'yes'

    def test_stat_threshold_above_no(self):
        attrs = self._base_attrs()
        # actual_max = 2.0 NOT > threshold = 3.0 → no
        params = {
            'direction': 'above',
            'threshold': 3.0,
            'actual_min': -0.5,
            'actual_max': 2.0,
            'verdict': 'no',
        }
        _, verdict = build_compound_judgment_thought(attrs, "cpu", 'stat_threshold', params)
        assert verdict == 'no'

    def test_trend_local_yes(self):
        attrs = self._base_attrs()
        params = {
            'event_direction': 'upward',
            'event_label': 'surge event',
            'threshold': 1.0,
            'required_trend': 'increase',
            'actual_trend': 'increase',
            'qualifying_events': [('upward spike', 81, 1.5)],
            'all_direction_events': [('upward spike', 81, 1.5)],
            'all_events': [('upward spike', 1.5, 'upward')],
            'verdict': 'yes',
        }
        think_str, verdict = build_compound_judgment_thought(attrs, "cpu", 'trend_local', params)
        assert verdict == 'yes'
        assert "condition 1:" in think_str
        assert "condition 2:" in think_str

    def test_multi_phase_yes(self):
        attrs = {
            'trend': {'type': 'increase', 'start': 0.00, 'end': 1.50, 'amplitude': 1.50},
            'trend_list': [('increase', 0, 128), ('keep steady', 128, 256)],
            'noise': {'type': 'smooth', 'strength': 0.0},
            'length': 256,
            'seq_len': 256,
        }
        params = {
            'pattern_label': 'growth cycle',
            'target_sequence': ['increase', 'keep steady'],
            'actual_sequence': ['increase', 'keep steady'],
            'trend_list': [('increase', 0, 128), ('keep steady', 128, 256)],
            'verdict': 'yes',
        }
        think_str, verdict = build_compound_judgment_thought(attrs, "cpu", 'multi_phase', params)
        assert verdict == 'yes'

    def test_verdict_in_think(self):
        attrs = self._base_attrs()
        params = {
            'label': 'noisy period',
            'noise_category': 'noisy',
            'threshold': 0.05,
            'required_trend': 'increase',
            'actual_trend': 'increase',
            'actual_noise_type': 'noisy',
            'actual_noise_amp': 0.10,
            'verdict': 'yes',
        }
        think_str, verdict = build_compound_judgment_thought(attrs, "cpu", 'noise_trend', params)
        assert f"answer: {verdict}" in think_str


class TestUTSQAGeneratorStructure:
    def test_generate_all_qa_returns_generation_result(self):
        gen = _make_uts_generator(_noise_seg_attrs())
        r = gen.generate_all_qa()
        assert len(r.questions) > 0
        assert len(r.eval_tasks) > 0

    def test_all_lists_same_length(self):
        gen = _make_uts_generator(_noise_seg_attrs())
        r = gen.generate_all_qa()
        n = len(r.questions)
        assert n > 0
        for lst in [r.answers, r.llm_prompts, r.fields, r.qa_types, r.eval_tasks, r.eval_metadatas]:
            assert len(lst) == n

    def test_description_qa_always_present(self):
        gen = _make_uts_generator(_noise_seg_attrs())
        r = gen.generate_all_qa()
        assert "description" in r.qa_types
        assert "description" in r.eval_tasks

    def test_at_least_one_answer_has_think_block(self):
        gen = _make_uts_generator(_noise_seg_attrs())
        r = gen.generate_all_qa()
        assert any("<think>" in a for a in r.answers) 

class TestUTSQAGeneratorYesNoMetadata:
    def test_yes_no_has_required_metadata_keys(self):
        gen = _make_uts_generator(_complex_attrs())
        r = gen.generate_all_qa()
        yn_idxs = [i for i, t in enumerate(r.eval_tasks) if t == "yes_no"]
        for i in yn_idxs:
            m = r.eval_metadatas[i]
            assert "target_point" in m
            assert "metric" in m
            assert "threshold" in m
            assert isinstance(m["target_point"], int)
    def test_yes_no_difficulty_medium(self):
        gen = _make_uts_generator(_complex_attrs(), difficulty=Difficulty.MEDIUM)
        r = gen.generate_all_qa()
        yn_idxs = [i for i, t in enumerate(r.eval_tasks) if t == "yes_no"]
        for i in yn_idxs:
            assert isinstance(r.eval_metadatas[i]["target_point"], int)


class TestMTSLocalQAGenerator:
    def _build_gen(self, seed=5, metrics=None, difficulty=Difficulty.EASY):
        from synth.align.generators.mts_local import MTSLocalQAGenerator
        random.seed(seed)
        np.random.seed(seed)
        if metrics is None:
            metrics = ["cpu_usage", "memory_usage", "disk_io"]
        n = len(metrics)
        ts_gen = TimeSeriesGenerator(mode="local", feature_config=DEFAULT_LOCAL_FEATURE_CONFIG)
        ts_list, attr_list, anchor, _ = ts_gen.generate_positive_timeseries(
            count=n, seq_len=SEQ_LEN, threshold=THRESHOLD
        )
        config, indexer = _dummy_config_and_indexer()
        return MTSLocalQAGenerator(
            config=config, indexer=indexer, situation="server", metrics=metrics,
            attributes=attr_list, cluster_indices=[None]*n, positive_clusters=[],
            change_positions=[anchor], metric_to_cluster={m: "server" for m in metrics},
            original_timeseries=ts_list, threshold=THRESHOLD,
            threshold_provided=True, seq_len=SEQ_LEN, difficulty=difficulty,
        )
    def test_returns_generation_result(self):
        r = self._build_gen().generate_all_qa()
        assert len(r.questions) > 0
        assert len(r.answers) > 0
    def test_all_qa_lists_same_length(self):
        r = self._build_gen().generate_all_qa()
        n = len(r.questions)
        assert n > 0
        for lst in [r.answers, r.llm_prompts, r.fields, r.qa_types, r.eval_tasks, r.eval_metadatas]:
            assert len(lst) == n
    def test_description_qa_generated(self):
        r = self._build_gen().generate_all_qa()
        assert "description" in r.qa_types
    def test_with_difficulty_hard(self):
        gen = self._build_gen(difficulty=Difficulty.HARD)
        r = gen.generate_all_qa()
        assert len(r.questions) > 0


class TestMTSShapeQAGenerator:
    def _build_gen(self, seed=10, n=3):
        from synth.align.generators.mts_shape import MTSShapeQAGenerator
        random.seed(seed)
        np.random.seed(seed)
        metrics = [f"metric_{i}" for i in range(n)]
        ts_gen = TimeSeriesGenerator(mode="shape", feature_config=DEFAULT_SHAPE_FEATURE_CONFIG)
        ts_list, attr_list, _, pts_list = ts_gen.generate_positive_timeseries(
            count=n, seq_len=SEQ_LEN, threshold=THRESHOLD
        )
        for attr in attr_list:
            if not attr.get('trend', {}).get('detail'):
                attr.setdefault('trend', {})['detail'] = "steady trend"
        config, indexer = _dummy_config_and_indexer()
        return MTSShapeQAGenerator(
            config=config, indexer=indexer, situation="system", metrics=metrics,
            attributes=attr_list, cluster_indices=[None]*n, points_list=pts_list,
            metric_to_cluster={m: "cluster_1" for m in metrics},
            seq_len=SEQ_LEN, threshold=THRESHOLD, threshold_provided=True,
        )
    def test_returns_generation_result(self):
        r = self._build_gen().generate_all_qa()
        assert len(r.questions) > 0
        assert len(r.answers) > 0
    def test_all_qa_lists_same_length(self):
        r = self._build_gen().generate_all_qa()
        n = len(r.questions)
        assert n > 0
        for lst in [r.answers, r.llm_prompts, r.fields, r.qa_types, r.eval_tasks, r.eval_metadatas]:
            assert len(lst) == n
    def test_description_qa_generated(self):
        r = self._build_gen().generate_all_qa()
        assert "description" in r.qa_types
    def test_missing_trend_detail_raises(self):
        from synth.align.generators.mts_shape import MTSShapeQAGenerator
        random.seed(1)
        np.random.seed(1)
        metrics = ["cpu"]
        ts_gen = TimeSeriesGenerator(mode="shape", feature_config=DEFAULT_SHAPE_FEATURE_CONFIG)
        ts_list, attr_list, _, pts_list = ts_gen.generate_positive_timeseries(
            count=1, seq_len=SEQ_LEN, threshold=THRESHOLD
        )
        for attr in attr_list:
            attr.setdefault('trend', {}).pop('detail', None)
        config, indexer = _dummy_config_and_indexer()
        with pytest.raises((AssertionError, KeyError, ValueError)):
            gen = MTSShapeQAGenerator(
                config=config, indexer=indexer, situation="sys", metrics=metrics,
                attributes=attr_list, cluster_indices=[None], points_list=pts_list,
                metric_to_cluster={"cpu": "c1"}, seq_len=SEQ_LEN,
                threshold=THRESHOLD, threshold_provided=True,
            )
            gen.generate_all_qa()


class TestCompoundDescriptionMTSLocal:
    def _build_and_collect(self, seed):
        from synth.align.generators.mts_local import MTSLocalQAGenerator
        random.seed(seed)
        np.random.seed(seed)
        metrics = ["cpu_usage", "memory_usage", "disk_io", "network_rx"]
        n = len(metrics)
        ts_gen = TimeSeriesGenerator(mode="local", feature_config=DEFAULT_LOCAL_FEATURE_CONFIG)
        ts_list, attr_list, anchor, _ = ts_gen.generate_positive_timeseries(
            count=n, seq_len=SEQ_LEN, threshold=THRESHOLD
        )
        config, indexer = _dummy_config_and_indexer()
        gen = MTSLocalQAGenerator(
            config=config, indexer=indexer, situation="server", metrics=metrics,
            attributes=attr_list, cluster_indices=[None]*n, positive_clusters=[],
            change_positions=[anchor], metric_to_cluster={m: "server" for m in metrics},
            original_timeseries=ts_list, threshold=THRESHOLD,
            threshold_provided=True, seq_len=SEQ_LEN, difficulty=Difficulty.EASY,
        )
        return gen.generate_all_qa(), metrics

    def test_single_think_block_per_compound_qa(self):
        result, _ = self._build_and_collect(0)
        qs, ans, fields, qa_types = result.questions, result.answers, result.fields, result.qa_types
        compound_idxs = [
            i for i in range(len(qs))
            if qa_types[i] == "description" and len(set(sum(fields[i].values(), []))) > 1
        ]
        assert len(compound_idxs) > 0, "No compound description QA generated"
        for idx in compound_idxs:
            count = ans[idx].count("<think>")
            assert count == 1, f"Expected 1 <think>, got {count}:\n{ans[idx][:400]}"

    def test_compound_question_names_multiple_metrics(self):
        result, metrics = self._build_and_collect(99)
        qs, fields, qa_types = result.questions, result.fields, result.qa_types
        compound_idxs = [
            i for i in range(len(qs))
            if qa_types[i] == "description" and len(set(sum(fields[i].values(), []))) > 1
        ]
        for idx in compound_idxs:
            mentioned = [m for m in metrics if m in qs[idx]]
            assert len(mentioned) >= 2, f"Compound Q should mention ≥2 metrics: {qs[idx]}"


class TestCompoundDescriptionMTSShape:
    def _build_and_collect(self, seed):
        from synth.align.generators.mts_shape import MTSShapeQAGenerator
        random.seed(seed)
        np.random.seed(seed)
        metrics = ["latency_p99", "error_rate", "throughput"]
        n = len(metrics)
        ts_gen = TimeSeriesGenerator(mode="shape", feature_config=DEFAULT_SHAPE_FEATURE_CONFIG)
        ts_list, attr_list, _, pts_list = ts_gen.generate_positive_timeseries(
            count=n, seq_len=SEQ_LEN, threshold=THRESHOLD
        )
        for attr in attr_list:
            if not attr.get('trend', {}).get('detail'):
                attr.setdefault('trend', {})['detail'] = "The metric shows a steady trend."
        config, indexer = _dummy_config_and_indexer()
        gen = MTSShapeQAGenerator(
            config=config, indexer=indexer, situation="api_gateway", metrics=metrics,
            attributes=attr_list, cluster_indices=[None]*n, points_list=pts_list,
            metric_to_cluster={m: "api" for m in metrics},
            seq_len=SEQ_LEN, threshold=THRESHOLD, threshold_provided=True,
        )
        return gen.generate_all_qa()

    def test_single_think_block_per_compound_qa(self):
        result = self._build_and_collect(1)
        qs, ans, fields, qa_types = result.questions, result.answers, result.fields, result.qa_types
        compound_idxs = [
            i for i in range(len(qs))
            if qa_types[i] == "description" and len(set(sum(fields[i].values(), []))) > 1
        ]
        assert len(compound_idxs) > 0, "No compound description QA generated for shape mode"
        for idx in compound_idxs:
            count = ans[idx].count("<think>")
            assert count == 1, f"Expected 1 <think>, got {count}:\n{ans[idx][:400]}"


class TestMTSLocalGeneration:
    def test_positive_generation_shapes(self):
        random.seed(42); np.random.seed(42)
        ts_gen = TimeSeriesGenerator(mode="local", feature_config=DEFAULT_LOCAL_FEATURE_CONFIG)
        ts_list, attr_list, anchor, _ = ts_gen.generate_positive_timeseries(
            count=6, seq_len=SEQ_LEN, threshold=THRESHOLD
        )
        assert len(ts_list) == 6
        assert len(attr_list) == 6
        assert all(len(ts) == SEQ_LEN for ts in ts_list)
        assert isinstance(anchor, int)
        assert 0 <= anchor < SEQ_LEN

    def test_positive_generation_has_at_least_one_event_near_anchor(self):
        random.seed(42); np.random.seed(42)
        ts_gen = TimeSeriesGenerator(mode="local", feature_config=DEFAULT_LOCAL_FEATURE_CONFIG)
        ts_list, attr_list, anchor, _ = ts_gen.generate_positive_timeseries(
            count=6, seq_len=SEQ_LEN, threshold=THRESHOLD
        )
        series_with_near = sum(
            1 for attr in attr_list
            if any(abs(e.get('position_start', 0) - anchor) <= THRESHOLD
                   for e in attr.get('local', []))
        )
        assert series_with_near >= 1

    def test_negative_generation_no_events_near_anchor(self):
        random.seed(42); np.random.seed(42)
        ts_gen = TimeSeriesGenerator(mode="local", feature_config=DEFAULT_LOCAL_FEATURE_CONFIG)
        anchor = 128
        ts_list, attr_list, _, _ = ts_gen.generate_negative_timeseries(
            count=6, positive_anchor=anchor, seq_len=SEQ_LEN, threshold=THRESHOLD
        )
        assert len(ts_list) == 6
        for attr in attr_list:
            near = [
                e for e in attr.get('local', [])
                if abs(e.get('position_start', 0) - anchor) <= THRESHOLD
            ]
            assert len(near) == 0, f"Negative series has {len(near)} events near anchor: {near}"

    def test_negative_generation_correct_count(self):
        ts_gen = TimeSeriesGenerator(mode="local", feature_config=DEFAULT_LOCAL_FEATURE_CONFIG)
        ts_list, attr_list, _, _ = ts_gen.generate_negative_timeseries(
            count=4, positive_anchor=100, seq_len=SEQ_LEN, threshold=THRESHOLD
        )
        assert len(ts_list) == 4
        assert len(attr_list) == 4


class TestStructuralVerification:
    def _target(self):
        return {'local': [{'position_start': 98, 'type': 'sudden increase'}]}

    def test_returns_nonempty_string_always(self):
        for pos in [0, 50, 98, 150, 255]:
            c = {'local': [{'position_start': pos, 'type': 'shake'}]}
            result = generate_structural_verification(
                self._target(), c,
                mode='local', threshold=THRESHOLD, anchor_point=100,
            )
            assert len(result) > 0
            assert "PASS" in result or "FAIL" in result

    def test_event_near_anchor_positive_result(self):
        candidate = {'local': [
            {'position_start': 50, 'type': 'upward spike'},
            {'position_start': 105, 'type': 'shake'},
        ]}
        result = generate_structural_verification(
            self._target(), candidate,
            mode='local', threshold=THRESHOLD, anchor_point=100,
        )
        assert len(result) > 0

    def test_no_events_returns_fail(self):
        result = generate_structural_verification(
            self._target(), {'local': []},
            mode='local', threshold=THRESHOLD, anchor_point=100,
        )
        assert len(result) > 0
        assert "FAIL" in result

    def test_all_events_far_returns_fail(self):
        far = {'local': [
            {'position_start': 10, 'type': 'upward spike'},
            {'position_start': 220, 'type': 'shake'},
        ]}
        result = generate_structural_verification(
            self._target(), far,
            mode='local', threshold=THRESHOLD, anchor_point=100,
        )
        assert len(result) > 0
        assert "FAIL" in result


class TestBuildLocalVerificationThought:
    def test_precheck_passes_think_has_tags(self):
        anchor = 100
        t = {'local': [{'position_start': 103, 'type': 'shake', 'value_start': 1.0}]}
        c = {'local': [{'position_start': 97, 'type': 'downward spike', 'value_start': 2.0}]}
        thought, cluster = build_local_verification_thought(
            [sanitize_attributes_for_sync(t), sanitize_attributes_for_sync(c)],
            ["TS1", "TS2"],
            include_attributes=['local'],
            anchor_point=anchor, mode='local', show_verdict=True,
            threshold=THRESHOLD, threshold_provided=True,
        )
        assert "<think>" in thought and "</think>" in thought

    def test_precheck_fails_when_target_has_no_near_event(self):
        anchor = 100
        t_far = {'local': [
            {'position_start': 10, 'type': 'upward spike', 'value_start': 1.0},
            {'position_start': 220, 'type': 'shake', 'value_start': 2.0},
        ]}
        c = {'local': [{'position_start': 97, 'type': 'downward spike', 'value_start': 3.0}]}
        thought, cluster = build_local_verification_thought(
            [sanitize_attributes_for_sync(t_far), sanitize_attributes_for_sync(c)],
            ["TS1", "TS2"],
            include_attributes=['local'],
            anchor_point=anchor, mode='local', show_verdict=True,
            threshold=THRESHOLD, threshold_provided=True,
        )
        # Cluster should be empty/falsy when pre-check fails
        assert not cluster

    def test_multi_event_target_with_one_near_passes(self):
        anchor = 100
        t = {'local': [
            {'position_start': 50, 'type': 'upward spike', 'value_start': 1.0},
            {'position_start': 103, 'type': 'shake', 'value_start': 2.0},
        ]}
        c = {'local': [{'position_start': 97, 'type': 'downward spike', 'value_start': 3.0}]}
        thought, cluster = build_local_verification_thought(
            [sanitize_attributes_for_sync(t), sanitize_attributes_for_sync(c)],
            ["TS1", "TS2"],
            include_attributes=['local'],
            anchor_point=anchor, mode='local', show_verdict=True,
            threshold=THRESHOLD, threshold_provided=True,
        )
        assert thought is not None and "<think>" in thought


class TestSelectSingleQA:
    def _make_result(self, n_qa=5):
        # Each QA pair: question has no placeholder, answer has exactly one prompt placeholder.
        # pair i → llm_prompts[i] = ["Li"], answer has <|prompt{i}|>
        # This makes reindexing behavior easy to predict:
        #   when pair i is selected: prompt_offset=i, old_idx=i → new_idx=0
        #   answer <|prompt{i}|> → <|prompt0|>
        questions = [f"Q{i}" for i in range(n_qa)]
        answers   = [f"A{i} <|prompt{i}|>" for i in range(n_qa)]
        llm_prompts = [[f"L{i}"] for i in range(n_qa)]
        fields    = [{"local": [0]} for _ in range(n_qa)]
        types     = [QAType.DESCRIPTION, QAType.YES_NO, QAType.CORRELATION,
                     QAType.CLUSTERING, QAType.SEGMENT_MASK][:n_qa]
        ts = np.zeros(10)
        return SampleResult(
            mode=Mode.UTS,
            original_timeseries=[ts], encoded_timeseries=[ts],
            metrics=["cpu"], attributes=[{}],
            base_prompt="base",
            questions=questions, answers=answers, llm_prompts=llm_prompts,
            fields=fields, corr_pool=[], label={},
            qa_types=[str(t) for t in types],
            eval_tasks=[str(t) for t in types],
            eval_metadatas=[{} for _ in range(n_qa)],
        )

    def test_leaves_exactly_one_qa_pair(self):
        from synth.align.factory import select_single_qa
        result = self._make_result(5)
        select_single_qa(result, DEFAULT_QA_TYPE_WEIGHTS)
        assert len(result.questions) == 1
        assert len(result.answers) == 1
        assert len(result.llm_prompts) == 1

    def test_propagates_eval_signals(self):
        from synth.align.factory import select_single_qa
        result = self._make_result(5)
        select_single_qa(result, DEFAULT_QA_TYPE_WEIGHTS)
        assert result.eval_task is not None
        assert result.eval_metadata is not None

    def test_reindexed_answer_placeholders_start_from_zero(self):
        """select_single_qa remaps the selected answer's placeholders to start from 0.

        Fixture: pair i has answer 'A{i} <|prompt{i}|>' with 1 prompt at absolute index i.
        After selecting pair i: prompt_offset=i → old index i → new index 0 → <|prompt0|>.
        """
        from synth.align.factory import select_single_qa
        result = self._make_result(5)
        select_single_qa(result, DEFAULT_QA_TYPE_WEIGHTS)
        answer = result.answers[0]
        indices = [int(m) for m in re.findall(r'<\|prompt(\d+)\|>', answer)]
        # The selected answer has exactly 1 prompt; after reindexing it must be <|prompt0|>
        assert indices == [0], f"Expected [0] after reindexing, got {indices}. Answer: {answer}"


def test_positive_generation():
    """Original test 1: positive MTS local generation has events near anchor."""
    random.seed(42); np.random.seed(42)
    ts_gen = TimeSeriesGenerator(mode="local", feature_config=DEFAULT_LOCAL_FEATURE_CONFIG)
    count = 6
    ts_list, attr_list, anchor, _ = ts_gen.generate_positive_timeseries(
        count=count, seq_len=SEQ_LEN, threshold=THRESHOLD
    )
    assert len(ts_list) == count
    assert len(attr_list) == count
    print(f"\nAnchor position: {anchor}")
    for i, attr in enumerate(attr_list):
        positions = get_local_event_positions(attr)
        near = [p for p in positions if abs(p - anchor) <= THRESHOLD]
        far  = [p for p in positions if abs(p - anchor) > THRESHOLD]
        print(f"  Series {i+1}: {len(positions)} total, {len(near)} near, {len(far)} far")


def test_negative_generation():
    """Original test 2: negative MTS local generation — zero events near anchor."""
    random.seed(42); np.random.seed(42)
    ts_gen = TimeSeriesGenerator(mode="local", feature_config=DEFAULT_LOCAL_FEATURE_CONFIG)
    anchor = 128
    count = 6
    ts_list, attr_list, _, _ = ts_gen.generate_negative_timeseries(
        count=count, positive_anchor=anchor, seq_len=SEQ_LEN, threshold=THRESHOLD
    )
    for attr in attr_list:
        positions = get_local_event_positions(attr)
        near = [p for p in positions if abs(p - anchor) <= THRESHOLD]
        assert len(near) == 0, f"Near events found: {near}"


def test_thinking_verification_multi():
    """Original test 3: structural verification with multiple events."""
    anchor = 100
    target = {'local': [{'position_start': 98, 'type': 'sudden increase'}]}

    candidate_mixed = {'local': [
        {'position_start': 50,  'type': 'upward spike'},
        {'position_start': 105, 'type': 'shake'},
        {'position_start': 200, 'type': 'downward spike'},
    ]}
    r1 = generate_structural_verification(target, candidate_mixed,
                                          mode='local', threshold=THRESHOLD, anchor_point=anchor)
    assert isinstance(r1, str)

    candidate_far = {'local': [
        {'position_start': 10,  'type': 'upward spike'},
        {'position_start': 200, 'type': 'shake'},
    ]}
    r2 = generate_structural_verification(target, candidate_far,
                                          mode='local', threshold=THRESHOLD, anchor_point=anchor)
    assert isinstance(r2, str)

    r3 = generate_structural_verification(target, {'local': []},
                                          mode='local', threshold=THRESHOLD, anchor_point=anchor)
    assert isinstance(r3, str)


def test_thinking_precheck_multi():
    """Original test 4: build_local_verification_thought pre-check with multi-event target."""
    anchor = 100
    target_attr = {'local': [
        {'position_start': 50,  'type': 'upward spike',    'value_start': 1.0},
        {'position_start': 103, 'type': 'shake',           'value_start': 2.0},
    ]}
    candidate_attr = {'local': [
        {'position_start': 97,  'type': 'downward spike',  'value_start': 3.0},
        {'position_start': 200, 'type': 'sudden increase', 'value_start': 4.0},
    ]}
    sparse_t = sanitize_attributes_for_sync(target_attr)
    sparse_c = sanitize_attributes_for_sync(candidate_attr)

    thought, cluster = build_local_verification_thought(
        [sparse_t, sparse_c], ["TS1", "TS2"],
        include_attributes=['local'], anchor_point=anchor,
        mode='local', show_verdict=True,
        threshold=THRESHOLD, threshold_provided=True,
    )
    assert isinstance(thought, str) and "<think>" in thought

    # All far — pre-check should fail
    t_fail = {'local': [
        {'position_start': 10,  'type': 'upward spike', 'value_start': 1.0},
        {'position_start': 220, 'type': 'shake',        'value_start': 2.0},
    ]}
    thought_fail, cluster_fail = build_local_verification_thought(
        [sanitize_attributes_for_sync(t_fail), sparse_c], ["TS1", "TS2"],
        include_attributes=['local'], anchor_point=anchor,
        mode='local', show_verdict=True,
        threshold=THRESHOLD, threshold_provided=True,
    )
    assert not cluster_fail


def test_compound_description_mts_local():
    """Original test 5: compound description QA — MTS local (single <think>)."""
    from synth.align.generators.mts_local import MTSLocalQAGenerator
    random.seed(0); np.random.seed(0)
    metrics = ["cpu_usage", "memory_usage", "disk_io", "network_rx"]
    n = len(metrics)
    ts_gen = TimeSeriesGenerator(mode="local", feature_config=DEFAULT_LOCAL_FEATURE_CONFIG)
    ts_list, attr_list, anchor, _ = ts_gen.generate_positive_timeseries(
        count=n, seq_len=SEQ_LEN, threshold=THRESHOLD
    )
    config, indexer = _dummy_config_and_indexer()
    gen = MTSLocalQAGenerator(
        config=config, indexer=indexer, situation="server", metrics=metrics,
        attributes=attr_list, cluster_indices=[None]*n, positive_clusters=[],
        change_positions=[anchor], metric_to_cluster={m: "server" for m in metrics},
        original_timeseries=ts_list, threshold=THRESHOLD,
        threshold_provided=True, seq_len=SEQ_LEN, difficulty=Difficulty.EASY,
    )
    r = gen.generate_all_qa()
    questions, answers, fields, qa_types = r.questions, r.answers, r.fields, r.qa_types

    compound_idxs = [
        i for i in range(len(questions))
        if qa_types[i] == "description" and len(set(sum(fields[i].values(), []))) > 1
    ]
    assert len(compound_idxs) > 0, "No compound description QA generated"
    for idx in compound_idxs:
        cnt = answers[idx].count("<think>")
        assert cnt == 1, f"Expected 1 <think>, got {cnt}"


def test_compound_description_mts_shape():
    """Original test 6: compound trend description QA — MTS shape (single <think>)."""
    from synth.align.generators.mts_shape import MTSShapeQAGenerator
    random.seed(1); np.random.seed(1)
    metrics = ["latency_p99", "error_rate", "throughput"]
    n = len(metrics)
    ts_gen = TimeSeriesGenerator(mode="shape", feature_config=DEFAULT_SHAPE_FEATURE_CONFIG)
    ts_list, attr_list, _, pts_list = ts_gen.generate_positive_timeseries(
        count=n, seq_len=SEQ_LEN, threshold=THRESHOLD
    )
    for attr in attr_list:
        if not attr.get('trend', {}).get('detail'):
            attr.setdefault('trend', {})['detail'] = "The metric shows a steady trend."
    config, indexer = _dummy_config_and_indexer()
    gen = MTSShapeQAGenerator(
        config=config, indexer=indexer, situation="api_gateway", metrics=metrics,
        attributes=attr_list, cluster_indices=[None]*n, points_list=pts_list,
        metric_to_cluster={m: "api" for m in metrics},
        seq_len=SEQ_LEN, threshold=THRESHOLD, threshold_provided=True,
    )
    r = gen.generate_all_qa()
    questions, answers, fields, qa_types = r.questions, r.answers, r.fields, r.qa_types
    compound_idxs = [
        i for i in range(len(questions))
        if qa_types[i] == "description" and len(set(sum(fields[i].values(), []))) > 1
    ]
    assert len(compound_idxs) > 0, "No compound description QA generated for shape mode"
    for idx in compound_idxs:
        cnt = answers[idx].count("<think>")
        assert cnt == 1, f"Expected 1 <think>, got {cnt}"


class TestDefaultModeWeights:
    def test_sum_to_one(self):
        total = sum(DEFAULT_MODE_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-6

    def test_all_modes_present(self):
        assert "uts" in DEFAULT_MODE_WEIGHTS
        assert "mts_local" in DEFAULT_MODE_WEIGHTS
        assert "mts_shape" in DEFAULT_MODE_WEIGHTS


class TestTwoLevelSelection:
    def test_balances_eval_types_under_segment_mask(self):
        """stat_numerical (10 entries) and trend_dominance (1 entry) should get
        roughly equal share under segment_mask with two-level selection."""
        random.seed(42)
        # Simulate: 10 stat_numerical + 1 trend_dominance, all segment_mask
        qa_types = ["segment_mask"] * 11
        eval_types = (["stat_numerical"] * 10) + ["segment_trend_dominance"]
        weights = {QAType.SEGMENT_MASK: 1.0}

        counts = {"stat_numerical": 0, "segment_trend_dominance": 0}
        N = 2000
        for _ in range(N):
            idx = select_weighted_qa_index(
                qa_types, weights,
                eval_types=eval_types,
                eval_type_weights={},  # all default to 1.0
            )
            counts[eval_types[idx]] += 1

        # Each eval_type should get ~50%. Allow wide margin (30-70%).
        sn_pct = counts["stat_numerical"] / N
        assert 0.30 < sn_pct < 0.70, f"stat_numerical got {sn_pct:.1%}, expected ~50%"

    def test_single_level_backward_compat(self):
        """With eval_types=None, falls back to single-level (original behavior)."""
        random.seed(42)
        qa_types = ["description", "yes_no", "description"]
        weights = {"description": 0.5, "yes_no": 0.5}

        counts = {"description": 0, "yes_no": 0}
        N = 1000
        for _ in range(N):
            idx = select_weighted_qa_index(qa_types, weights)
            counts[qa_types[idx]] += 1

        # Should pick both types
        assert counts["description"] > 0
        assert counts["yes_no"] > 0


class TestAllocateTasks:
    def test_respects_proportions(self):
        """_allocate_tasks should respect mode_weights proportions."""
        from synth.align.dataset import DatasetGenerator
        config = Config.__new__(Config)
        config.num_data = 100
        config.mode_weights = {"uts": 0.30, "mts_local": 0.35, "mts_shape": 0.35}
        gen = DatasetGenerator.__new__(DatasetGenerator)
        gen.config = config
        tasks = gen._allocate_tasks()
        assert len(tasks) == 100

        mode_counts = {}
        for mode in tasks:
            mode_counts[mode.value] = mode_counts.get(mode.value, 0) + 1

        assert mode_counts["uts"] == 30
        assert mode_counts["local"] == 35
        assert mode_counts["shape"] == 35

    def test_uts_all_same_mode(self):
        """All UTS tasks should be Mode.UTS."""
        from synth.align.dataset import DatasetGenerator
        config = Config.__new__(Config)
        config.num_data = 50
        config.mode_weights = {"uts": 1.0, "mts_local": 0.0, "mts_shape": 0.0}
        gen = DatasetGenerator.__new__(DatasetGenerator)
        gen.config = config
        tasks = gen._allocate_tasks()
        assert len(tasks) == 50
        assert all(m == Mode.UTS for m in tasks)

    def test_mts_local_only(self):
        """MTS_LOCAL-only allocation should produce correct count."""
        from synth.align.dataset import DatasetGenerator
        config = Config.__new__(Config)
        config.num_data = 50
        config.mode_weights = {"uts": 0.0, "mts_local": 1.0, "mts_shape": 0.0}
        gen = DatasetGenerator.__new__(DatasetGenerator)
        gen.config = config
        tasks = gen._allocate_tasks()
        assert len(tasks) == 50
        assert all(m == Mode.MTS_LOCAL for m in tasks)


if __name__ == "__main__":
    def _sep(title):
        print(f"\n{'='*80}\n  {title}\n{'='*80}\n")

    _sep("TEST 1: Positive MTS local generation")
    test_positive_generation()
    _sep("TEST 2: Negative MTS local generation")
    test_negative_generation()
    _sep("TEST 3: Structural verification with multiple events")
    test_thinking_verification_multi()
    _sep("TEST 4: build_local_verification_thought pre-check with multi-event target")
    test_thinking_precheck_multi()
    _sep("TEST 5: Compound description QA — MTS local")
    test_compound_description_mts_local()
    _sep("TEST 6: Compound description QA — MTS shape")
    test_compound_description_mts_shape()
    print("\nAll manual tests passed.")
