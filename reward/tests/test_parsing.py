"""
Tests for reward.parsing — edge cases in extracting structured data from think blocks.
"""
import pytest
from reward.parsing import (
    extract_think_content,
    extract_answer_scalar,
    extract_answer_set,
    extract_yes_no_result,
    extract_verification_per_metric,
    extract_metric_blocks,
    _extract_local_events,
    _extract_trend_segments,
    _extract_key_points,
    _extract_params,
)


# ── extract_think_content ──────────────────────────────────────────────

class TestExtractThinkContent:
    def test_basic(self):
        assert extract_think_content("<think>hello</think>") == "hello"

    def test_multiline(self):
        text = "<think>\nline1\nline2\n</think>"
        result = extract_think_content(text)
        assert "line1" in result and "line2" in result

    def test_no_think_block(self):
        assert extract_think_content("no think block here") is None

    def test_empty_think_block(self):
        assert extract_think_content("<think></think>") == ""

    def test_nested_angle_brackets(self):
        # Should not break on angle brackets inside the block
        result = extract_think_content("<think>value > 5 and value < 10</think>")
        assert "value > 5" in result

    def test_surrounding_text(self):
        text = "preamble <think>content</think> postamble"
        assert extract_think_content(text) == "content"

    def test_only_first_think_block(self):
        text = "<think>first</think> <think>second</think>"
        assert extract_think_content(text) == "first"


# ── extract_answer_scalar ──────────────────────────────────────────────

class TestExtractAnswerScalar:
    def test_basic(self):
        assert extract_answer_scalar("answer: yes") == "yes"

    def test_case_insensitive(self):
        assert extract_answer_scalar("Answer: NO") == "no"

    def test_numeric(self):
        assert extract_answer_scalar("answer: 3.14") == "3.14"

    def test_missing(self):
        assert extract_answer_scalar("no answer line here") is None

    def test_multiline(self):
        text = "metric: X\nstats: mean=5\nanswer: increase"
        assert extract_answer_scalar(text) == "increase"

    def test_answer_with_extra_text(self):
        # Should only capture first non-whitespace token
        result = extract_answer_scalar("answer: yes (confirmed)")
        assert result == "yes"


# ── extract_answer_set ─────────────────────────────────────────────────

class TestExtractAnswerSet:
    def test_basic(self):
        result = extract_answer_set("answer: (A, B, C)")
        assert result == {"a", "b", "c"}

    def test_single(self):
        result = extract_answer_set("answer: (MetricX)")
        assert result == {"metricx"}

    def test_empty_parens(self):
        result = extract_answer_set("answer: ()")
        assert result == set()

    def test_missing(self):
        assert extract_answer_set("no answer here") is None

    def test_spaces_in_names(self):
        result = extract_answer_set("answer: ( Foo ,  Bar )")
        assert result == {"foo", "bar"}


# ── extract_yes_no_result ──────────────────────────────────────────────

class TestExtractYesNoResult:
    def test_pass(self):
        text = "data\n===\ntarget metric: PASS."
        assert extract_yes_no_result(text) is True

    def test_fail(self):
        text = "data\n===\ntarget metric: FAIL."
        assert extract_yes_no_result(text) is False

    def test_no_verification_section(self):
        assert extract_yes_no_result("just some text") is None

    def test_skips_vs_lines(self):
        text = "data\n===\nvs Other: PASS.\ntarget: FAIL."
        assert extract_yes_no_result(text) is False

    def test_skips_threshold_lines(self):
        text = "data\n===\nthreshold=5: PASS.\ntarget: FAIL."
        assert extract_yes_no_result(text) is False

    def test_pass_and_fail_in_same_line(self):
        # "FAIL." takes precedence when both present
        text = "data\n===\nPASS. but also FAIL."
        assert extract_yes_no_result(text) is False


# ── extract_verification_per_metric ────────────────────────────────────

class TestExtractVerificationPerMetric:
    def test_basic(self):
        text = "data\n===\nvs Alpha: PASS.\nvs Beta: FAIL."
        result = extract_verification_per_metric(text)
        assert result == {"Alpha": True, "Beta": False}

    def test_no_vs_lines(self):
        text = "data\n===\njust a verdict: PASS."
        assert extract_verification_per_metric(text) == {}

    def test_no_verification_section(self):
        assert extract_verification_per_metric("no === here") == {}


# ── extract_metric_blocks ─────────────────────────────────────────────

class TestExtractMetricBlocks:
    def test_single_metric(self):
        think = (
            "metric: TestMetric\n"
            "noise=smooth, strength=0.05\n"
            "start=10.0, end=20.0, amp=10.0, overall trend=increase\n"
            "trend_segments=[(increase, 1, 100)]\n"
            "season=no periodic fluctuation, period=0.00, amp=0.00"
        )
        blocks = extract_metric_blocks(think)
        assert "TestMetric" in blocks
        attrs = blocks["TestMetric"]
        assert attrs["noise"]["type"] == "smooth"
        assert attrs["noise"]["strength"] == 0.05
        assert attrs["trend"]["type"] == "increase"
        assert attrs["trend"]["start"] == 10.0
        assert attrs["trend"]["end"] == 20.0
        assert len(attrs["trend_segments"]) == 1

    def test_multi_metric(self):
        think = (
            "metric: Alpha\n"
            "noise=smooth, strength=0.00\n"
            "---\n"
            "metric: Beta\n"
            "noise=noisy, strength=0.50"
        )
        blocks = extract_metric_blocks(think)
        assert len(blocks) == 2
        assert blocks["Alpha"]["noise"]["type"] == "smooth"
        assert blocks["Beta"]["noise"]["type"] == "noisy"

    def test_stats_line(self):
        think = "metric: M\nstats: mean=5.000, std=1.200, min=2.000, max=8.000"
        blocks = extract_metric_blocks(think)
        stats = blocks["M"]["stats"]
        assert stats["mean"] == 5.0
        assert stats["std"] == 1.2
        assert stats["min"] == 2.0
        assert stats["max"] == 8.0

    def test_malformed_float_in_stats(self):
        """The '71.251.' bug — trailing dot should not crash, just skip the bad value."""
        think = "metric: M\nstats: mean=71.251., std=1.200"
        blocks = extract_metric_blocks(think)
        stats = blocks["M"]["stats"]
        assert "mean" not in stats  # malformed value skipped
        assert stats["std"] == 1.2  # valid value kept

    def test_stats_with_negative_exponent(self):
        think = "metric: M\nstats: mean=1.5e-3, std=2.0E+1"
        blocks = extract_metric_blocks(think)
        stats = blocks["M"]["stats"]
        assert abs(stats["mean"] - 0.0015) < 1e-8
        assert stats["std"] == 20.0

    def test_len_line(self):
        think = "metric: M\nlen=256"
        blocks = extract_metric_blocks(think)
        assert blocks["M"]["length"] == 256

    def test_no_metric_name(self):
        think = "noise=smooth, strength=0.00"
        blocks = extract_metric_blocks(think)
        assert len(blocks) == 0  # No metric: line → not captured

    def test_local_events(self):
        think = (
            "metric: M\n"
            "local=[upward spike, amp=5.0, metadata=[start@(10, 1.0), end@(20, 6.0)]]"
        )
        blocks = extract_metric_blocks(think)
        local = blocks["M"]["local"]
        assert len(local) == 1
        assert local[0]["type"] == "upward spike"
        assert local[0]["amplitude"] == 5.0

    def test_empty_local(self):
        think = "metric: M\nlocal=[]"
        blocks = extract_metric_blocks(think)
        assert blocks["M"]["local"] == []

    def test_seasonal_no_fluctuation(self):
        think = "metric: M\nseason=no periodic fluctuation, period=0.00, amp=0.00"
        blocks = extract_metric_blocks(think)
        s = blocks["M"]["seasonal"]
        assert s["type"] == "no periodic fluctuation"
        assert s["period"] == 0.0


# ── _extract_trend_segments ────────────────────────────────────────────

class TestExtractTrendSegments:
    def test_basic(self):
        line = "trend_segments=[(increase, 1, 50), (decrease, 51, 100)]"
        segs = _extract_trend_segments(line)
        assert len(segs) == 2
        assert segs[0] == {"type": "increase", "start": 1, "end": 50}
        assert segs[1] == {"type": "decrease", "start": 51, "end": 100}

    def test_single_segment(self):
        line = "trend_segments=[(keep steady, 1, 256)]"
        segs = _extract_trend_segments(line)
        assert len(segs) == 1
        assert segs[0]["type"] == "keep steady"

    def test_empty(self):
        line = "trend_segments=[]"
        segs = _extract_trend_segments(line)
        assert segs == []


# ── _extract_key_points ────────────────────────────────────────────────

class TestExtractKeyPoints:
    def test_new_format(self):
        ev = "upward spike, amp=5.0, metadata=[start@(10, 1.0), end@(20, 6.0)]"
        kps = _extract_key_points(ev)
        assert len(kps) == 2
        assert kps[0]["label"] == "start"
        assert kps[0]["index"] == 10
        assert kps[0]["value"] == 1.0

    def test_old_format(self):
        ev = "spike, points=[peak@50]"
        kps = _extract_key_points(ev)
        assert len(kps) == 1
        assert kps[0]["label"] == "peak"
        assert kps[0]["index"] == 50

    def test_no_key_points(self):
        ev = "just a type description"
        kps = _extract_key_points(ev)
        assert kps == []


# ── _extract_params ────────────────────────────────────────────────────

class TestExtractParams:
    def test_new_format_in_metadata(self):
        ev = "type, metadata=[width=10.0, sub_amplitudes=[0.5, 0.3]]"
        params = _extract_params(ev)
        assert params["width"] == 10.0
        assert params["sub_amplitudes"] == [0.5, 0.3]

    def test_old_format_in_params(self):
        ev = "type, params={width=10.0}"
        params = _extract_params(ev)
        assert params["width"] == 10.0

    def test_no_params(self):
        ev = "just text"
        assert _extract_params(ev) == {}


# ── Malformed / adversarial inputs ─────────────────────────────────────

class TestMalformedInputs:
    def test_stats_trailing_dot(self):
        """Regression test for '71.251.' crash — gracefully skips bad value."""
        think = "metric: M\nstats: mean=71.251."
        blocks = extract_metric_blocks(think)
        stats = blocks["M"]["stats"]
        assert isinstance(stats, dict)
        assert "mean" not in stats  # malformed value skipped

    def test_stats_double_dot(self):
        """stats: mean=1..5 should not crash — gracefully skips bad value."""
        think = "metric: M\nstats: mean=1..5"
        blocks = extract_metric_blocks(think)
        stats = blocks["M"]["stats"]
        assert isinstance(stats, dict)
        assert "mean" not in stats  # malformed value skipped

    def test_stats_nan(self):
        """NaN should not be accepted as a stat value."""
        # regex won't match 'nan' (no digits), so it won't appear in stats
        think = "metric: M\nstats: mean=nan"
        blocks = extract_metric_blocks(think)
        # 'nan' doesn't match [\d.eE+-]+ regex, so stats should be None
        assert blocks["M"]["stats"] is None

    def test_stats_inf(self):
        """inf should not be accepted."""
        think = "metric: M\nstats: mean=inf"
        blocks = extract_metric_blocks(think)
        assert blocks["M"]["stats"] is None

    def test_empty_think(self):
        blocks = extract_metric_blocks("")
        assert blocks == {}

    def test_only_separator(self):
        blocks = extract_metric_blocks("---")
        assert blocks == {}

    def test_metric_no_attributes(self):
        blocks = extract_metric_blocks("metric: Lonely")
        assert "Lonely" in blocks
        # All attrs should be None
        for v in blocks["Lonely"].values():
            assert v is None

    def test_trend_missing_fields(self):
        """Partial trend line should not populate trend."""
        think = "metric: M\nstart=10.0, overall trend=increase"
        blocks = extract_metric_blocks(think)
        # Missing end= and amp= → regex won't match fully
        assert blocks["M"]["trend"] is None

    def test_noise_missing_strength(self):
        think = "metric: M\nnoise=smooth"
        blocks = extract_metric_blocks(think)
        assert blocks["M"]["noise"] is None

    def test_season_missing_period(self):
        think = "metric: M\nseason=periodic"
        blocks = extract_metric_blocks(think)
        assert blocks["M"]["seasonal"] is None

    def test_unicode_in_metric_name(self):
        think = "metric: Métric_α\nnoise=smooth, strength=0.1"
        blocks = extract_metric_blocks(think)
        assert "Métric_α" in blocks

    def test_very_large_float(self):
        think = "metric: M\nstats: mean=9999999999.999"
        blocks = extract_metric_blocks(think)
        assert blocks["M"]["stats"]["mean"] == 9999999999.999

    def test_negative_values(self):
        think = "metric: M\nstart=-10.5, end=-5.0, amp=5.5, overall trend=increase"
        blocks = extract_metric_blocks(think)
        assert blocks["M"]["trend"]["start"] == -10.5
        assert blocks["M"]["trend"]["end"] == -5.0

    def test_local_multiple_events(self):
        think = (
            "metric: M\n"
            "local=[upward spike, amp=5.0, metadata=[start@(10, 1.0)]; "
            "downward spike, amp=3.0, metadata=[start@(50, 8.0)]]"
        )
        blocks = extract_metric_blocks(think)
        assert len(blocks["M"]["local"]) == 2

    def test_stats_regex_greediness(self):
        """Ensure regex doesn't match across lines."""
        think = "metric: M\nstats: mean=5.0\nstats: std=1.0"
        blocks = extract_metric_blocks(think)
        assert blocks["M"]["stats"]["mean"] == 5.0
        assert blocks["M"]["stats"]["std"] == 1.0
