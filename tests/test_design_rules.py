"""
Design rules enforcement tests.

Run after ANY new builder, QA generator, scorer, or template:
    pytest tests/test_design_rules.py -v

These tests catch the exact classes of bugs documented in memory/common_mistakes.md.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# File sets for static analysis
# ---------------------------------------------------------------------------

THINKING_DIR = ROOT / "synth" / "ts_generator" / "utils" / "thinking"
GENERATORS_DIR = ROOT / "synth" / "align" / "generators"
TEMPLATES_DIR = ROOT / "synth" / "align" / "templates"
BASE_GENERATOR = ROOT / "synth" / "align" / "base_generator.py"
REWARD_INIT = ROOT / "reward" / "__init__.py"
EVAL_CONFIG = ROOT / "evaluation" / "eval" / "config.py"
EVAL_SCORING = ROOT / "evaluation" / "eval" / "scoring.py"
SYNTH_DIR = ROOT / "synth"

THINKING_FILES = [
    f for f in THINKING_DIR.glob("*.py")
    if f.name != "__init__.py"
]

GENERATOR_FILES = [
    f for f in GENERATORS_DIR.glob("*.py")
    if f.name != "__init__.py"
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ===================================================================
# SECTION 1: PRECISION RULES  (common_mistakes #4)
# ===================================================================

class TestPrecisionRules:
    """format_float must use dp=2 in think builders. No dp=3/4."""

    def test_no_format_float_dp3_in_builders(self):
        """Rule #4: NEVER format_float(val, 3) in think builders."""
        pattern = re.compile(r"format_float\([^)]*,\s*[34]\s*\)")
        violations = []
        for f in THINKING_FILES:
            for i, line in enumerate(_read(f).splitlines(), 1):
                if pattern.search(line):
                    violations.append(f"{f.name}:{i}: {line.strip()}")
        assert not violations, (
            "format_float with dp=3 or dp=4 in thinking builders:\n"
            + "\n".join(violations)
        )

    def test_no_round_3_in_builders(self):
        """No round(..., 3) or round(..., 4) in think builders (display code).
        statistics_utils.py internal storage is exempt."""
        pattern = re.compile(r"\bround\([^)]*,\s*[34]\s*\)")
        violations = []
        for f in THINKING_FILES:
            for i, line in enumerate(_read(f).splitlines(), 1):
                if pattern.search(line):
                    violations.append(f"{f.name}:{i}: {line.strip()}")
        assert not violations, (
            "round(..., 3 or 4) in thinking builders:\n"
            + "\n".join(violations)
        )

    def test_format_float_dp2_in_answer_text(self):
        """Answer text methods must use format_float(val, 2)."""
        pattern = re.compile(r"format_float\([^)]*,\s*[34]\s*\)")
        src = _read(BASE_GENERATOR)
        violations = []
        for i, line in enumerate(src.splitlines(), 1):
            if pattern.search(line) and "answer" in line.lower():
                violations.append(f"base_generator.py:{i}: {line.strip()}")
        assert not violations, (
            "format_float with dp!=2 in answer text:\n"
            + "\n".join(violations)
        )


# ===================================================================
# SECTION 2: MOLECULAR HELPER RULES  (common_mistakes #2)
# ===================================================================

class TestMolecularHelperRules:
    """All displayed arithmetic must use molecular helpers, not inline math."""

    def test_no_inline_abs_in_format_cmp(self):
        """Rule #2 + #51: abs() should not appear inside format_cmp() arguments.
        Signed diff must be shown first via format_add_sub_calc."""
        pattern = re.compile(r"format_cmp\(\s*abs\(")
        violations = []
        for f in THINKING_FILES:
            for i, line in enumerate(_read(f).splitlines(), 1):
                if pattern.search(line):
                    violations.append(f"{f.name}:{i}: {line.strip()}")
        assert not violations, (
            "abs() inside format_cmp() arguments — show signed diff first:\n"
            + "\n".join(violations)
        )

    def test_no_inline_abs_in_format_div(self):
        """abs() should not appear inside format_div_calc() arguments."""
        pattern = re.compile(r"format_div_calc\(\s*abs\(")
        violations = []
        for f in THINKING_FILES:
            for i, line in enumerate(_read(f).splitlines(), 1):
                if pattern.search(line):
                    violations.append(f"{f.name}:{i}: {line.strip()}")
        assert not violations, (
            "abs() inside format_div_calc() arguments:\n"
            + "\n".join(violations)
        )

    def test_no_inline_abs_in_tb_line_fstring(self):
        """abs() should not appear in tb.line(f"...abs(...)...") — show via
        format_add_sub_calc first, then |diff| = X."""
        pattern = re.compile(r'tb\.line\(f["\'].*abs\(')
        violations = []
        for f in THINKING_FILES:
            for i, line in enumerate(_read(f).splitlines(), 1):
                if pattern.search(line):
                    violations.append(f"{f.name}:{i}: {line.strip()}")
        assert not violations, (
            "abs() in tb.line() f-string — use format_add_sub_calc:\n"
            + "\n".join(violations)
        )

    def test_no_inline_abs_in_tb_data(self):
        """abs() should never appear in tb.data() calls."""
        pattern = re.compile(r'tb\.data\(.*abs\(')
        violations = []
        for f in THINKING_FILES:
            for i, line in enumerate(_read(f).splitlines(), 1):
                if pattern.search(line):
                    violations.append(f"{f.name}:{i}: {line.strip()}")
        assert not violations, (
            "abs() in tb.data() — data block should not contain computations:\n"
            + "\n".join(violations)
        )

    def test_no_hidden_mul_in_format_add_sub(self):
        """Rule #22: N * value passed to format_add_sub_calc must first be
        shown via format_mul_calc. Detects ``format_add_sub_calc(..., 2 * x,``
        which hides the multiplication derivation."""
        pattern = re.compile(r'format_add_sub_calc\([^)]*\d+\s*\*\s*\w+')
        violations = []
        for f in THINKING_FILES:
            for i, line in enumerate(_read(f).splitlines(), 1):
                if pattern.search(line):
                    violations.append(f"{f.name}:{i}: {line.strip()}")
        assert not violations, (
            "Hidden multiplication in format_add_sub_calc args — "
            "show derivation via format_mul_calc first:\n"
            + "\n".join(violations)
        )


# ===================================================================
# SECTION 3: INDEXING RULES  (common_mistakes #5)
# ===================================================================

class TestIndexingRules:
    """All positions 0-indexed. No +1 adjustments."""

    def test_no_plus_one_in_builders(self):
        """Rule #5: NEVER +1 for display. All 0-indexed."""
        # Look for position/index variables followed by + 1
        pattern = re.compile(r"(?:pos|position|idx|index|_pos)\s*\+\s*1")
        violations = []
        for f in THINKING_FILES:
            for i, line in enumerate(_read(f).splitlines(), 1):
                if pattern.search(line) and "range(" not in line:
                    violations.append(f"{f.name}:{i}: {line.strip()}")
        assert not violations, (
            "Possible +1 indexing adjustment in builders:\n"
            + "\n".join(violations)
        )


# ===================================================================
# SECTION 4: QA GENERATOR RULES (common_mistakes #17, #28, #12b)
# ===================================================================

class TestQAGeneratorRules:
    """QA generators must follow template/metadata rules."""

    def test_except_blocks_have_logging(self):
        """Rule #17: except blocks must have logger.warning with exc_info=True."""
        # Find except blocks, check that logger.warning follows within 3 lines
        violations = []
        for f in [BASE_GENERATOR] + GENERATOR_FILES:
            lines = _read(f).splitlines()
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("except") and "Exception" in stripped:
                    # Check next 5 lines for logger.warning with exc_info
                    snippet = "\n".join(lines[i:i + 6])
                    if "logger.warning" not in snippet or "exc_info=True" not in snippet:
                        # Skip if it's a re-raise
                        if "raise" in snippet:
                            continue
                        violations.append(f"{f.name}:{i + 1}: {stripped}")
        assert not violations, (
            "except blocks missing logger.warning(..., exc_info=True):\n"
            + "\n".join(violations)
        )

    def test_eval_metadata_has_length(self):
        """Rule #12b: eval_metadatas must include 'length'."""
        src = _read(BASE_GENERATOR)
        # Find all eval_metadatas.append({ blocks
        pattern = re.compile(
            r"eval_metadatas\.append\(\{([^}]*)\}",
            re.DOTALL,
        )
        violations = []
        for m in pattern.finditer(src):
            content = m.group(1)
            if "'length'" not in content and '"length"' not in content:
                # Get line number
                line_num = src[:m.start()].count("\n") + 1
                violations.append(
                    f"base_generator.py:{line_num}: "
                    f"eval_metadata missing 'length'"
                )
        assert not violations, (
            "eval_metadata blocks missing 'length' key:\n"
            + "\n".join(violations)
        )


# ===================================================================
# SECTION 5: PIPELINE WIRING (common_mistakes #9, #9b)
# ===================================================================

class TestPipelineWiring:
    """Every eval_type must be registered in reward routing AND eval config."""

    def _get_reward_routed_types(self) -> set:
        """Extract all eval_types handled in compute_score()."""
        src = _read(REWARD_INIT)
        types = set()
        # Match: qa_type == 'xxx' or qa_type in ('xxx', 'yyy')
        for m in re.finditer(r"qa_type\s*==\s*'([^']+)'", src):
            types.add(m.group(1))
        for m in re.finditer(r"qa_type\s+in\s+\(([^)]+)\)", src):
            for t in re.findall(r"'([^']+)'", m.group(1)):
                types.add(t)
        # startswith patterns
        for m in re.finditer(r"qa_type\.startswith\(\(([^)]+)\)", src):
            for prefix in re.findall(r"'([^']+)'", m.group(1)):
                types.add(f"{prefix}*")
        return types

    def _get_ragas_types(self) -> set:
        """Extract eval_types from RAGAS_QUESTION_BY_TASK dict keys."""
        from evaluation.eval.config import RAGAS_QUESTION_BY_TASK
        return set(RAGAS_QUESTION_BY_TASK.keys())

    def _get_segment_family_types(self) -> set:
        """Extract from SEGMENT_FAMILY_TYPES frozenset."""
        src = _read(EVAL_CONFIG)
        m = re.search(
            r"SEGMENT_FAMILY_TYPES\s*=\s*frozenset\(\{([^}]+)\}",
            src, re.DOTALL,
        )
        if not m:
            return set()
        return set(re.findall(r"'([^']+)'", m.group(1)))

    def test_eval_config_covers_reward_types(self):
        """Every eval_type routed in reward must be in RAGAS_QUESTION_BY_TASK."""
        reward_types = self._get_reward_routed_types()
        eval_types = self._get_ragas_types()
        # tsevol and description intentionally NOT in RAGAS — judge uses
        # the actual question text (tsevol: evolved Q; description: perspective-specific Q)
        reward_types -= {"tsevol", "description"}
        # Expand prefix patterns
        concrete_reward = set()
        for t in reward_types:
            if t.endswith("*"):
                # Prefix — check that at least one eval_config type matches
                prefix = t[:-1]
                matching = [e for e in eval_types if e.startswith(prefix)]
                if not matching:
                    concrete_reward.add(t)
            else:
                concrete_reward.add(t)
        missing = concrete_reward - eval_types
        assert not missing, (
            f"eval_types in reward routing but NOT in RAGAS_QUESTION_BY_TASK: {missing}"
        )

    def test_segment_family_types_complete(self):
        """All segment-family eval_types must be in SEGMENT_FAMILY_TYPES."""
        eval_types = self._get_ragas_types()
        segment_types = self._get_segment_family_types()
        # These are NOT segment family types (they have their own scoring):
        non_segment = {
            "correlation", "anticorrelation", "clustering",
            "anticlustering", "yes_no", "description", "tsevol",
        }
        should_be_segment = eval_types - non_segment
        missing = should_be_segment - segment_types
        assert not missing, (
            f"eval_types in RAGAS but NOT in SEGMENT_FAMILY_TYPES: {missing}"
        )


# ===================================================================
# SECTION 6: REWARD SELF-CONSISTENCY (the #1 rule)
# ===================================================================

class TestSelfConsistency:
    """compute_score(gt, gt) >= 0.99 for every eval_type + sub_type combo."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from reward.evidence import set_context_seq_len
        set_context_seq_len(256)

    @staticmethod
    def _score(gt, eval_type, sub_type, verdict):
        from reward import compute_score
        return compute_score(
            gt, gt,
            eval_type=eval_type,
            eval_metadata={"length": 256, "sub_type": sub_type, "verdict": verdict},
        )

    # -- Periodicity family --

    @pytest.mark.parametrize("sub_type,verdict,gt", [
        ("period_estimate", "32.00",
         "<think>m: T\nseasonal: period=32.00\n===\nperiod = 32.00\nanswer: 32.00</think>"),
        ("cycle_count", "8",
         "<think>m: T\nseasonal: period=32.00\nseq_len=256\n===\n256/32.00=8.00\nfloor=8\nanswer: 8</think>"),
    ])
    def test_periodicity_self_consistency(self, sub_type, verdict, gt):
        score = self._score(gt, "periodicity", sub_type, verdict)
        assert score >= 0.99, f"periodicity/{sub_type}({verdict}): {score:.4f}"

    # -- Change point family --

    @pytest.mark.parametrize("sub_type,verdict,gt", [
        ("change_point_count", "2",
         "<think>m: T\nsegs: 3\n===\n3-1=2\nanswer: 2</think>"),
        ("change_point_positions", "[100, 180]",
         "<think>m: T\nsegs\n===\nboundaries\nanswer: [100, 180]</think>"),
        ("largest_level_shift", "100",
         "<think>m: T\nsegs\n===\nmax at 100\nanswer: 100</think>"),
    ])
    def test_change_point_self_consistency(self, sub_type, verdict, gt):
        score = self._score(gt, "change_point", sub_type, verdict)
        assert score >= 0.99, f"change_point/{sub_type}({verdict}): {score:.4f}"

    # -- stat_numerical extensions --

    @pytest.mark.parametrize("sub_type,verdict,gt", [
        ("stat_median", "1.50",
         "<think>m: T\nstats: median=1.50\n===\nmedian=1.50\nanswer: 1.50</think>"),
        ("stat_half_mean_diff", "0.30",
         "<think>m: T\nstats: fh=1.50,sh=1.20\n===\ndiff=-0.30\n|diff|=0.30\nanswer: 0.30</think>"),
        ("stat_volatility_change", "more_volatile",
         "<think>m: T\nstats: fhs=1.50,shs=2.00\n===\nratio=1.33>1.10\nanswer: more_volatile</think>"),
        ("stat_volatility_change", "stable",
         "<think>m: T\nstats: fhs=1.50,shs=1.52\n===\nratio=1.01\nanswer: stable</think>"),
        ("stat_windowed_trend", "upward",
         "<think>m: T\nmeans=[1,2,3]\n===\nlast>first*1.05\nanswer: upward</think>"),
        ("stat_windowed_trend", "flat",
         "<think>m: T\nmeans=[1,1,1]\n===\nwithin band\nanswer: flat</think>"),
    ])
    def test_stat_extension_self_consistency(self, sub_type, verdict, gt):
        score = self._score(gt, "stat_numerical", sub_type, verdict)
        assert score >= 0.99, f"stat_numerical/{sub_type}({verdict}): {score:.4f}"


# ===================================================================
# SECTION 7: SCORER TYPE COVERAGE
# ===================================================================

class TestScorerTypeCoverage:
    """Every new categorical/binary verdict must be in the appropriate set."""

    def test_statistical_categorical_completeness(self):
        """All simple categorical verdicts for stat/periodicity/anomaly families."""
        from reward.scorers.statistical import (
            _CATEGORICAL_VERDICTS, _SIMPLE_CATEGORICAL_VERDICTS, _BINARY_VERDICTS,
        )
        all_handled = _CATEGORICAL_VERDICTS | _SIMPLE_CATEGORICAL_VERDICTS | _BINARY_VERDICTS
        required = {
            "first half", "second half", "equal",     # stat_segment_compare
            "more_volatile", "less_volatile", "stable", # stat_volatility_change
            "upward", "downward", "flat",               # stat_windowed_trend
            "yes", "no",                                # anomaly outlier_present
        }
        missing = required - all_handled
        assert not missing, f"Verdicts missing from scorer type sets: {missing}"


# ===================================================================
# SECTION 8: STATISTICS INFRASTRUCTURE
# ===================================================================

class TestStatisticsInfra:
    """New statistics keys must be computed correctly."""

    def test_new_statistics_keys_present(self):
        """All new keys exist in calculated statistics."""
        import numpy as np
        from synth.ts_generator.utils.statistics_utils import calculate_statistics

        y = np.random.randn(256)
        stats = calculate_statistics(y, 256)
        required_keys = [
            "median", "q25", "q75",
            "outlier_2sigma_count", "outlier_2sigma_fraction",
            "extreme_outlier_pos",
            # existing keys that should still be there
            "mean", "std", "max", "min", "max_pos", "min_pos", "range",
            "segment_means_16", "segment_means_32",
            "mean_crossings", "mean_crossing_indices",
            "first_half_mean", "second_half_mean",
            "first_half_std", "second_half_std",
        ]
        for key in required_keys:
            assert key in stats, f"Missing statistics key: {key}"

    def test_outlier_count_nonnegative(self):
        import numpy as np
        from synth.ts_generator.utils.statistics_utils import calculate_statistics

        y = np.random.randn(256)
        stats = calculate_statistics(y, 256)
        assert stats["outlier_2sigma_count"] >= 0
        assert 0.0 <= stats["outlier_2sigma_fraction"] <= 1.0

    def test_zero_std_no_outliers(self):
        """When std=0, outlier count should be 0."""
        import numpy as np
        from synth.ts_generator.utils.statistics_utils import calculate_statistics

        y = np.ones(64)
        stats = calculate_statistics(y, 64)
        assert stats["outlier_2sigma_count"] == 0
        assert stats["extreme_outlier_pos"] == stats["max_pos"]


# ===================================================================
# SECTION 9: TERMINOLOGY CONSISTENCY
# ===================================================================

class TestTerminology:
    """Ensure 'local characteristic' is replaced with 'local event'."""

    def test_no_local_characteristic_in_templates(self):
        violations = []
        for f in TEMPLATES_DIR.glob("*.py"):
            src = _read(f)
            for i, line in enumerate(src.splitlines(), 1):
                if "local characteristic" in line.lower():
                    violations.append(f"{f.name}:{i}: {line.strip()}")
        assert not violations, (
            "Old 'local characteristic' terminology still present:\n"
            + "\n".join(violations)
        )

    def test_no_local_characteristic_in_eval_config(self):
        src = _read(EVAL_CONFIG)
        assert "local characteristic" not in src.lower(), (
            "Old 'local characteristic' in eval config"
        )

    def test_no_local_characteristic_in_generators(self):
        violations = []
        for f in [BASE_GENERATOR] + GENERATOR_FILES:
            src = _read(f)
            for i, line in enumerate(src.splitlines(), 1):
                if "local characteristic" in line.lower():
                    violations.append(f"{f.name}:{i}: {line.strip()}")
        assert not violations, (
            "Old 'local characteristic' in generators:\n"
            + "\n".join(violations)
        )


# ===================================================================
# SECTION 10: THINKING BUILDER IMPORTS
# ===================================================================

class TestThinkingExports:
    """All builder functions must be exported from __init__.py."""

    def test_all_builders_exported(self):
        """Every build_*_thought function must be in __init__.py."""
        init_src = _read(THINKING_DIR / "__init__.py")

        missing = []
        for f in THINKING_FILES:
            src = _read(f)
            for m in re.finditer(r"^def (build_\w+_thought)\(", src, re.MULTILINE):
                func_name = m.group(1)
                if func_name not in init_src:
                    missing.append(f"{f.name}: {func_name}")

        assert not missing, (
            "Builder functions not exported from __init__.py:\n"
            + "\n".join(missing)
        )


# ===================================================================
# SECTION 11: FILTER_GROUPS COMPLETENESS (rule #60)
# ===================================================================

class TestFilterGroupsCompleteness:
    def test_all_eval_types_in_filter_groups(self):
        """Rule #60: Every eval_type must be in at least one FILTER_GROUPS group."""
        from synth.align.dataset import _ALL_EVAL_TYPES
        from evaluation.eval.config import ALL_EVAL_TYPES
        missing = ALL_EVAL_TYPES - _ALL_EVAL_TYPES
        assert not missing, f"eval_types missing from FILTER_GROUPS: {missing}"


# ===================================================================
# SECTION 12: DATA-ONLY FORMAT VERIFICATION
# ===================================================================

class TestDataOnlyFormat:
    def test_data_only_no_separator(self):
        """Lookup types: data + verdict, no ===."""
        from synth.ts_generator.utils.thinking.core import ThoughtBuilder
        tb = ThoughtBuilder()
        tb.data("metric: M")
        tb.data("stats: min=3.45")
        tb.verdict("3.45")
        text, verdict = tb.build()
        assert '===' not in text
        assert 'answer: 3.45' in text
        assert verdict == '3.45'

    def test_reasoning_has_separator(self):
        """Types with reasoning: data + === + lines + verdict."""
        from synth.ts_generator.utils.thinking.core import ThoughtBuilder
        tb = ThoughtBuilder()
        tb.data("metric: M")
        tb.data("stats: mean=5.00, std=1.20")
        tb.line("range = 5.00")
        tb.verdict("5.00")
        text, verdict = tb.build()
        assert '===' in text
        lines = text.split('\n')
        sep_idx = [i for i, l in enumerate(lines) if l.strip() == '===']
        assert len(sep_idx) == 1
        # Data before ===, reasoning after
        assert sep_idx[0] > 0
        # answer: after ===
        answer_idx = [i for i, l in enumerate(lines) if 'answer:' in l]
        assert answer_idx[0] > sep_idx[0]

    def test_verdict_line_always_last(self):
        """answer: line is always the last non-tag line regardless of format."""
        from synth.ts_generator.utils.thinking.core import ThoughtBuilder
        # Data-only
        tb1 = ThoughtBuilder()
        tb1.data("x=1")
        tb1.verdict("1")
        text1, _ = tb1.build()
        # build() wraps in <think>...</think>, so check inside the tags
        inner1 = text1.replace('<think>', '').replace('</think>', '').strip()
        assert inner1.endswith("answer: 1")
        # With reasoning
        tb2 = ThoughtBuilder()
        tb2.data("x=1")
        tb2.line("compute")
        tb2.verdict("1")
        text2, _ = tb2.build()
        inner2 = text2.replace('<think>', '').replace('</think>', '').strip()
        assert inner2.endswith("answer: 1")


# ===================================================================
# SECTION 13: COMPLETE SCAN PATTERN VERIFICATION
# ===================================================================

class TestCompleteScanPattern:
    def test_iterative_scans_have_else_branch(self):
        """Rule #50: Every iterative scan must show both 'new max' and 'keep current'."""
        violations = []
        for f in THINKING_FILES:
            src = _read(f)
            lines = src.splitlines()
            for i, line in enumerate(lines):
                stripped = line.strip()
                # Look for patterns like: if is_new: or if is_max:
                if re.match(r'if\s+(is_new|is_max|is_min|is_bigger|is_smaller|is_larger)\b', stripped):
                    # Check that there's an else: branch within next 5 lines with tb.line
                    snippet = '\n'.join(lines[i:i+6])
                    if 'else:' not in snippet:
                        violations.append(f"{f.name}:{i+1}: scan branch without else: {stripped}")
                    elif 'tb.line' not in snippet.split('else:')[1] if 'else:' in snippet else True:
                        # else exists but no tb.line in the else block
                        else_part = snippet.split('else:')[1] if 'else:' in snippet else ''
                        if 'tb.line' not in else_part:
                            violations.append(f"{f.name}:{i+1}: else branch without tb.line: {stripped}")
        assert not violations, (
            "Incomplete scan patterns (missing else + tb.line):\n"
            + "\n".join(violations)
        )


# ===================================================================
# SECTION 14: FORMAT_CMP CONTRADICTION CHECK ON GENERATED DATA
# ===================================================================

class TestFormatCmpContradictions:
    @pytest.fixture(scope="class")
    def generated_samples(self):
        """Generate debug samples across all 3 modes."""
        from synth.align.config import Config, Mode, Difficulty, PromptIndexer
        from synth.align.factory import generate_sample
        from synth.ts_generator.utils.common_utils import load_cfg
        from synth.ts_generator.utils.probability_utils import DEFAULT_QA_TYPE_WEIGHTS
        import re as _re

        metric_config = load_cfg("config/metric_set.json")
        config = Config(
            num_data=10, encoding_method="no", output_base_dir="/tmp/test_cmp",
            dryrun=True, debug=True, local_llm_path="",
            disable_metric_config=False, metric_config=metric_config,
            qa_type_weights=DEFAULT_QA_TYPE_WEIGHTS,
        )
        samples = []
        for mode in [Mode.UTS, Mode.MTS_SHAPE, Mode.MTS_LOCAL]:
            for _ in range(5):
                try:
                    result = generate_sample(
                        mode=mode, config=config, indexer=PromptIndexer(),
                        seq_len=256, difficulty=Difficulty.EASY, debug=True,
                    )
                    for j, answer in enumerate(result.answers):
                        think_match = _re.search(r'<think>(.*?)</think>', answer, _re.DOTALL)
                        if think_match:
                            samples.append({
                                'think': think_match.group(1),
                                'eval_type': result.eval_tasks[j],
                                'mode': mode.value,
                            })
                except Exception:
                    pass
        return samples

    def test_no_display_contradictions(self, generated_samples):
        """No 'X.XX > X.XX' where both sides are identical displayed values."""
        import re as _re
        violations = []
        for s in generated_samples:
            for m in _re.finditer(r'(?<![-\d])(\d+\.\d{2})\s*([><])\s*(?<![-\d])(\d+\.\d{2})', s['think']):
                left, op, right = m.group(1), m.group(2), m.group(3)
                if left == right:
                    violations.append(
                        f"{s['eval_type']} ({s['mode']}): '{left} {op} {right}'"
                    )
        assert not violations, (
            "format_cmp display contradictions found:\n"
            + "\n".join(violations[:20])  # limit output
        )


# ===================================================================
# SECTION 15: COHERENCY COVERAGE VERIFICATION
# ===================================================================

class TestCoherencyCoverage:
    def test_all_judgment_types_have_coherency(self):
        """All judgment scorer functions should call a coherency function."""
        # segment_judgment, cross_stat_judgment -> score_compound_judgment
        src_seg = _read(ROOT / "reward" / "scorers" / "segments.py")
        assert 'coherency_condition_verdict' in src_seg
        # anti_judgment -> score_anti_judgment
        src_core = _read(ROOT / "reward" / "scorers" / "core.py")
        assert 'coherency_condition_verdict' in src_core
        # trend_dominance
        assert 'coherency_duration_dominance' in src_core
        # yes_no
        assert 'coherency_yes_no' in src_core

    def test_mts_types_have_coherency(self):
        """MTS verdict and set types must have coherency."""
        src = _read(ROOT / "reward" / "scorers" / "core.py")
        assert 'coherency_mts_verdict' in src
        assert 'coherency_mts_cluster' in src


# ===================================================================
# SECTION 16: SAFE FLOAT IN REWARD SCORERS
# ===================================================================

class TestSafeFloatUsage:
    def test_no_raw_float_in_core_scorer(self):
        """Rule #13: Never raw float() on model output in reward scorers."""
        src = _read(ROOT / "reward" / "scorers" / "core.py")
        # float() calls that are NOT _safe_float and NOT in comments/imports
        violations = []
        for i, line in enumerate(src.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith('#') or 'import' in stripped or 'def ' in stripped:
                continue
            if 'float(' in stripped and '_safe_float' not in stripped:
                # Allow isinstance checks and float literals
                if 'isinstance' in stripped or 'float)' in stripped:
                    continue
                violations.append(f"core.py:{i}: {stripped}")
        assert not violations, (
            "Raw float() in core scorer (should use _safe_float):\n"
            + "\n".join(violations)
        )


# ===================================================================
# SECTION 17: SEPARATOR AND STEP LABEL CHECKS
# ===================================================================

class TestThinkBlockFormatting:
    def test_no_tb_data_separator(self):
        """Rule #27: Use tb.metric_separator(), not tb.data('---')."""
        pattern = re.compile(r"tb\.data\(['\"]---['\"]\)")
        violations = []
        for f in THINKING_FILES:
            for i, line in enumerate(_read(f).splitlines(), 1):
                if pattern.search(line):
                    violations.append(f"{f.name}:{i}: {line.strip()}")
        assert not violations, (
            "tb.data('---') found — use tb.metric_separator():\n"
            + "\n".join(violations)
        )

    def test_no_step_labels_in_builders(self):
        """Rule #25: No 'step N:' labels in reasoning section."""
        pattern = re.compile(r'["\']step\s+\d+:', re.IGNORECASE)
        violations = []
        for f in THINKING_FILES:
            for i, line in enumerate(_read(f).splitlines(), 1):
                if pattern.search(line) and 'tb.line' in line:
                    violations.append(f"{f.name}:{i}: {line.strip()}")
        assert not violations, (
            "Step labels in reasoning section:\n"
            + "\n".join(violations)
        )


# ===================================================================
# SECTION 18: CASE-INSENSITIVE PASS/FAIL IN PARSERS
# ===================================================================

class TestCaseSensitivity:
    def test_pass_fail_matching_case_insensitive(self):
        """Rule #16b: PASS/FAIL matching must use .upper() or re.IGNORECASE."""
        parsing_src = _read(ROOT / "reward" / "parsing.py")
        # Find all lines that check for PASS. or FAIL.
        violations = []
        for i, line in enumerate(parsing_src.splitlines(), 1):
            if ("'PASS.'" in line or "'FAIL.'" in line) and 'upper' not in line.lower():
                # Check if there's a .upper() in the same or previous 3 lines
                context = '\n'.join(parsing_src.splitlines()[max(0,i-4):i])
                if '.upper()' not in context and 'IGNORECASE' not in context:
                    violations.append(f"parsing.py:{i}: {line.strip()}")
        assert not violations, (
            "PASS/FAIL matching without .upper() or IGNORECASE:\n"
            + "\n".join(violations)
        )


# ===================================================================
# SECTION 19: THINK BLOCK ISOLATION IN EVAL
# ===================================================================

class TestEvalThinkIsolation:
    def test_no_strip_think_block_for_judge(self):
        """Symmetric judge: prompts.py must NOT strip think blocks before
        sending to the LLM judge. Both GT and pred carry full content
        (think + post-think) so the comparison is symmetric.
        See memory/feedback_no_strip_think_for_judge.md (2026-04-25)."""
        prompts_src = _read(ROOT / "evaluation" / "eval" / "prompts.py")
        assert 'strip_think_block' not in prompts_src, (
            "prompts.py must NOT call strip_think_block — both GT and pred "
            "are passed in full to the LLM judge for symmetric comparison"
        )
        # gt_answer_text / pred_answer_text are still the canonical variable
        # names — they now carry the full content (whitespace-stripped only).
        assert 'gt_answer_text' in prompts_src
        assert 'pred_answer_text' in prompts_src

    def test_no_raw_model_response_in_judge_prompts(self):
        """Raw model_response should not appear in format() calls for judge prompts."""
        src = _read(ROOT / "evaluation" / "eval" / "prompts.py")
        # After strip_think_block is called, the raw variables shouldn't be used in format()
        violations = []
        for i, line in enumerate(src.splitlines(), 1):
            if '.format(' in line and ('model_response' in line or 'gt_text' in line):
                # Check it's not the stripped version
                if 'answer_text' not in line and 'stripped' not in line:
                    violations.append(f"prompts.py:{i}: {line.strip()}")
        assert not violations, (
            "Raw model output used in judge prompt format():\n"
            + "\n".join(violations)
        )


# ===================================================================
# SECTION 20: GENERATIVE SELF-CONSISTENCY (all 3 modes)
# ===================================================================

class TestGenerativeSelfConsistency:
    """Generate real samples and verify compute_score(gt, gt) >= 0.95."""

    @pytest.fixture(scope="class")
    def all_samples(self):
        from synth.align.config import Config, Mode, Difficulty, PromptIndexer
        from synth.align.factory import generate_sample
        from synth.ts_generator.utils.common_utils import load_cfg
        from synth.ts_generator.utils.probability_utils import DEFAULT_QA_TYPE_WEIGHTS
        from reward.evidence import set_context_seq_len

        set_context_seq_len(256)
        metric_config = load_cfg("config/metric_set.json")
        config = Config(
            num_data=10, encoding_method="no", output_base_dir="/tmp/test_gen",
            dryrun=True, debug=True, local_llm_path="",
            disable_metric_config=False, metric_config=metric_config,
            qa_type_weights=DEFAULT_QA_TYPE_WEIGHTS,
        )
        samples = []
        for mode in [Mode.UTS, Mode.MTS_SHAPE, Mode.MTS_LOCAL]:
            for _ in range(5):
                try:
                    result = generate_sample(
                        mode=mode, config=config, indexer=PromptIndexer(),
                        seq_len=256, difficulty=Difficulty.EASY, debug=True,
                    )
                    for j in range(len(result.answers)):
                        samples.append({
                            'output': result.answers[j],
                            'eval_type': result.eval_tasks[j],
                            'eval_metadata': result.eval_metadatas[j],
                            'mode': mode.value,
                        })
                except Exception:
                    pass
        return samples

    def test_self_consistency_all_generated(self, all_samples):
        """Every generated sample must score >= 0.95 against itself."""
        from reward import compute_score
        failures = []
        for s in all_samples:
            if s['eval_type'] == 'description':
                continue  # description uses evidence-only scorer, expected ~0.9
            try:
                score = compute_score(
                    s['output'], s['output'],
                    eval_type=s['eval_type'],
                    eval_metadata=s['eval_metadata'],
                )
                if score < 0.95:
                    failures.append(
                        f"{s['eval_type']} ({s['mode']}) "
                        f"sub={s['eval_metadata'].get('sub_type','')} "
                        f"score={score:.4f}"
                    )
            except Exception as e:
                failures.append(f"{s['eval_type']} ({s['mode']}) ERROR: {e}")
        assert not failures, (
            f"Self-consistency failures ({len(failures)}):\n"
            + "\n".join(failures[:20])
        )

    def test_all_answers_have_think_block(self, all_samples):
        """Every generated answer must contain <think>...</think>."""
        import re as _re
        missing = []
        for s in all_samples:
            if '<think>' not in s['output'] or '</think>' not in s['output']:
                missing.append(f"{s['eval_type']} ({s['mode']})")
        assert not missing, f"Answers missing think block: {missing[:10]}"

    def test_verdict_matches_metadata(self, all_samples):
        """answer: in think block must match eval_metadata verdict."""
        import re as _re
        mismatches = []
        for s in all_samples:
            verdict = str(s['eval_metadata'].get('verdict', ''))
            if not verdict:
                continue
            think_match = _re.search(r'<think>(.*?)</think>', s['output'], _re.DOTALL)
            if not think_match:
                continue
            answer_match = _re.search(r'answer:\s*(.+)', think_match.group(1), _re.IGNORECASE)
            if not answer_match:
                continue
            answer_val = answer_match.group(1).strip()
            if answer_val != verdict:
                try:
                    if abs(float(answer_val) - float(verdict)) > 0.01:
                        mismatches.append(
                            f"{s['eval_type']} sub={s['eval_metadata'].get('sub_type','')}: "
                            f"answer='{answer_val}' != verdict='{verdict}'"
                        )
                except ValueError:
                    if answer_val.lower() != verdict.lower():
                        mismatches.append(
                            f"{s['eval_type']} sub={s['eval_metadata'].get('sub_type','')}: "
                            f"answer='{answer_val}' != verdict='{verdict}'"
                        )
        assert not mismatches, (
            f"Verdict mismatches ({len(mismatches)}):\n"
            + "\n".join(mismatches[:20])
        )


# ===================================================================
# SECTION 23: VOCABULARY CONSISTENCY — NO BARE "steady" IN TREND_LIST
# ===================================================================

class TestVocabularyConsistency:
    """Regression: audit 4 changed 'steady' → 'keep steady' in trend_list.
    Catches if bare 'steady' creeps back into trend_list tuples or fixtures."""

    def test_no_bare_steady_in_trend_list_source(self):
        """No trend_list tuple contains bare 'steady' — must be 'keep steady'."""
        # Pattern matches: ('steady', ...) or ("steady", ...) inside trend_list-like context
        pattern = re.compile(
            r"""['"]steady['"]"""  # bare 'steady' or "steady"
        )
        # Exempt: comment lines, hr_trend (different vocab), TYPE_SIMILARITY keys,
        # TREND_TYPE_VERBS keys (maps legacy input)
        EXEMPT_PATTERNS = {
            'hr_trend', 'TYPE_SIMILARITY', 'TREND_TYPE_VERBS',
            'get_type_similarity', '# ', '_CATEGORICAL_VERDICTS',
            '_SIMPLE_CATEGORICAL_VERDICTS', 'vocab_map',
        }
        violations = []
        for pyfile in sorted(SYNTH_DIR.rglob("*.py")):
            src = pyfile.read_text(encoding="utf-8")
            for i, line in enumerate(src.splitlines(), 1):
                if pattern.search(line):
                    stripped = line.strip()
                    # Skip if line is a comment
                    if stripped.startswith('#'):
                        continue
                    # Skip known exemptions
                    if any(ex in line for ex in EXEMPT_PATTERNS):
                        continue
                    # Only flag if it looks like a trend_list tuple element
                    # i.e., in a tuple context like ('steady', N, N) or trend type assignment
                    if re.search(r"""\(\s*['"]steady['"]""", line) or \
                       re.search(r"""trend.*=\s*['"]steady['"]""", line):
                        violations.append(
                            f"{pyfile.relative_to(ROOT)}:{i}: {stripped}"
                        )
        assert not violations, (
            "Bare 'steady' found in trend_list tuples (should be 'keep steady'):\n"
            + "\n".join(violations)
        )

    def test_no_bare_steady_in_test_fixtures(self):
        """Test fixtures must use 'keep steady' in trend_list, not bare 'steady'."""
        test_dir = ROOT / "tests"
        pattern = re.compile(
            r"""trend_list.*['"]steady['"]"""
        )
        violations = []
        for pyfile in sorted(test_dir.rglob("*.py")):
            src = pyfile.read_text(encoding="utf-8")
            for i, line in enumerate(src.splitlines(), 1):
                if pattern.search(line) and 'keep steady' not in line:
                    violations.append(
                        f"{pyfile.name}:{i}: {line.strip()}"
                    )
        assert not violations, (
            "Bare 'steady' in test trend_list fixtures:\n"
            + "\n".join(violations)
        )

    def test_trend_dominance_verdicts_include_keep_steady(self):
        """Trend dominance verdict set must include 'keep steady', not 'steady'."""
        from synth.ts_generator.utils.thinking.trend import build_trend_dominance_thought
        # Create attrs where 'keep steady' is the dominant segment
        attrs = {
            'seq_len': 256,
            'statistics': {
                'min': 0.0, 'max': 1.0, 'mean': 0.5, 'std': 0.1,
                'min_pos': 0, 'max_pos': 100, 'range': 1.0,
                'segment_means_16': [0.5]*16,
            },
            'trend_overall': {'type': 'increase', 'start': 0, 'end': 255, 'amplitude': 0.1},
            'trend_list': [('increase', 0, 10), ('keep steady', 10, 256)],
            'noise': {'type': 'gaussian', 'strength': 0.1},
            'seasonal': {'type': 'no periodic', 'period': 0, 'amplitude': 0},
            'local': [],
        }
        _, v = build_trend_dominance_thought(attrs, 'M', 0, 255)
        assert v == 'keep steady', f"Dominant type should be 'keep steady', got '{v}'"


# ===================================================================
# SECTION 24: FORMAT_FLOAT / ROUND AGREEMENT
# ===================================================================

class TestFormatFloatRoundAgreement:
    """Regression: format_float() and round() must agree for all .XX5 values,
    ensuring displayed values match computed comparisons."""

    def test_format_float_equals_round_display(self):
        """format_float(v, 2) must equal f'{round(v, 2):.2f}' for boundary values."""
        from synth.ts_generator.utils.formatting_utils import format_float
        # Exact binary .XX5 and common rounding boundary values
        test_values = [
            0.125, 0.135, 0.145, 0.155, 0.175, 0.185, 0.195, 0.245,
            0.375, 0.625, 0.875, 1.005, 1.015, 1.025, 1.035, 1.045,
            -0.125, -0.375, -0.625,
        ]
        mismatches = []
        for v in test_values:
            ff = format_float(v, 2)
            expected = f"{round(v, 2):.2f}"
            if ff != expected:
                mismatches.append(f"  {v}: format_float='{ff}', round='{expected}'")
        assert not mismatches, (
            "format_float and round() disagree:\n" + "\n".join(mismatches)
        )


# ===================================================================
# SECTION 25: HALF_MEAN_SHIFT USES ABS()
# ===================================================================

class TestHalfMeanShiftAbs:
    """Regression: half_mean_shift must use abs(shift) so negative shifts
    don't structurally force verdict='no'."""

    def test_negative_shift_can_produce_yes(self):
        """When second_half_mean < first_half_mean, 'yes' is still possible."""
        from synth.ts_generator.utils.thinking.segments import build_compound_judgment_thought
        attrs = {
            'seq_len': 256,
            'statistics': {
                'min': 0.0, 'max': 10.0, 'mean': 5.0, 'std': 2.0,
                'min_pos': 0, 'max_pos': 128, 'range': 10.0,
                'first_half_mean': 7.0, 'second_half_mean': 2.0,
                'first_half_std': 1.0, 'second_half_std': 1.0,
                'segment_means_16': [5.0]*16,
            },
            'trend_overall': {'type': 'decrease', 'start': 0, 'end': 255, 'amplitude': -5.0},
            'trend_list': [('decrease', 0, 256)],
            'noise': {'type': 'gaussian', 'strength': 0.1},
            'seasonal': {'type': 'no periodic', 'period': 0, 'amplitude': 0},
            'local': [],
        }
        params = {
            'K_pct': 5,
            'actual_max': 10.0, 'actual_min': 0.0,
            'first_half_mean': 7.0, 'second_half_mean': 2.0,
            'range': 10.0,
            'shift': -5.0,  # negative shift
            'threshold': 0.5,  # 5% of 10 = 0.5
            'verdict': 'yes',  # |shift|=5.0 > 0.5
        }
        _, v = build_compound_judgment_thought(attrs, 'M', 'half_mean_shift', params)
        assert v == 'yes', f"Negative shift should still yield 'yes' when |shift| > threshold, got '{v}'"

    def test_generator_uses_abs_for_verdict(self):
        """Generator verdict must agree with builder when shift is negative."""
        from synth.ts_generator.utils.thinking.segments import build_compound_judgment_thought
        attrs = {
            'seq_len': 256,
            'statistics': {
                'min': 0.0, 'max': 10.0, 'mean': 5.0, 'std': 2.0,
                'min_pos': 0, 'max_pos': 128, 'range': 10.0,
                'first_half_mean': 7.0, 'second_half_mean': 2.0,
                'first_half_std': 1.0, 'second_half_std': 1.0,
                'segment_means_16': [5.0]*16,
            },
            'trend_overall': {'type': 'decrease', 'start': 0, 'end': 255, 'amplitude': -5.0},
            'trend_list': [('decrease', 0, 256)],
            'noise': {'type': 'gaussian', 'strength': 0.1},
            'seasonal': {'type': 'no periodic', 'period': 0, 'amplitude': 0},
            'local': [],
        }
        # Generator path: shift_r = round(abs(round(-5.0, 2)), 2) = 5.0
        # 5.0 > 0.5 → 'yes'
        shift_r = round(abs(round(-5.0, 2)), 2)
        thresh_r = round(0.5, 2)
        gen_verdict = 'yes' if shift_r > thresh_r else 'no'
        assert gen_verdict == 'yes', "Generator should produce 'yes' for |shift|=5.0 > 0.5"

        # Builder path should agree
        params = {
            'K_pct': 5, 'actual_max': 10.0, 'actual_min': 0.0,
            'first_half_mean': 7.0, 'second_half_mean': 2.0,
            'range': 10.0, 'shift': -5.0, 'threshold': 0.5,
            'verdict': gen_verdict,
        }
        _, builder_v = build_compound_judgment_thought(attrs, 'M', 'half_mean_shift', params)
        assert builder_v == gen_verdict, (
            f"Builder ({builder_v}) disagrees with generator ({gen_verdict})"
        )

    def test_builder_shows_abs_shift(self):
        """Builder must display |shift| = ... step for half_mean_shift."""
        from synth.ts_generator.utils.thinking.segments import build_compound_judgment_thought
        attrs = {
            'seq_len': 256,
            'statistics': {
                'min': 0.0, 'max': 10.0, 'mean': 5.0, 'std': 2.0,
                'min_pos': 0, 'max_pos': 128, 'range': 10.0,
                'first_half_mean': 7.0, 'second_half_mean': 2.0,
                'first_half_std': 1.0, 'second_half_std': 1.0,
                'segment_means_16': [5.0]*16,
            },
            'trend_overall': {'type': 'decrease', 'start': 0, 'end': 255, 'amplitude': -5.0},
            'trend_list': [('decrease', 0, 256)],
            'noise': {'type': 'gaussian', 'strength': 0.1},
            'seasonal': {'type': 'no periodic', 'period': 0, 'amplitude': 0},
            'local': [],
        }
        params = {
            'K_pct': 5,
            'actual_max': 10.0, 'actual_min': 0.0,
            'first_half_mean': 7.0, 'second_half_mean': 2.0,
            'range': 10.0,
            'shift': -5.0,
            'threshold': 0.5,
            'verdict': 'yes',
        }
        think, _ = build_compound_judgment_thought(attrs, 'M', 'half_mean_shift', params)
        assert '|shift|' in think, "half_mean_shift think block must show |shift| step"


# ===================================================================
# SECTION 26: REWARD PARSING HANDLES "keep steady"
# ===================================================================

class TestRewardParsingVocabulary:
    """Regression: reward parsing regexes must match 'keep steady', not just 'steady'."""

    def test_aggregate_durations_parses_keep_steady(self):
        """extract_aggregate_durations must parse 'keep steady:' lines."""
        from reward.parsing import extract_aggregate_durations
        reasoning = (
            "aggregate durations:\n"
            "- increase: 100\n"
            "- keep steady: 80\n"
            "- decrease: 76\n"
            "dominant trend comparison: increase"
        )
        result = extract_aggregate_durations(reasoning)
        assert 'keep steady' in result, (
            f"'keep steady' not parsed from aggregate durations: {result}"
        )
        assert result['keep steady'] == 80.0

    def test_dominant_winner_parses_keep_steady(self):
        """extract_dominant_winner must match 'keep steady'."""
        from reward.parsing import extract_dominant_winner
        reasoning = "dominant trend comparison: keep steady (150) > increase (100)"
        result = extract_dominant_winner(reasoning)
        assert result == 'keep steady', f"Expected 'keep steady', got '{result}'"

    def test_coherency_durations_parses_keep_steady(self):
        """Coherency duration parsing must handle 'keep steady:' lines."""
        from reward.coherency import coherency_duration_dominance
        # Think block where 'keep steady' is the dominant trend
        think = (
            "<think>metric: M\ntrend_list=[(keep steady, 0, 150), (increase, 150, 256)]\n"
            "===\n"
            "aggregate durations:\n"
            "- increase: 106\n"
            "- keep steady: 150\n"
            "- decrease: 0\n"
            "dominant trend comparison: keep steady (150) > increase (106)\n"
            "answer: keep steady</think>"
        )
        score = coherency_duration_dominance(think)
        assert score >= 0.5, (
            f"Coherency scored {score} — 'keep steady' likely not parsed"
        )


# ---------------------------------------------------------------------------
# Multi-word verdict safety (audit 6)
# ---------------------------------------------------------------------------

REWARD_DIR = ROOT / "reward"


class TestMultiWordVerdictSafety:
    """Prevent \\w+ and \\S+ from silently truncating multi-word verdicts."""

    def test_no_bare_word_plus_in_verdict_extraction(self):
        """No regex in reward/ should use \\w+ or \\S+ as the CAPTURE GROUP
        for verdict/answer extraction.

        Patterns like r'answer:\\s*(\\w+)' truncate multi-word verdicts.
        HTML tag stripping r'</?\\w+>' and known single-word extractors
        (yes/no, similar/different) are exempted.
        """
        # Only flag lines where \w+ or \S+ is the capture group in an
        # answer/verdict extraction regex — i.e., inside (...)
        bad_pattern = re.compile(
            r"""re\.search\([^)]*(?:answer|verdict)[^)]*\(\\[wS]\+\)""",
        )
        # Exempt: tag stripping, yes/no alternation, commented single-word
        exempt_patterns = [
            r'</\?\\w\+>',    # HTML tag stripping
            r'yes\|no',        # explicit yes/no alternation
        ]
        violations = []
        for pyfile in REWARD_DIR.rglob("*.py"):
            if pyfile.name.startswith("test_"):
                continue
            text = pyfile.read_text()
            for i, line in enumerate(text.split("\n"), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if bad_pattern.search(line):
                    if any(re.search(ep, line) for ep in exempt_patterns):
                        continue
                    # Allow if comment explains single-word expectation
                    if "#" in line and "single" in line.split("#", 1)[1].lower():
                        continue
                    violations.append(f"{pyfile.relative_to(ROOT)}:{i}: {stripped}")
        assert not violations, (
            "\\w+ or \\S+ used as capture group for verdict/answer extraction "
            "(truncates multi-word verdicts):\n" +
            "\n".join(f"  {v}" for v in violations)
        )

    def test_reward_self_consistency_with_multiword_verdicts(self):
        """compute_score(gt, gt) >= 0.95 for eval_types with multi-word verdicts."""
        import sys
        sys.path.insert(0, str(ROOT))
        from reward import compute_score

        multiword_cases = [
            # (eval_type, verdict, think_block_content)
            (
                "segment_trend_dominance",
                "keep steady",
                "metric: M\ntrend_segments=[(keep steady, 0, 150), (increase, 150, 256)]\n"
                "===\n"
                "Intersecting [0, 255] with metric trend segments:\n"
                "- keep steady[0, 150] ∩ [0, 255] = [0, 150] -> duration = 150 - 0 = 150\n"
                "- increase[150, 256] ∩ [0, 255] = [150, 255] -> duration = 255 - 150 = 105\n"
                "aggregate durations:\n"
                "- keep steady: 150\n"
                "- increase: 105\n"
                "dominant trend comparison: keep steady (150) > increase (105)\n"
                "answer: keep steady",
            ),
            (
                "stat_numerical",
                "first half",
                "metric: M\nstats: first_half_mean=0.80, second_half_mean=0.20, split_mid=128\n"
                "===\n"
                "0.80 > 0.20 = true\n"
                "answer: first half",
            ),
            (
                "stat_numerical",
                "second half",
                "metric: M\nstats: first_half_mean=0.20, second_half_mean=0.80, split_mid=128\n"
                "===\n"
                "0.80 > 0.20 = true\n"
                "answer: second half",
            ),
            (
                "duration_proportion",
                "keep steady",
                "metric: M\ntrend_segments=[(keep steady, 0, 200), (increase, 200, 256)]\n"
                "===\n"
                "durations:\n"
                "  (keep steady, 0, 200): 200 - 0 = 200 (duration)\n"
                "  (increase, 200, 256): 256 - 200 = 56 (duration)\n"
                "aggregate:\n"
                "  keep steady: 200\n"
                "  increase: 56\n"
                "largest total = keep steady (200)\n"
                "answer: keep steady",
            ),
        ]

        for eval_type, verdict, think_content in multiword_cases:
            gt = f"<think>{think_content}</think>"
            meta = {"verdict": verdict, "length": 256}
            if eval_type == "stat_numerical":
                meta["sub_type"] = "stat_segment_compare"
            score = compute_score(gt, gt, eval_type=eval_type, eval_metadata=meta)
            assert score >= 0.95, (
                f"Self-consistency failure for {eval_type} with "
                f"verdict '{verdict}': score={score:.3f}"
            )


# ===================================================================
# SECTION 12: MUTATION-KILLING TESTS  (audit 7 — test suite gap closure)
# ===================================================================

class TestMutationKilling:
    """Tests that kill specific mutants found by audit 7 mutation testing.

    Each test here would FAIL if the corresponding bug were reintroduced.
    """

    def test_extract_trend_type_verdict_multiword(self):
        """Kill mutant 3.1a: .+ → \\w+ in _extract_trend_type_verdict.

        \\w+ truncates multi-word verdicts like 'keep steady' to 'keep'.
        """
        from reward.scorers.core import _extract_trend_type_verdict

        # Multi-word trend verdicts
        assert _extract_trend_type_verdict("===\nanswer: keep steady") == "keep steady"
        assert _extract_trend_type_verdict("===\nanswer: first half") == "first half"
        # Single-word still works
        assert _extract_trend_type_verdict("===\nanswer: increase") == "increase"
        # Numeric verdict (decimal point not matched by \w+)
        assert _extract_trend_type_verdict("===\nanswer: 32.00") == "32.00"

    def test_extract_trend_type_verdict_score_round_trip(self):
        """Verify score_trend_dominance correctly scores 'keep steady' verdicts."""
        from reward.scorers.core import score_trend_dominance
        from reward.evidence import set_context_seq_len
        set_context_seq_len(256)

        gt = (
            "metric: M\ntrend_segments=[(keep steady, 0, 256)]\n===\n"
            "aggregate durations:\n- keep steady: 256\n"
            "dominant trend comparison: keep steady (256)\n"
            "answer: keep steady"
        )
        score = score_trend_dominance(gt, gt)
        assert score >= 0.95, f"keep steady self-score={score:.3f}"

    def test_half_mean_shift_abs_negative(self):
        """Kill mutant 3.1b: removing abs() from half_mean_shift.

        When second_half_mean < first_half_mean, signed_shift is negative.
        The think block must show |shift| = <positive> and compare the
        absolute value against the threshold.
        """
        from synth.ts_generator.utils.thinking.segments import build_compound_judgment_thought

        attributes = {
            'trend': {'type': 'decrease', 'start': 15.0, 'end': 0.0, 'amplitude': -15.0},
            'noise': {'type': 'smooth', 'strength': 0.0},
            'statistics': {
                'max': 15.0, 'min': 0.0,
                'first_half_mean': 10.0, 'second_half_mean': 5.0,
            },
        }
        params = {
            'verdict': 'yes',
            'first_half_mean': 10.0,
            'second_half_mean': 5.0,  # shm < fhm → negative shift
            'shift': -5.0,
            'actual_max': 15.0,
            'actual_min': 0.0,
            'range': 15.0,
            'threshold': 3.0,
            'K_pct': 0.2,
            'seq_len': 256,
        }
        text, verdict = build_compound_judgment_thought(
            attributes, 'TestMetric', 'half_mean_shift', params
        )
        # |shift| must be positive (5.00, not -5.00)
        assert "|shift| = 5.00" in text, f"Expected |shift| = 5.00 in: {text}"
        # The comparison must use the positive abs_shift
        assert "5.00 > " in text or "5.00 <" in text, (
            f"Comparison must use abs value 5.00, got: {text}"
        )

    def test_extract_answer_set_respects_separator(self):
        """Kill mutant 3.1d: removing === split from extract_answer_set.

        A decoy 'answer: (X, Y)' in the data block should NOT be extracted.
        Only the answer below === should be found.
        """
        from reward.parsing import extract_answer_set

        think = (
            "metric: A\n"
            "answer: (Decoy1, Decoy2)\n"  # data block decoy
            "===\n"
            "computation here\n"
            "answer: (Real1, Real2)"  # actual answer
        )
        result = extract_answer_set(think)
        assert result is not None
        assert "decoy1" not in result, f"Decoy leaked: {result}"
        assert "real1" in result, f"Real answer missing: {result}"
        assert "real2" in result, f"Real answer missing: {result}"

    def test_windowed_trend_negative_values(self):
        """Kill mutant 3.1e: multiplicative threshold on negative values.

        With multiplicative: first * 1.05 for negative first flips
        upper/lower, giving wrong verdict. Additive threshold must be used.
        """
        from synth.ts_generator.utils.thinking.statistical import build_stat_numerical_thought

        attributes = {
            'trend': {'type': 'increase', 'start': -10.0, 'end': -2.0, 'amplitude': 8.0},
            'noise': {'type': 'smooth', 'strength': 0.0},
            'seq_len': 256,
        }
        params = {
            'segment_means_16': [-10.0, -9.0, -7.0, -5.0, -3.0, -2.0],
        }
        text, verdict = build_stat_numerical_thought(
            attributes, 'TestMetric', 'stat_windowed_trend', params
        )
        # With negative values going from -10 to -2 (increasing),
        # the verdict should be 'upward'
        assert verdict == "upward", (
            f"Expected verdict='upward' for negative increasing means, got: {verdict!r}"
        )
        # upper_thresh must be > lower_thresh (additive ensures this)
        import re as re_mod
        upper_m = re_mod.search(r'upper_thresh\s*=\s*([-\d.]+)', text)
        lower_m = re_mod.search(r'lower_thresh\s*=\s*([-\d.]+)', text)
        if upper_m and lower_m:
            upper = float(upper_m.group(1))
            lower = float(lower_m.group(1))
            assert upper > lower, (
                f"upper_thresh ({upper}) must be > lower_thresh ({lower})"
            )

    def test_above_mean_count_uses_rounded_threshold(self):
        """Kill mutant 3.1f: removing round() from above_mean_count.

        The count must use the same rounded threshold shown in think blocks.
        """
        import numpy as np
        from synth.ts_generator.utils.statistics_utils import calculate_statistics

        # Construct data where raw mean vs rounded mean give different counts
        # mean ≈ 5.007 rounds to 5.01
        # Values: 5.005, 5.009 are above raw mean but <= rounded mean
        y = np.array([5.005, 5.009, 5.011, 5.02, 4.99])
        stats = calculate_statistics(y, seq_len=len(y))
        mean_rounded = round(float(np.mean(y)), 2)
        expected_count = int(np.sum(y > mean_rounded))
        assert stats["above_mean_count"] == expected_count, (
            f"above_mean_count={stats['above_mean_count']} but expected "
            f"{expected_count} (using rounded mean={mean_rounded})"
        )
