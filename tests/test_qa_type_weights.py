"""Regression tests for the qa_type_weights validation + taxonomy split.

Guards against the silent-drop bug we caught 2026-04-19: the yaml used to
lack a `taxonomy` weight, so `select_weighted_qa_index`'s `.get(t, 0.0)`
silently dropped every `atomic_*`/`rl_*`/`bridge_*` sample in production.

The fix has three parts we test here:
  1. Startup validation aborts if the yaml is missing a qa_type the code emits.
  2. The former `taxonomy` lump is split into `sft_atomic` / `sft_bridge` /
     `rl_composition` so macro weights are semantically meaningful.
  3. `_classify_taxonomy_qa` dispatches bridges (eval_metadata['bridge']=True)
     BEFORE checking the `rl_` / `atomic_` prefix — bridges share eval_types
     with their RL counterparts, so prefix-only dispatch would mis-bucket them.
"""
import pytest
import warnings
from synth.align.base_generator import _classify_taxonomy_qa
from synth.align.config import _validate_qa_type_weights, Config
from synth.ts_generator.utils.probability_utils import (
    QAType, DEFAULT_QA_TYPE_WEIGHTS, select_weighted_qa_index,
)


# ---------------------------------------------------------------------------
# Validation aborts on missing keys (the silent-drop guard)
# ---------------------------------------------------------------------------

class TestValidationAborts:
    def test_missing_qa_type_aborts(self):
        """Every QAType the enum declares must have a weight in yaml."""
        full = {qt.value: 0.1 for qt in QAType}
        for qt in QAType:
            partial = {k: v for k, v in full.items() if k != qt.value}
            with pytest.raises(ValueError, match="missing entries"):
                _validate_qa_type_weights(partial, "test.yaml")

    def test_valid_weights_pass_silently(self):
        """A complete weights dict produces no exception and no warning."""
        full = {qt.value: 0.1 for qt in QAType}
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            _validate_qa_type_weights(full, "test.yaml")
            # Only the "not 1.0" warning should fire (our default is 0.1 each)
            raise_count = sum(1 for w in ws if "missing" in str(w.message))
            assert raise_count == 0

    def test_dead_key_warns_does_not_abort(self):
        """Dead keys are a hygiene issue, not a correctness issue."""
        full = {qt.value: 0.1 for qt in QAType}
        full["legacy_removed_bucket"] = 0.05
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            _validate_qa_type_weights(full, "test.yaml")
            msgs = [str(w.message) for w in ws]
            assert any("not recognized" in m for m in msgs)

    def test_real_config_passes(self):
        """The checked-in datagen_config.yaml must pass validation."""
        c = Config.from_yaml()
        assert set(c.qa_type_weights.keys()) == {qt.value for qt in QAType}
        assert abs(sum(c.qa_type_weights.values()) - 1.0) < 1e-3


# ---------------------------------------------------------------------------
# Taxonomy dispatch correctness
# ---------------------------------------------------------------------------

class TestTaxonomyDispatch:
    def test_atomic_single_goes_to_sft_atomic(self):
        r = {'eval_type': 'atomic_global_mean', 'eval_metadata': {'length': 256}}
        assert _classify_taxonomy_qa(r) == QAType.SFT_ATOMIC

    def test_atomic_cross_goes_to_sft_atomic(self):
        r = {'eval_type': 'atomic_cross_stat_compare',
             'eval_metadata': {'length': 256}}
        assert _classify_taxonomy_qa(r) == QAType.SFT_ATOMIC

    def test_rl_single_goes_to_rl_composition(self):
        r = {'eval_type': 'rl_half_mean_compare', 'eval_metadata': {'length': 256}}
        assert _classify_taxonomy_qa(r) == QAType.RL_COMPOSITION

    def test_rl_cross_goes_to_rl_composition(self):
        r = {'eval_type': 'rl_cross_stat_ratio', 'eval_metadata': {'length': 256}}
        assert _classify_taxonomy_qa(r) == QAType.RL_COMPOSITION

    def test_bridge_flag_overrides_rl_prefix(self):
        """Bridges share eval_types with their RL counterparts — the bridge
        flag must dispatch first, or they'd get bucketed as RL_COMPOSITION
        and lose their force-SFT routing's statistical weight."""
        r = {'eval_type': 'rl_half_mean_compare',
             'eval_metadata': {'length': 256, 'bridge': True}}
        assert _classify_taxonomy_qa(r) == QAType.SFT_BRIDGE

    def test_bridge_flag_overrides_atomic_prefix(self):
        r = {'eval_type': 'atomic_interval_mean',
             'eval_metadata': {'length': 256, 'bridge': True}}
        assert _classify_taxonomy_qa(r) == QAType.SFT_BRIDGE


# ---------------------------------------------------------------------------
# Fail-open default at level-1 (defense-in-depth)
# ---------------------------------------------------------------------------

class TestSelectionFailsOpen:
    def test_missing_weight_defaults_to_one_not_zero(self):
        """Even if startup validation is bypassed (e.g. runtime-injected
        weights dict), a missing qa_type should default to weight 1.0 —
        never weight 0.0 (which would silently drop every sample)."""
        # One seed with two QA types; only one has a weight
        qa_types = [QAType.DESCRIPTION.value, QAType.RL_COMPOSITION.value]
        weights = {QAType.DESCRIPTION.value: 0.10}  # rl_composition unspecified

        # Run many trials. rl_composition should get selected at non-zero rate.
        n = 1000
        rl_selected = sum(
            1 for _ in range(n)
            if qa_types[select_weighted_qa_index(qa_types, weights)]
            == QAType.RL_COMPOSITION.value
        )
        # With fail-open default 1.0, rl weight ≫ description weight
        # (1.0 vs 0.10), so rl should dominate selections.
        assert rl_selected > n // 2, (
            f"Expected rl_composition to dominate with fail-open default, "
            f"got {rl_selected}/{n}. Likely means default reverted to 0.0 "
            f"(silent drop bug)."
        )
