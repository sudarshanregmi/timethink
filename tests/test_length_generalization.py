"""Regression tests for length generalization (Option C).

After removing `seq_len > 256` gates, generators must produce valid samples
at long lengths (768, 1536, 4096). This test catches regressions where
a generator silently returns None or crashes at a long seq_len.

See memory/pipeline_length_generalization_2026_04_21.md for design rationale.
"""
import random
import numpy as np
import pytest

from synth.align.config import Config, Mode, PromptIndexer, Difficulty
from synth.align.factory import generate_sample


@pytest.fixture(scope="module")
def config():
    return Config.from_yaml("config/datagen_config.yaml")


@pytest.mark.parametrize("mode", [Mode.UTS, Mode.MTS_LOCAL, Mode.MTS_SHAPE])
@pytest.mark.parametrize("seq_len", [256, 768, 1536, 4096])
def test_generate_sample_at_long_length(config, mode, seq_len):
    """Each mode must generate at least one valid sample at each length in
    [256, 4096]. Before the gate-relaxation fix, long lengths would silently
    crash the sample via the None-return bug in AtomicMeanGenerator.
    """
    random.seed(42)
    np.random.seed(42)
    # Allow up to 10 trials per (mode, length) — generation is stochastic
    # and some trials may legitimately skip (e.g., no periodicity found).
    succeeded = 0
    for _ in range(10):
        try:
            indexer = PromptIndexer()
            result = generate_sample(
                mode=mode,
                config=config,
                indexer=indexer,
                seq_len=seq_len,
                difficulty=Difficulty.EASY,
            )
            assert result.original_timeseries is not None
            assert len(result.original_timeseries[0]) == seq_len
            assert len(result.questions) >= 1
            assert len(result.answers) >= 1
            assert "<think>" in result.answers[0]
            assert "</think>" in result.answers[0]
            succeeded += 1
        except Exception:
            continue
    assert succeeded >= 3, (
        f"{mode.value} at seq_len={seq_len}: only {succeeded}/10 trials succeeded"
    )


def test_atomic_global_mean_at_long_length():
    """AtomicMeanGenerator.generate_global_mean must not return None at long
    seq_len. The prior gate (MAX_RANGE=256) combined with the missing None
    check in generate_all caused silent crashes.
    """
    from synth.align.generators.atomic_qa import AtomicMeanGenerator

    random.seed(0)
    np.random.seed(0)
    for seq_len in [256, 768, 1536, 4096]:
        ts = np.random.randn(seq_len).astype(float)
        gen = AtomicMeanGenerator(ts, "metric", seq_len)
        r = gen.generate_global_mean()
        assert r is not None, f"A1 returned None at seq_len={seq_len}"
        assert r["eval_type"] == "atomic_global_mean"
        assert r["eval_metadata"]["length"] == seq_len
        # Sanity: think block must contain chunk computation
        assert "<think>" in r["answer"] and "</think>" in r["answer"]


def test_atomic_mean_generate_all_robust_to_none():
    """generate_all must None-check every sub-method (regression test for the
    append(None) bug at atomic_qa.py:389).
    """
    from synth.align.generators.atomic_qa import AtomicMeanGenerator

    # seq_len=16 is below MIN_RANGE=32, so A2/A3 should return None.
    # But A1 still works at any length. The test ensures no None leaks into
    # the returned list.
    ts = np.array([1.0] * 16)
    gen = AtomicMeanGenerator(ts, "metric", 16)
    results = gen.generate_all()
    for r in results:
        assert r is not None
        assert "question" in r
        assert "answer" in r
