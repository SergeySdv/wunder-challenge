import numpy as np
from src.features.extractor import FeatureExtractor, feature_dim


def test_feature_dim():
    assert feature_dim(10) == 1185


def test_stream_vs_batch_consistency():
    rng = np.random.default_rng(0)
    n_lags = 10
    feat = FeatureExtractor(n_lags=n_lags)
    # synthetic sequence: 15 steps, 32 dims
    seq = rng.normal(size=(15, 32)).astype(np.float32)

    stream_feats = []
    for step, state in enumerate(seq):
        f = feat.stream(state, step_in_seq=step, seq_ix=0)
        if f is not None:
            stream_feats.append(f)

    batch_feats = []
    for end in range(n_lags - 1, len(seq) - 1):
        window = seq[end - n_lags + 1 : end + 1]
        f = feat.build_window_features(window, step_in_seq=end)
        batch_feats.append(f)

    assert len(stream_feats) == len(batch_feats)
    for sf, bf in zip(stream_feats, batch_feats):
        np.testing.assert_allclose(sf, bf, rtol=1e-5, atol=1e-6)
