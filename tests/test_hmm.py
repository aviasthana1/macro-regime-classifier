from macro_regime_classifier.data.synthetic import generate_cross_asset_panel
from macro_regime_classifier.detectors.hmm import HMMRegimeDetector
from macro_regime_classifier.features.engineering import build_features


def test_hmm_predict() -> None:
    panel = generate_cross_asset_panel(n_days=300)
    feats = build_features(panel)
    det = HMMRegimeDetector(n_components=4).fit(feats)
    pred = det.predict(feats)
    assert len(pred) == len(feats)
    assert det.transition_matrix().shape == (4, 4)
