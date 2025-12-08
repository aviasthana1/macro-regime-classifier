from macro_regime_classifier.data.synthetic import generate_cross_asset_panel
from macro_regime_classifier.features.engineering import FEATURE_COLUMNS, build_features


def test_build_features_shape() -> None:
    panel = generate_cross_asset_panel(n_days=200)
    feats = build_features(panel)
    assert len(feats) > 100
    assert list(feats.columns) == FEATURE_COLUMNS
