from macro_regime_classifier.data.synthetic import generate_cross_asset_panel
from macro_regime_classifier.evaluation.walk_forward import walk_forward_eval


def test_walk_forward_runs() -> None:
    panel = generate_cross_asset_panel(n_days=700)
    results = walk_forward_eval(panel, train_days=200, test_days=42)
    assert len(results) >= 1
    assert 0.0 <= results[0].accuracy <= 1.0
