"""End-to-end macro regime pipeline example."""

from macro_regime_classifier.data.synthetic import generate_cross_asset_panel
from macro_regime_classifier.ensemble.predictor import RegimePredictor
from macro_regime_classifier.evaluation.walk_forward import walk_forward_eval


def main() -> None:
    panel = generate_cross_asset_panel(n_days=900)
    model = RegimePredictor().fit(panel)
    signals = model.predict(panel)
    print(signals.tail())
    folds = walk_forward_eval(panel)
    print(f"Walk-forward folds: {len(folds)}, mean acc: {sum(f.accuracy for f in folds)/len(folds):.3f}")


if __name__ == "__main__":
    main()
