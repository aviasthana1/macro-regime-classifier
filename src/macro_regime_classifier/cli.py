"""Command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from macro_regime_classifier.data.loader import load_panel
from macro_regime_classifier.ensemble.predictor import RegimePredictor
from macro_regime_classifier.evaluation.walk_forward import walk_forward_eval
from macro_regime_classifier.export.signals import export_signals


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Macro regime classifier")
    p.add_argument("--csv", type=Path, help="Optional CSV panel path")
    p.add_argument("--output", type=Path, default=Path("regime_signals.csv"))
    p.add_argument("--walk-forward", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    panel = load_panel(args.csv)
    if args.walk_forward:
        results = walk_forward_eval(panel)
        for r in results:
            print(f"fold={r.fold} acc={r.accuracy:.3f} test_end={r.test_end}")
        return
    preds = RegimePredictor().fit(panel).predict(panel)
    path = export_signals(preds, args.output)
    print(f"Wrote {len(preds)} rows to {path}")
