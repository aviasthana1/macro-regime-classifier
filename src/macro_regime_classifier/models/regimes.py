"""Macro regime taxonomy."""

from enum import IntEnum


class MacroRegime(IntEnum):
    RISK_ON = 0
    RISK_OFF = 1
    REFLATION = 2
    SLOWDOWN = 3


REGIME_LABELS = {
    MacroRegime.RISK_ON: "risk_on",
    MacroRegime.RISK_OFF: "risk_off",
    MacroRegime.REFLATION: "reflation",
    MacroRegime.SLOWDOWN: "slowdown",
}
