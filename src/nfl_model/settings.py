"""Operating settings.

Defaults match the decisions locked on 2026-09-26, including 2- and 3-leg
parlays. Other straight-bet staking methods stay selectable.
"""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass


class FairPriceSource(enum.Enum):
    MODEL = "model"
    MARKET = "market"
    BOTH = "both"


class StakingMethod(enum.Enum):
    POLICY = "policy"
    FLAT = "flat"
    UNITS = "units"
    FULL = "full"
    HALF = "half"
    QUARTER = "quarter"


class BankrollMode(enum.Enum):
    SHARED = "shared"
    SLEEVES = "sleeves"


class SettingsError(ValueError):
    """A flag or environment value cannot be used."""


class MissingBankrollError(SettingsError):
    """A live recommendation needs a dollar bankroll."""


# Locked 2026-09-26. `--staking` can still select another method.
DEFAULT_FAIR_PRICE = FairPriceSource.BOTH
DEFAULT_STAKING = StakingMethod.POLICY
DEFAULT_FLAT_FRACTION = 0.01
DEFAULT_BANKROLL_MODE = BankrollMode.SLEEVES
DEFAULT_SLEEVE_SIDES = 0.65
DEFAULT_SLEEVE_PROPS = 0.20
DEFAULT_SLEEVE_LIVE = 0.10
DEFAULT_SLEEVE_PARLAYS = 0.05
DEFAULT_MIN_EDGE = 0.02
DEFAULT_MIN_EV = 0.0
DEFAULT_MAX_STAKE_FRACTION = 0.05
SIMULATION_BANKROLL = 1000.0

FAMILIES = ("sides", "props", "live", "parlays")


@dataclass(frozen=True)
class Settings:
    fair_price: FairPriceSource = DEFAULT_FAIR_PRICE
    staking: StakingMethod = DEFAULT_STAKING
    bankroll_mode: BankrollMode = DEFAULT_BANKROLL_MODE
    bankroll: float | None = None
    flat_fraction: float = DEFAULT_FLAT_FRACTION
    unit_size: float | None = None
    max_stake_fraction: float = DEFAULT_MAX_STAKE_FRACTION
    min_edge: float = DEFAULT_MIN_EDGE
    min_ev: float = DEFAULT_MIN_EV
    sleeve_sides: float = DEFAULT_SLEEVE_SIDES
    sleeve_props: float = DEFAULT_SLEEVE_PROPS
    sleeve_live: float = DEFAULT_SLEEVE_LIVE
    sleeve_parlays: float = DEFAULT_SLEEVE_PARLAYS

    def __post_init__(self) -> None:
        if self.bankroll is not None and self.bankroll <= 0:
            raise SettingsError("Bankroll must be a positive number of dollars")
        if not 0 < self.flat_fraction <= 1:
            raise SettingsError("--flat-fraction must be between 0 and 1 (1% is 0.01)")
        if not 0 < self.max_stake_fraction <= 1:
            raise SettingsError("--max-stake-fraction must be between 0 and 1")
        if self.min_edge < 0:
            raise SettingsError("--min-edge cannot be negative. Use 0 to drop the edge hurdle.")
        if self.min_ev < 0:
            raise SettingsError("--min-ev cannot be negative")
        if self.unit_size is not None and self.unit_size <= 0:
            raise SettingsError("--unit-size must be a positive dollar amount")
        total = self.sleeve_sides + self.sleeve_props + self.sleeve_live + self.sleeve_parlays
        if abs(total - 1.0) > 1e-6:
            raise SettingsError(
                "Sleeve fractions must sum to 1. "
                f"Got sides={self.sleeve_sides:.4f}, props={self.sleeve_props:.4f}, "
                f"live={self.sleeve_live:.4f}, parlays={self.sleeve_parlays:.4f}."
            )
        if min(self.sleeve_sides, self.sleeve_props, self.sleeve_live, self.sleeve_parlays) < 0:
            raise SettingsError("Sleeve fractions cannot be negative")

    def pool(self, family: str) -> float:
        """Dollars this family is allowed to draw from."""
        if self.bankroll is None:
            raise MissingBankrollError(_bankroll_help())
        if family not in FAMILIES:
            raise SettingsError(f"Unknown bet family {family}")
        if self.bankroll_mode is BankrollMode.SHARED:
            return self.bankroll
        fraction = {
            "sides": self.sleeve_sides,
            "props": self.sleeve_props,
            "live": self.sleeve_live,
            "parlays": self.sleeve_parlays,
        }[family]
        return self.bankroll * fraction

    def describe(self) -> str:
        staking = _staking_phrase(self)
        if self.bankroll is None:
            bank = "bankroll not set"
        elif self.bankroll_mode is BankrollMode.SHARED:
            bank = f"one shared bankroll ${self.bankroll:,.2f}"
        else:
            bank = (
                "separate sleeves "
                f"sides ${self.pool('sides'):,.2f} ({self.sleeve_sides:.0%}), "
                f"props ${self.pool('props'):,.2f} ({self.sleeve_props:.0%}), "
                f"live ${self.pool('live'):,.2f} ({self.sleeve_live:.0%}), "
                f"parlays ${self.pool('parlays'):,.2f} ({self.sleeve_parlays:.0%})"
            )
        return "\n".join(
            [
                "Defaults locked 2026-09-26.",
                f"  fair price: {self.fair_price.value} (stake only when model and market are both +EV; a single signal is a lean)",
                f"  staking: {staking}",
                "  parlays: 2 or 3 legs, flat 1% of the parlay sleeve, no quarter-Kelly upgrade",
                f"  bankroll: {bank}",
                f"  edge threshold: {self.min_edge:.1%}   minimum EV: {self.min_ev:.1%} (settings, not a fixed cutoff)",
                f"  stake cap: {self.max_stake_fraction:.1%} of the sleeve the bet draws from",
                "Spreads and moneylines share a closing-line record. Totals keep their own. Props keep their own.",
                "Live stays flat. Other straight staking methods: --staking flat|units|full|half|quarter.",
            ]
        )


def _staking_phrase(settings: Settings) -> str:
    if settings.staking is StakingMethod.POLICY:
        return (
            f"flat {settings.flat_fraction:.0%} of the sleeve until that record qualifies, "
            f"then quarter Kelly capped at {settings.max_stake_fraction:.0%}. "
            "Live and parlays stay flat."
        )
    if settings.staking is StakingMethod.FLAT:
        return f"flat {settings.flat_fraction:.1%} of the applicable bankroll"
    if settings.staking is StakingMethod.UNITS:
        if settings.unit_size is None:
            return f"one unit, sized at {settings.flat_fraction:.1%} of the applicable bankroll until --unit-size is set"
        return f"one unit of ${settings.unit_size:,.2f}"
    names = {
        StakingMethod.FULL: "full Kelly",
        StakingMethod.HALF: "half Kelly",
        StakingMethod.QUARTER: "quarter Kelly",
    }
    return names[settings.staking]


def _bankroll_help() -> str:
    return (
        "A dollar bankroll is required so stakes are in dollars.\n"
        "  python -m nfl_model board --bankroll 1000\n"
        "or:\n"
        "  export NFL_BANKROLL=1000\n"
        "Use your real bankroll. The figure is only used to size a suggestion."
    )


def _env_str(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return raw.strip().lower()


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise SettingsError(f"{name} must be a number, got {raw!r}") from exc


def _enum_value(enum_cls: type[enum.Enum], raw: str | None, label: str):
    if raw is None:
        return None
    try:
        return enum_cls(raw)
    except ValueError as exc:
        choices = ", ".join(item.value for item in enum_cls)
        raise SettingsError(f"{label} must be one of: {choices}. Got {raw!r}.") from exc


def load_settings(
    *,
    fair_price: str | None = None,
    staking: str | None = None,
    bankroll_mode: str | None = None,
    bankroll: float | None = None,
    flat_fraction: float | None = None,
    unit_size: float | None = None,
    max_stake_fraction: float | None = None,
    min_edge: float | None = None,
    min_ev: float | None = None,
    sleeve_sides: float | None = None,
    sleeve_props: float | None = None,
    sleeve_live: float | None = None,
    sleeve_parlays: float | None = None,
    bankroll_fallback: float | None = None,
) -> Settings:
    """CLI values win, then environment variables, then the locked defaults."""
    resolved_bankroll = bankroll
    if resolved_bankroll is None:
        resolved_bankroll = _env_float("NFL_BANKROLL")
    if resolved_bankroll is None:
        resolved_bankroll = bankroll_fallback
    return Settings(
        fair_price=_enum_value(FairPriceSource, fair_price, "--fair-price")
        or _enum_value(FairPriceSource, _env_str("NFL_FAIR_PRICE"), "NFL_FAIR_PRICE")
        or DEFAULT_FAIR_PRICE,
        staking=_enum_value(StakingMethod, staking, "--staking")
        or _enum_value(StakingMethod, _env_str("NFL_STAKING"), "NFL_STAKING")
        or DEFAULT_STAKING,
        bankroll_mode=_enum_value(BankrollMode, bankroll_mode, "--bankroll-mode")
        or _enum_value(BankrollMode, _env_str("NFL_BANKROLL_MODE"), "NFL_BANKROLL_MODE")
        or DEFAULT_BANKROLL_MODE,
        bankroll=resolved_bankroll,
        flat_fraction=flat_fraction if flat_fraction is not None else (
            _env_float("NFL_FLAT_FRACTION") if _env_float("NFL_FLAT_FRACTION") is not None else DEFAULT_FLAT_FRACTION
        ),
        unit_size=unit_size if unit_size is not None else _env_float("NFL_UNIT_SIZE"),
        max_stake_fraction=max_stake_fraction if max_stake_fraction is not None else (
            _env_float("NFL_MAX_STAKE_FRACTION")
            if _env_float("NFL_MAX_STAKE_FRACTION") is not None
            else DEFAULT_MAX_STAKE_FRACTION
        ),
        min_edge=min_edge if min_edge is not None else (
            _env_float("NFL_MIN_EDGE") if _env_float("NFL_MIN_EDGE") is not None else DEFAULT_MIN_EDGE
        ),
        min_ev=min_ev if min_ev is not None else (
            _env_float("NFL_MIN_EV") if _env_float("NFL_MIN_EV") is not None else DEFAULT_MIN_EV
        ),
        sleeve_sides=sleeve_sides if sleeve_sides is not None else (
            _env_float("NFL_SLEEVE_SIDES") if _env_float("NFL_SLEEVE_SIDES") is not None else DEFAULT_SLEEVE_SIDES
        ),
        sleeve_props=sleeve_props if sleeve_props is not None else (
            _env_float("NFL_SLEEVE_PROPS") if _env_float("NFL_SLEEVE_PROPS") is not None else DEFAULT_SLEEVE_PROPS
        ),
        sleeve_live=sleeve_live if sleeve_live is not None else (
            _env_float("NFL_SLEEVE_LIVE") if _env_float("NFL_SLEEVE_LIVE") is not None else DEFAULT_SLEEVE_LIVE
        ),
        sleeve_parlays=sleeve_parlays if sleeve_parlays is not None else (
            _env_float("NFL_SLEEVE_PARLAYS")
            if _env_float("NFL_SLEEVE_PARLAYS") is not None
            else DEFAULT_SLEEVE_PARLAYS
        ),
    )
