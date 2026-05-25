"""Sensitivity analyses — sweep one or more inputs and watch the output move."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Iterable

import numpy as np
import pandas as pd

from .scenarios import (
    BuyAssumptions,
    FinancialContext,
    Property,
    RentAssumptions,
    run_scenario,
)


def sweep_1d(
    prop: Property,
    buy: BuyAssumptions,
    rent: RentAssumptions,
    ctx: FinancialContext,
    *,
    param: str,
    values: Iterable[float],
    target: str = "true_economic_gain",
) -> pd.DataFrame:
    """Vary one parameter, hold everything else constant.

    Parameters
    ----------
    param: dotted path, e.g. "buy.annual_appreciation" or "ctx.hold_period_years"
    values: iterable of values to try
    target: ScenarioResult attribute to return for each value
    """
    section, field_name = param.split(".")
    rows = []
    for v in values:
        b, r, c = buy, rent, ctx
        if section == "buy":
            b = replace(buy, **{field_name: v})
        elif section == "rent":
            r = replace(rent, **{field_name: v})
        elif section == "ctx":
            c = replace(ctx, **{field_name: v})
        else:
            raise ValueError(f"Unknown section: {section}")
        result = run_scenario(prop, b, r, c)
        rows.append({param: v, target: getattr(result, target)})
    return pd.DataFrame(rows)


def sweep_2d(
    prop: Property,
    buy: BuyAssumptions,
    rent: RentAssumptions,
    ctx: FinancialContext,
    *,
    param_x: str,
    values_x: Iterable[float],
    param_y: str,
    values_y: Iterable[float],
    target: str = "true_economic_gain",
) -> pd.DataFrame:
    """Sweep two parameters — returns a pivoted DataFrame ready for heatmap."""
    sx, fx = param_x.split(".")
    sy, fy = param_y.split(".")
    rows = []
    for x in values_x:
        for y in values_y:
            b, r, c = buy, rent, ctx
            for section, field_name, value in [(sx, fx, x), (sy, fy, y)]:
                if section == "buy":
                    b = replace(b, **{field_name: value})
                elif section == "rent":
                    r = replace(r, **{field_name: value})
                elif section == "ctx":
                    c = replace(c, **{field_name: value})
            result = run_scenario(prop, b, r, c)
            rows.append({param_x: x, param_y: y, target: getattr(result, target)})
    df = pd.DataFrame(rows)
    return df.pivot(index=param_y, columns=param_x, values=target)


def tornado(
    prop: Property,
    buy: BuyAssumptions,
    rent: RentAssumptions,
    ctx: FinancialContext,
    *,
    perturbations: dict[str, tuple[float, float]] | None = None,
    target: str = "true_economic_gain",
) -> pd.DataFrame:
    """One-at-a-time sensitivity. Returns swing magnitudes sorted descending.

    perturbations: dict mapping dotted-path param to (low, high) values.
    """
    if perturbations is None:
        # Reasonable defaults for a Chicago condo
        perturbations = {
            "buy.annual_appreciation": (0.00, 0.06),
            "buy.mortgage_rate": (0.055, 0.080),
            "ctx.hold_period_years": (3, 10),
            "rent.monthly_rent": (rent.monthly_rent * 0.85, rent.monthly_rent * 1.15),
            "buy.annual_hoa_increase": (0.00, 0.07),
            "ctx.opportunity_cost_rate": (0.04, 0.10),
        }

    baseline = run_scenario(prop, buy, rent, ctx)
    baseline_val = getattr(baseline, target)

    rows = []
    for param, (lo, hi) in perturbations.items():
        section, field_name = param.split(".")
        low_b, low_r, low_c = buy, rent, ctx
        high_b, high_r, high_c = buy, rent, ctx
        if section == "buy":
            low_b = replace(buy, **{field_name: lo})
            high_b = replace(buy, **{field_name: hi})
        elif section == "rent":
            low_r = replace(rent, **{field_name: lo})
            high_r = replace(rent, **{field_name: hi})
        elif section == "ctx":
            low_c = replace(ctx, **{field_name: lo})
            high_c = replace(ctx, **{field_name: hi})

        low_val = getattr(run_scenario(prop, low_b, low_r, low_c), target)
        high_val = getattr(run_scenario(prop, high_b, high_r, high_c), target)

        rows.append(
            {
                "param": param,
                "low_input": lo,
                "high_input": hi,
                "low_output": low_val,
                "high_output": high_val,
                "baseline_output": baseline_val,
                "swing": abs(high_val - low_val),
            }
        )

    df = pd.DataFrame(rows).sort_values("swing", ascending=False).reset_index(drop=True)
    return df


def breakeven_appreciation(
    prop: Property,
    buy: BuyAssumptions,
    rent: RentAssumptions,
    ctx: FinancialContext,
    *,
    lo: float = -0.05,
    hi: float = 0.15,
    tol: float = 0.0005,
) -> float | None:
    """Find the annual appreciation rate at which buying breaks even vs. renting.

    Returns the rate, or None if no breakeven exists in the range.
    """

    def gain_at(rate: float) -> float:
        b = replace(buy, annual_appreciation=rate)
        return run_scenario(prop, b, rent, ctx).true_economic_gain

    g_lo, g_hi = gain_at(lo), gain_at(hi)
    if g_lo * g_hi > 0:
        return None  # same sign throughout — no crossing

    while hi - lo > tol:
        mid = (lo + hi) / 2
        g_mid = gain_at(mid)
        if g_mid * g_lo < 0:
            hi, g_hi = mid, g_mid
        else:
            lo, g_lo = mid, g_mid
    return (lo + hi) / 2


def breakeven_hold_period(
    prop: Property,
    buy: BuyAssumptions,
    rent: RentAssumptions,
    ctx: FinancialContext,
    *,
    max_years: int = 20,
) -> int | None:
    """Find the minimum hold period (years) at which buying beats renting."""
    for y in range(1, max_years + 1):
        c = replace(ctx, hold_period_years=y)
        if run_scenario(prop, buy, rent, c).true_economic_gain > 0:
            return y
    return None
