"""Mortgage and amortization calculations.

Pure functions only — no I/O, no side effects. This makes the math
testable and reusable across notebooks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Loan:
    """A mortgage loan."""

    principal: float
    annual_rate: float  # e.g. 0.065 for 6.5%
    term_years: int = 30

    @property
    def monthly_rate(self) -> float:
        return self.annual_rate / 12

    @property
    def n_payments(self) -> int:
        return self.term_years * 12

    @property
    def monthly_payment(self) -> float:
        """Monthly principal + interest payment."""
        r = self.monthly_rate
        n = self.n_payments
        if r == 0:
            return self.principal / n
        return self.principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)


def amortization_schedule(loan: Loan) -> pd.DataFrame:
    """Return month-by-month amortization as a DataFrame.

    Columns: month, payment, interest, principal, balance, cumulative_interest,
             cumulative_principal
    """
    r = loan.monthly_rate
    pmt = loan.monthly_payment
    balance = loan.principal

    rows = []
    cum_interest = 0.0
    cum_principal = 0.0
    for m in range(1, loan.n_payments + 1):
        interest = balance * r
        principal = pmt - interest
        balance -= principal
        cum_interest += interest
        cum_principal += principal
        rows.append(
            {
                "month": m,
                "payment": pmt,
                "interest": interest,
                "principal": principal,
                "balance": max(balance, 0.0),
                "cumulative_interest": cum_interest,
                "cumulative_principal": cum_principal,
            }
        )
    return pd.DataFrame(rows)


def balance_after_months(loan: Loan, months: int) -> float:
    """Outstanding balance after N months — fast closed-form."""
    r = loan.monthly_rate
    n = loan.n_payments
    pmt = loan.monthly_payment
    if r == 0:
        return max(loan.principal - pmt * months, 0.0)
    bal = loan.principal * (1 + r) ** months - pmt * (((1 + r) ** months - 1) / r)
    return max(bal, 0.0)


def pmi_monthly(loan_amount: float, home_price: float, annual_pmi_rate: float = 0.007) -> float:
    """PMI cost when down payment is under 20%.

    Default rate of 0.7%/yr is a rough mid-range estimate.
    PMI typically drops off when LTV reaches 80% (i.e. 20% equity).
    """
    ltv = loan_amount / home_price
    if ltv <= 0.80:
        return 0.0
    return loan_amount * annual_pmi_rate / 12


def months_until_pmi_drops(loan: Loan, home_price: float) -> int | None:
    """How many months until LTV reaches 80% via principal paydown alone.

    Returns None if loan is already at or below 80% LTV.
    Ignores property appreciation (which would let you request earlier removal).
    """
    target_balance = home_price * 0.80
    if loan.principal <= target_balance:
        return None
    for m in range(1, loan.n_payments + 1):
        if balance_after_months(loan, m) <= target_balance:
            return m
    return None
