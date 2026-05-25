"""Sanity tests on the calculator math.

Run with: pytest
"""
import pytest

from mortgage_calc import (
    BuyAssumptions,
    FinancialContext,
    Loan,
    Property,
    RentAssumptions,
    amortization_schedule,
    balance_after_months,
    breakeven_appreciation,
    pmi_monthly,
    run_scenario,
)


def test_loan_payment_known_value():
    """$200K at 6% over 30 years = $1,199.10/mo (well-known reference)."""
    loan = Loan(principal=200_000, annual_rate=0.06, term_years=30)
    assert loan.monthly_payment == pytest.approx(1199.10, abs=0.5)


def test_zero_rate_loan():
    """Zero-interest loan: payment = principal / n."""
    loan = Loan(principal=120_000, annual_rate=0.0, term_years=10)
    assert loan.monthly_payment == pytest.approx(1000.0)


def test_amortization_pays_off_loan():
    """Sum of all principal payments should equal original principal."""
    loan = Loan(principal=360_000, annual_rate=0.065, term_years=30)
    schedule = amortization_schedule(loan)
    assert schedule["principal"].sum() == pytest.approx(360_000, abs=1)
    assert schedule.iloc[-1]["balance"] == pytest.approx(0, abs=1)


def test_balance_after_months_matches_schedule():
    """Closed-form balance should match iterative schedule."""
    loan = Loan(principal=360_000, annual_rate=0.065, term_years=30)
    schedule = amortization_schedule(loan)
    for m in [12, 60, 120, 240]:
        expected = schedule.iloc[m - 1]["balance"]
        actual = balance_after_months(loan, m)
        assert actual == pytest.approx(expected, abs=1.0)


def test_pmi_zero_when_20_pct_down():
    """No PMI at exactly 80% LTV."""
    assert pmi_monthly(loan_amount=320_000, home_price=400_000) == 0.0


def test_pmi_positive_when_under_20_down():
    """PMI applies under 80% LTV."""
    assert pmi_monthly(loan_amount=360_000, home_price=400_000) > 0


def test_scenario_smoke():
    """Default scenario runs without error and produces sane outputs."""
    prop = Property(name="Test", price=400_000, monthly_hoa=500, monthly_property_tax=700)
    buy = BuyAssumptions()
    rent = RentAssumptions(monthly_rent=4000)
    ctx = FinancialContext()
    result = run_scenario(prop, buy, rent, ctx)

    assert result.cash_invested > 0
    assert result.sale_price > prop.price  # default 3% appreciation
    assert len(result.yearly_detail) == ctx.hold_period_years


def test_higher_appreciation_increases_gain():
    """Monotonic: more appreciation = more economic gain."""
    prop = Property(name="Test", price=400_000, monthly_hoa=500, monthly_property_tax=700)
    rent = RentAssumptions(monthly_rent=4000)
    ctx = FinancialContext()

    gains = []
    for app in [0.0, 0.02, 0.04, 0.06]:
        buy = BuyAssumptions(annual_appreciation=app)
        gains.append(run_scenario(prop, buy, rent, ctx).true_economic_gain)
    assert gains == sorted(gains)  # monotonically increasing


def test_breakeven_appreciation_exists_for_typical_case():
    """Should find a breakeven rate between -5% and +15% for a reasonable property."""
    prop = Property(name="Test", price=400_000, monthly_hoa=500, monthly_property_tax=700)
    buy = BuyAssumptions()
    rent = RentAssumptions(monthly_rent=4000)
    ctx = FinancialContext()
    breakeven = breakeven_appreciation(prop, buy, rent, ctx)
    assert breakeven is not None
    assert -0.05 < breakeven < 0.15
