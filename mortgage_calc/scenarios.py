"""Rent-vs-buy scenario engine.

Compares buying a property (with eventual sale) against renting an
equivalent unit over the same period. Returns full breakdowns so
notebooks can dissect any component.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from .mortgage import Loan, balance_after_months, pmi_monthly


@dataclass
class Property:
    """A property under evaluation."""

    name: str
    price: float
    monthly_hoa: float
    monthly_property_tax: float
    monthly_insurance: float = 50.0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BuyAssumptions:
    """Inputs governing the buy scenario."""

    down_payment_pct: float = 0.10       # 10% down
    mortgage_rate: float = 0.065         # 6.5%
    mortgage_term_years: int = 30
    closing_cost_pct: float = 0.025      # 2.5% of price
    selling_cost_pct: float = 0.06       # 6% of sale price
    annual_appreciation: float = 0.03    # 3%/yr
    annual_hoa_increase: float = 0.04    # 4%/yr — HOAs creep
    annual_tax_increase: float = 0.03    # 3%/yr
    pmi_annual_rate: float = 0.007       # 0.7%/yr if under 20% down
    annual_maintenance_pct: float = 0.01 # 1% of home value/yr


@dataclass
class RentAssumptions:
    """Inputs governing the rent scenario."""

    monthly_rent: float
    annual_rent_increase: float = 0.03   # 3%/yr
    renters_insurance_monthly: float = 20.0


@dataclass
class FinancialContext:
    """Outside-the-property assumptions."""

    opportunity_cost_rate: float = 0.07   # what cash earns elsewhere
    marginal_tax_rate: float = 0.24       # for mortgage interest deduction
    use_tax_benefit: bool = False         # conservative default: off
    hold_period_years: int = 5


@dataclass
class ScenarioResult:
    """Output of a rent-vs-buy scenario."""

    # Buy side
    cash_invested: float
    monthly_payment_year_1: float
    total_paid_owning: float
    sale_price: float
    selling_costs: float
    mortgage_payoff: float
    net_cash_from_sale: float
    profit_on_sale: float
    cash_on_cash_return_pct: float
    principal_paid: float
    interest_paid: float
    maintenance_paid: float

    # Rent side
    total_rent_paid: float

    # Comparison
    monthly_savings_vs_rent: float  # how much owning cost less per month avg
    opportunity_cost_of_cash: float  # forgone returns on cash_invested
    tax_benefit: float
    true_economic_gain: float        # the headline rent-vs-buy number

    # Per-year detail for charting / inspection
    yearly_detail: list[dict[str, Any]] = field(default_factory=list)


def run_scenario(
    prop: Property,
    buy: BuyAssumptions,
    rent: RentAssumptions,
    ctx: FinancialContext,
) -> ScenarioResult:
    """Run a single rent-vs-buy scenario and return a full breakdown."""
    # --- Setup ---
    down_payment = prop.price * buy.down_payment_pct
    closing_costs = prop.price * buy.closing_cost_pct
    cash_invested = down_payment + closing_costs

    loan_amount = prop.price - down_payment
    loan = Loan(
        principal=loan_amount,
        annual_rate=buy.mortgage_rate,
        term_years=buy.mortgage_term_years,
    )

    hold_months = ctx.hold_period_years * 12

    # --- Run month by month so HOA, taxes, rent, PMI all evolve ---
    hoa = prop.monthly_hoa
    tax = prop.monthly_property_tax
    ins = prop.monthly_insurance
    current_rent = rent.monthly_rent

    total_paid_owning = 0.0
    total_rent_paid = 0.0
    total_interest = 0.0
    total_principal = 0.0
    total_tax_paid_yearly = 0.0  # for tax-benefit calc
    yearly_detail = []

    balance = loan_amount
    r = loan.monthly_rate
    pmt = loan.monthly_payment

    for m in range(1, hold_months + 1):
        # Monthly mortgage breakdown
        interest = balance * r
        principal = pmt - interest
        balance -= principal
        total_interest += interest
        total_principal += principal

        # PMI (recalculated as balance shrinks)
        pmi = pmi_monthly(balance, prop.price, buy.pmi_annual_rate)
        maintenance = prop.price * buy.annual_maintenance_pct / 12

        own_monthly = pmt + hoa + tax + ins + pmi + maintenance
        total_paid_owning += own_monthly

        # Renter pays this month
        total_rent_paid += current_rent + rent.renters_insurance_monthly

        # End-of-year escalations
        if m % 12 == 0:
            year = m // 12
            yearly_detail.append(
                {
                    "year": year,
                    "balance_end": balance,
                    "interest_ytd": total_interest,
                    "principal_ytd": total_principal,
                    "own_monthly_avg": own_monthly,
                    "rent_monthly": current_rent,
                    "hoa_monthly": hoa,
                    "tax_monthly": tax,
                }
            )
            hoa *= 1 + buy.annual_hoa_increase
            tax *= 1 + buy.annual_tax_increase
            current_rent *= 1 + rent.annual_rent_increase

    # --- Sale ---
    sale_price = prop.price * (1 + buy.annual_appreciation) ** ctx.hold_period_years
    selling_costs = sale_price * buy.selling_cost_pct
    mortgage_payoff = balance
    net_cash_from_sale = sale_price - selling_costs - mortgage_payoff
    profit_on_sale = net_cash_from_sale - cash_invested
    cash_on_cash = (profit_on_sale / cash_invested) * 100 if cash_invested else 0.0

    # --- Comparison ---
    monthly_savings = (total_rent_paid - total_paid_owning) / hold_months
    opportunity_cost = cash_invested * (
        (1 + ctx.opportunity_cost_rate) ** ctx.hold_period_years - 1
    )

    # Tax benefit: mortgage interest deduction (very rough — assumes itemizing)
    tax_benefit = 0.0
    if ctx.use_tax_benefit:
        tax_benefit = total_interest * ctx.marginal_tax_rate

    true_gain = (
        profit_on_sale
        + (total_rent_paid - total_paid_owning)  # rent savings
        - opportunity_cost
        + tax_benefit
    )

    return ScenarioResult(
        cash_invested=cash_invested,
        monthly_payment_year_1=pmt + prop.monthly_hoa + prop.monthly_property_tax
        + prop.monthly_insurance + pmi_monthly(loan_amount, prop.price, buy.pmi_annual_rate)
        + prop.price * buy.annual_maintenance_pct / 12,
        total_paid_owning=total_paid_owning,
        sale_price=sale_price,
        selling_costs=selling_costs,
        mortgage_payoff=mortgage_payoff,
        net_cash_from_sale=net_cash_from_sale,
        profit_on_sale=profit_on_sale,
        cash_on_cash_return_pct=cash_on_cash,
        principal_paid=total_principal,
        interest_paid=total_interest,
        maintenance_paid=prop.price * buy.annual_maintenance_pct * ctx.hold_period_years,
        total_rent_paid=total_rent_paid,
        monthly_savings_vs_rent=monthly_savings,
        opportunity_cost_of_cash=opportunity_cost,
        tax_benefit=tax_benefit,
        true_economic_gain=true_gain,
        yearly_detail=yearly_detail,
    )
