"""Rent-out scenario: instead of selling, keep the property and lease it.

Models cash flow as a landlord. Useful for evaluating your "what if we
relocate in 2-3 years" path.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from .mortgage import Loan, pmi_monthly
from .scenarios import BuyAssumptions, Property


@dataclass
class RentOutAssumptions:
    """How a rental performs as an investment."""

    expected_rent: float                 # what you'd lease it for
    annual_rent_growth: float = 0.03
    vacancy_rate: float = 0.08           # 8% vacancy/turnover loss
    property_mgmt_pct: float = 0.10      # 10% of collected rent (if remote)
    maintenance_annual_pct: float = 0.01 # 1% of home value/yr
    landlord_insurance_uplift: float = 0.25  # 25% more than HO-6


@dataclass
class RentOutYearly:
    year: int
    gross_rent: float
    vacancy_loss: float
    effective_rent: float
    mortgage_pi: float
    hoa: float
    property_tax: float
    insurance: float
    pmi: float
    property_mgmt: float
    maintenance: float
    net_cash_flow: float
    principal_paid: float


@dataclass
class RentOutResult:
    yearly: list[RentOutYearly] = field(default_factory=list)
    cumulative_cash_flow: float = 0.0
    cumulative_principal_paid: float = 0.0
    final_balance: float = 0.0
    final_property_value: float = 0.0


def run_rent_out(
    prop: Property,
    buy: BuyAssumptions,
    rent_out: RentOutAssumptions,
    *,
    years: int,
) -> RentOutResult:
    """Model years of holding the property as a rental."""
    down_payment = prop.price * buy.down_payment_pct
    loan_amount = prop.price - down_payment
    loan = Loan(
        principal=loan_amount,
        annual_rate=buy.mortgage_rate,
        term_years=buy.mortgage_term_years,
    )

    balance = loan_amount
    r = loan.monthly_rate
    pmt = loan.monthly_payment

    hoa = prop.monthly_hoa
    tax = prop.monthly_property_tax
    base_ins = prop.monthly_insurance * (1 + rent_out.landlord_insurance_uplift)
    current_rent = rent_out.expected_rent

    result = RentOutResult()
    total_cf = 0.0
    total_principal = 0.0

    for y in range(1, years + 1):
        gross_rent = current_rent * 12
        vacancy_loss = gross_rent * rent_out.vacancy_rate
        effective_rent = gross_rent - vacancy_loss
        mgmt_fee = effective_rent * rent_out.property_mgmt_pct

        # Walk through 12 months to get accurate principal/interest split + PMI
        year_pi = 0.0
        year_principal = 0.0
        year_pmi = 0.0
        for _m in range(12):
            interest = balance * r
            principal = pmt - interest
            balance -= principal
            year_pi += pmt
            year_principal += principal
            year_pmi += pmi_monthly(balance, prop.price, buy.pmi_annual_rate)

        year_hoa = hoa * 12
        year_tax = tax * 12
        year_ins = base_ins * 12
        year_maint = prop.price * rent_out.maintenance_annual_pct

        net_cf = (
            effective_rent
            - year_pi
            - year_hoa
            - year_tax
            - year_ins
            - year_pmi
            - mgmt_fee
            - year_maint
        )

        result.yearly.append(
            RentOutYearly(
                year=y,
                gross_rent=gross_rent,
                vacancy_loss=vacancy_loss,
                effective_rent=effective_rent,
                mortgage_pi=year_pi,
                hoa=year_hoa,
                property_tax=year_tax,
                insurance=year_ins,
                pmi=year_pmi,
                property_mgmt=mgmt_fee,
                maintenance=year_maint,
                net_cash_flow=net_cf,
                principal_paid=year_principal,
            )
        )

        total_cf += net_cf
        total_principal += year_principal

        # Annual escalations
        hoa *= 1 + buy.annual_hoa_increase
        tax *= 1 + buy.annual_tax_increase
        base_ins *= 1.03
        current_rent *= 1 + rent_out.annual_rent_growth

    result.cumulative_cash_flow = total_cf
    result.cumulative_principal_paid = total_principal
    result.final_balance = balance
    result.final_property_value = prop.price * (1 + buy.annual_appreciation) ** years
    return result
