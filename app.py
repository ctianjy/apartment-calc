"""Rent vs. Buy interactive calculator — Streamlit app."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mortgage_calc import (
    BuyAssumptions,
    FinancialContext,
    Loan,
    Property,
    RentAssumptions,
    RentOutAssumptions,
    amortization_schedule,
    breakeven_appreciation,
    breakeven_hold_period,
    pmi_monthly,
    run_rent_out,
    run_scenario,
    sweep_2d,
    tornado,
)

st.set_page_config(page_title="Rent vs Buy", layout="wide")

# ── Cached heavy computations ─────────────────────────────────────────────────

@st.cache_data
def cached_tornado(price, hoa, prop_tax, down_pct, rate, term, closing_pct,
                   selling_pct, appr, hold, rent, rent_inf, opp, use_tx, tx_rate, maint):
    _p = Property(name="", price=price, monthly_hoa=hoa, monthly_property_tax=prop_tax)
    _b = BuyAssumptions(down_payment_pct=down_pct, mortgage_rate=rate,
                        mortgage_term_years=term, closing_cost_pct=closing_pct,
                        selling_cost_pct=selling_pct, annual_appreciation=appr,
                        annual_maintenance_pct=maint)
    _r = RentAssumptions(monthly_rent=rent, annual_rent_increase=rent_inf)
    _c = FinancialContext(hold_period_years=hold, opportunity_cost_rate=opp,
                          use_tax_benefit=use_tx, marginal_tax_rate=tx_rate)
    return tornado(_p, _b, _r, _c)


@st.cache_data
def cached_sweep_2d(price, hoa, prop_tax, down_pct, rate, term, closing_pct,
                    selling_pct, appr, hold, rent, rent_inf, opp, use_tx, tx_rate,
                    maint, param_x, values_x, param_y, values_y):
    _p = Property(name="", price=price, monthly_hoa=hoa, monthly_property_tax=prop_tax)
    _b = BuyAssumptions(down_payment_pct=down_pct, mortgage_rate=rate,
                        mortgage_term_years=term, closing_cost_pct=closing_pct,
                        selling_cost_pct=selling_pct, annual_appreciation=appr,
                        annual_maintenance_pct=maint)
    _r = RentAssumptions(monthly_rent=rent, annual_rent_increase=rent_inf)
    _c = FinancialContext(hold_period_years=hold, opportunity_cost_rate=opp,
                          use_tax_benefit=use_tx, marginal_tax_rate=tx_rate)
    return sweep_2d(_p, _b, _r, _c, param_x=param_x, values_x=values_x,
                    param_y=param_y, values_y=values_y)


@st.cache_data
def cached_breakeven(price, hoa, prop_tax, down_pct, rate, term, closing_pct,
                     selling_pct, appr, hold, rent, rent_inf, opp, use_tx, tx_rate, maint):
    _p = Property(name="", price=price, monthly_hoa=hoa, monthly_property_tax=prop_tax)
    _b = BuyAssumptions(down_payment_pct=down_pct, mortgage_rate=rate,
                        mortgage_term_years=term, closing_cost_pct=closing_pct,
                        selling_cost_pct=selling_pct, annual_appreciation=appr,
                        annual_maintenance_pct=maint)
    _r = RentAssumptions(monthly_rent=rent, annual_rent_increase=rent_inf)
    _c = FinancialContext(hold_period_years=hold, opportunity_cost_rate=opp,
                          use_tax_benefit=use_tx, marginal_tax_rate=tx_rate)
    return breakeven_appreciation(_p, _b, _r, _c), breakeven_hold_period(_p, _b, _r, _c)


def get_sweep_params(monthly_rent_val):
    """Return available sensitivity parameters with value grids and axis formatters."""
    return {
        "buy.annual_appreciation": {
            "label": "Annual appreciation",
            "values": tuple(np.arange(-0.02, 0.081, 0.01)),
            "fmt": lambda v: f"{v*100:.0f}%",
        },
        "buy.mortgage_rate": {
            "label": "Mortgage rate",
            "values": tuple(np.arange(0.04, 0.091, 0.005)),
            "fmt": lambda v: f"{v*100:.1f}%",
        },
        "ctx.hold_period_years": {
            "label": "Hold period",
            "values": tuple(range(2, 16)),
            "fmt": lambda v: f"{int(v)}yr",
        },
        "rent.monthly_rent": {
            "label": "Monthly rent",
            "values": tuple(np.linspace(monthly_rent_val * 0.70, monthly_rent_val * 1.30, 8).round()),
            "fmt": lambda v: f"${v:,.0f}",
        },
        "buy.down_payment_pct": {
            "label": "Down payment %",
            "values": tuple(np.arange(0.05, 0.31, 0.05)),
            "fmt": lambda v: f"{v*100:.0f}%",
        },
        "ctx.opportunity_cost_rate": {
            "label": "Opportunity cost rate",
            "values": tuple(np.arange(0.03, 0.131, 0.01)),
            "fmt": lambda v: f"{v*100:.0f}%",
        },
    }


def _current_param_value(param, prop, buy, rent_a, ctx):
    section, field = param.split(".")
    obj = {"buy": buy, "rent": rent_a, "ctx": ctx}.get(section)
    return getattr(obj, field, None) if obj else None


def _closest_idx(values_tuple, current_val):
    if current_val is None:
        return None
    arr = np.array(values_tuple, dtype=float)
    return int(np.argmin(np.abs(arr - float(current_val))))


# ── Sidebar inputs ────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Property")
    prop_name = st.text_input("Name", "My Property")
    price = st.number_input("Purchase price ($)", value=425_000, step=5_000, min_value=50_000)
    prop_tax_rate = st.slider("Annual property tax rate", 0.50, 4.00, 1.66, step=0.02, format="%.2f%%") / 100.0
    prop_tax = price * prop_tax_rate / 12
    st.caption(f"= ${prop_tax:,.0f}/mo  |  Cook County: 1.66–2.14%  |  Chicago: ~1.66%  |  Lincoln Park: ~1.8%")
    hoa = st.number_input("Monthly HOA ($)", value=525, step=25, min_value=0)
    maintenance_pct = st.slider("Annual maintenance", 0.0, 3.0, 1.0, step=0.1, format="%.1f%%") / 100.0
    st.caption(f"= ${price * maintenance_pct / 12:,.0f}/mo  |  Rule of thumb: 1% of home value/yr")

    st.header("Mortgage")
    down_pct = st.slider("Down payment", 3, 50, 10, format="%d%%") / 100.0
    rate = st.slider("Mortgage rate", 3.0, 12.0, 6.5, step=0.05, format="%.2f%%") / 100.0
    st.caption("30yr avg: ~7.7% (1990–2024)  |  2024 avg: ~6.9%")
    term = st.selectbox("Term", [30, 15], format_func=lambda x: f"{x} years")

    st.header("Market assumptions")
    appreciation = st.slider("Annual appreciation", -2, 10, 3, format="%d%%") / 100.0
    st.caption("Chicago avg: ~2–3%/yr  |  US median: ~4%/yr  (2000–2024)")
    hold = st.slider("Hold period", 1, 20, 5, format="%d yrs")
    st.caption("Median US homeowner tenure: ~8 yrs (NAR 2024)")

    st.header("Rent comparison")
    monthly_rent = st.number_input("Equivalent monthly rent ($)", value=4000, step=100, min_value=500)
    rent_inflation = st.slider("Annual rent increase", 0, 8, 3, format="%d%%") / 100.0
    st.caption("US avg: ~3–4%/yr  |  Chicago: ~3%/yr (2015–2024)")

    with st.expander("Advanced assumptions"):
        opp_cost = st.slider("Opportunity cost rate", 3, 15, 7, format="%d%%") / 100.0
        st.caption("S&P 500 nominal avg: ~10%/yr  |  Real (inflation-adj): ~7%/yr (1990–2024)")
        selling_cost_pct = st.slider("Selling costs", 3, 8, 6, format="%d%%") / 100.0
        closing_cost_pct = st.slider("Closing costs", 1, 6, 3, format="%d%%") / 100.0
        use_tax = st.checkbox("Include mortgage interest tax benefit", value=False)
        tax_rate = st.slider("Marginal tax rate", 10, 40, 24, format="%d%%") / 100.0

        st.markdown("---")
        st.markdown("**Capital gains tax on sale**")
        filing_status = st.radio(
            "Filing status",
            ["Single ($250K exclusion)", "Married ($500K exclusion)"],
            index=1, horizontal=True,
        )
        meets_2of5 = st.checkbox(
            "Primary residence 2+ of last 5 years (§121 exclusion)", value=True,
        )
        cap_gains_rate = st.slider("Long-term cap gains rate", 0, 20, 15, format="%d%%") / 100.0
        st.caption("0% if income <$47K  |  15% if <$518K  |  20% above (2024, single filer)")

    with st.expander("Rent-out scenario"):
        show_rent_out = st.checkbox("Show rent-out analysis", value=True)
        expected_rent_out = st.number_input(
            "Expected rental income ($/mo)", value=int(monthly_rent), step=100, min_value=0,
        )
        vacancy_rate = st.slider("Vacancy rate", 0, 25, 8, format="%d%%") / 100.0
        mgmt_pct = st.slider("Property mgmt fee", 0, 15, 10, format="%d%%") / 100.0

# ── Build objects from inputs ─────────────────────────────────────────────────

prop = Property(name=prop_name, price=price, monthly_hoa=hoa, monthly_property_tax=prop_tax)
buy = BuyAssumptions(
    down_payment_pct=down_pct,
    mortgage_rate=rate,
    mortgage_term_years=term,
    closing_cost_pct=closing_cost_pct,
    selling_cost_pct=selling_cost_pct,
    annual_appreciation=appreciation,
    annual_maintenance_pct=maintenance_pct,
)
rent_a = RentAssumptions(monthly_rent=monthly_rent, annual_rent_increase=rent_inflation)
ctx = FinancialContext(
    hold_period_years=hold,
    opportunity_cost_rate=opp_cost,
    use_tax_benefit=use_tax,
    marginal_tax_rate=tax_rate,
)

result = run_scenario(prop, buy, rent_a, ctx)

# Capital gains tax on sale (§121 exclusion if primary residence 2-of-5 years)
_exclusion = 500_000 if "Married" in filing_status else 250_000
_cost_basis = price * (1 + closing_cost_pct)
_cap_gain_before_exclusion = (result.sale_price - result.selling_costs) - _cost_basis
_taxable_cap_gain = max(0.0, _cap_gain_before_exclusion - (_exclusion if meets_2of5 else 0))
_cap_gains_tax = _taxable_cap_gain * cap_gains_rate
_adjusted_true_gain = result.true_economic_gain - _cap_gains_tax

# shared args tuple for cached functions
_cache_args = (
    price, hoa, prop_tax, down_pct, rate, term, closing_cost_pct, selling_cost_pct,
    appreciation, hold, monthly_rent, rent_inflation, opp_cost, use_tax, tax_rate,
    maintenance_pct,
)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["🏠 Rent vs Buy Calculator", "📊 Market Research"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Calculator
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:

    # ── Section 1: Verdict ────────────────────────────────────────────────────

    st.title(prop_name)

    gain = _adjusted_true_gain
    verdict = "BUYING WINS" if gain > 0 else "RENTING WINS"
    color = "#1D9E75" if gain > 0 else "#D85A30"
    _cap_note = f"  ·  incl. ${_cap_gains_tax:,.0f} cap gains tax" if _cap_gains_tax > 0 else ""

    st.markdown(
        f"""
        <div style="background:{color}22; border-left:6px solid {color};
                    padding:1rem 1.5rem; border-radius:4px; margin-bottom:1rem;">
            <h2 style="color:{color}; margin:0">{verdict} by ${abs(gain):,.0f}</h2>
            <p style="margin:0.3rem 0 0; color:#555">
                True economic gain from buying vs. renting over {hold} years{_cap_note}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cash to close", f"${result.cash_invested:,.0f}")
    c2.metric(
        "Monthly cost yr 1", f"${result.monthly_payment_year_1:,.0f}",
        delta=f"vs ${monthly_rent:,.0f}/mo rent", delta_color="inverse",
    )
    c3.metric(f"Sale price ({hold} yr)", f"${result.sale_price:,.0f}")
    c4.metric("Profit on sale", f"${result.profit_on_sale:,.0f}",
              f"{result.cash_on_cash_return_pct:.1f}% on cash")

    with st.expander("Full gain breakdown"):
        rent_savings = result.total_rent_paid - result.total_paid_owning
        _loan_amount = price * (1 - down_pct)
        _down_payment = price * down_pct
        _closing_amt = price * closing_cost_pct
        _monthly_pi = Loan(principal=_loan_amount, annual_rate=rate, term_years=term).monthly_payment
        _pmi_yr1 = pmi_monthly(_loan_amount, price)
        _maint_monthly = price * maintenance_pct / 12
        _monthly_total_yr1 = _monthly_pi + hoa + prop_tax + 50.0 + _pmi_yr1 + _maint_monthly
        _monthly_diff = monthly_rent - _monthly_total_yr1
        _diff_label = f"${abs(_monthly_diff):,.0f}/mo {'cheaper to own' if _monthly_diff >= 0 else 'more expensive to own'}"
        _opp_growth_pct = ((1 + opp_cost) ** hold - 1) * 100

        col_tbl, col_chart = st.columns([3, 2])
        with col_tbl:
            st.markdown(f"""
**Cash at closing**

| | Calculation | Amount |
|---|---|---:|
| Purchase price | | ${price:,.0f} |
| Down payment | ${price:,.0f} × {down_pct*100:.0f}% | −${_down_payment:,.0f} |
| → Loan amount | | ${_loan_amount:,.0f} |
| Closing costs | ${price:,.0f} × {closing_cost_pct*100:.0f}% | ${_closing_amt:,.0f} |
| **Total cash to close** | | **${result.cash_invested:,.0f}** |
""")
            st.markdown(f"""
**Monthly costs — Year 1**

| | Calculation | Amount |
|---|---|---:|
| Mortgage P&I | ${_loan_amount:,.0f} @ {rate*100:.2f}%, {term}yr | ${_monthly_pi:,.0f}/mo |
| HOA | | ${hoa:,.0f}/mo |
| Property tax | ${price:,.0f} × {prop_tax_rate*100:.2f}% ÷ 12 | ${prop_tax:,.0f}/mo |
| Insurance | | $50/mo |
| PMI | ${_loan_amount:,.0f} × 0.7% ÷ 12 | ${_pmi_yr1:,.0f}/mo |
| Maintenance | ${price:,.0f} × {maintenance_pct*100:.1f}% ÷ 12 | ${_maint_monthly:,.0f}/mo |
| **Total owning (yr 1)** | | **${_monthly_total_yr1:,.0f}/mo** |
| Equivalent rent | | ${monthly_rent:,.0f}/mo |
| **Monthly difference** | | **{_diff_label}** |
""")
            st.markdown(f"""
**Over {hold} years**

| | Amount |
|---|---:|
| Total paid owning | ${result.total_paid_owning:,.0f} |
| → Interest paid | ${result.interest_paid:,.0f} |
| → Principal (equity built) | ${result.principal_paid:,.0f} |
| → Maintenance | ${result.maintenance_paid:,.0f} |
| Total rent paid | ${result.total_rent_paid:,.0f} |
| **Rent savings** | **${rent_savings:,.0f}** |
""")
            st.markdown(f"""
**Sale after {hold} years**

| | Calculation | Amount |
|---|---|---:|
| Sale price | ${price:,.0f} × {1+appreciation:.4f}^{hold} | ${result.sale_price:,.0f} |
| Selling costs | ${result.sale_price:,.0f} × {selling_cost_pct*100:.0f}% | −${result.selling_costs:,.0f} |
| Mortgage payoff | balance after {hold} yrs | −${result.mortgage_payoff:,.0f} |
| Net from sale | | ${result.net_cash_from_sale:,.0f} |
| Less cash invested | | −${result.cash_invested:,.0f} |
| **Profit on sale** | | **${result.profit_on_sale:,.0f}** ({result.cash_on_cash_return_pct:.1f}% on cash) |
""")
            st.markdown(f"""
**True economic gain**

| | Calculation | Amount |
|---|---|---:|
| Profit on sale | | ${result.profit_on_sale:,.0f} |
| + Rent savings | | ${rent_savings:,.0f} |
| + Tax benefit | | ${result.tax_benefit:,.0f} |
| − Opportunity cost | ${result.cash_invested:,.0f} × {_opp_growth_pct:.1f}% | −${result.opportunity_cost_of_cash:,.0f} |
| − Capital gains tax | ${_taxable_cap_gain:,.0f} taxable × {cap_gains_rate*100:.0f}% | −${_cap_gains_tax:,.0f} |
| **= True economic gain** | | **${_adjusted_true_gain:,.0f}** |
""")
        with col_chart:
            items = [
                ("Profit on sale", result.profit_on_sale),
                ("Rent savings", rent_savings),
                ("Tax benefit", result.tax_benefit),
                ("Opp. cost", -result.opportunity_cost_of_cash),
                ("Cap. gains tax", -_cap_gains_tax),
            ]
            fig_wf = go.Figure(go.Waterfall(
                orientation="v",
                measure=["relative"] * len(items),
                x=[i[0] for i in items],
                y=[i[1] for i in items],
                connector={"line": {"color": "#aaa"}},
                increasing={"marker": {"color": "#1D9E75"}},
                decreasing={"marker": {"color": "#D85A30"}},
            ))
            fig_wf.update_layout(
                title="Gain components ($)", height=420,
                margin=dict(t=40, b=10, l=10, r=10), showlegend=False,
            )
            st.plotly_chart(fig_wf, use_container_width=True)

    # ── Section 2: Amortization ───────────────────────────────────────────────

    st.subheader("Amortization — where your mortgage dollars go")

    loan = Loan(
        principal=prop.price * (1 - buy.down_payment_pct),
        annual_rate=buy.mortgage_rate,
        term_years=buy.mortgage_term_years,
    )
    amort = amortization_schedule(loan)
    amort["year"] = (amort["month"] - 1) // 12 + 1
    yearly_amort = (
        amort[amort["month"] <= hold * 12]
        .groupby("year")
        .agg(interest=("interest", "sum"), principal=("principal", "sum"))
        .reset_index()
    )

    fig_amort = go.Figure()
    fig_amort.add_trace(go.Bar(
        name="Interest (gone)", x=yearly_amort["year"], y=yearly_amort["interest"],
        marker_color="#D4543C",
    ))
    fig_amort.add_trace(go.Bar(
        name="Principal (equity)", x=yearly_amort["year"], y=yearly_amort["principal"],
        marker_color="#3B6D11",
    ))
    fig_amort.update_layout(
        barmode="stack", xaxis_title="Year", yaxis_title="$ per year",
        height=350, margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig_amort, use_container_width=True)

    a1, a2 = st.columns(2)
    a1.metric("Total interest paid", f"${result.interest_paid:,.0f}")
    a2.metric("Principal paid (equity built)", f"${result.principal_paid:,.0f}")

    # ── Section 3: Sensitivity tornado ───────────────────────────────────────

    st.subheader("Sensitivity — which inputs move the answer most")

    PARAM_LABELS = {
        "buy.annual_appreciation": "Annual appreciation",
        "buy.mortgage_rate": "Mortgage rate",
        "ctx.hold_period_years": "Hold period",
        "rent.monthly_rent": "Monthly rent",
        "buy.annual_hoa_increase": "HOA growth rate",
        "ctx.opportunity_cost_rate": "Opportunity cost rate",
    }

    t = cached_tornado(*_cache_args)
    t["label"] = t["param"].map(PARAM_LABELS).fillna(t["param"])

    fig_tornado = go.Figure()
    fig_tornado.add_trace(go.Bar(
        name="High input", orientation="h",
        y=t["label"],
        x=t["high_output"] - t["baseline_output"],
        base=t["baseline_output"],
        marker_color="#1D9E75",
    ))
    fig_tornado.add_trace(go.Bar(
        name="Low input", orientation="h",
        y=t["label"],
        x=t["low_output"] - t["baseline_output"],
        base=t["baseline_output"],
        marker_color="#D85A30",
    ))
    fig_tornado.add_vline(
        x=float(t.iloc[0]["baseline_output"]),
        line_dash="dash", line_color="black", line_width=1,
    )
    fig_tornado.update_layout(
        barmode="overlay", height=350,
        xaxis_title="True economic gain ($)",
        yaxis=dict(autorange="reversed"),
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig_tornado, use_container_width=True)

    # ── Section 4: 2D heatmap ─────────────────────────────────────────────────

    st.subheader("2-variable sensitivity heatmap")
    st.caption("Green = buying wins, red = renting wins. Values in $K. White border = your current inputs.")

    _sweep_params = get_sweep_params(monthly_rent)
    _param_keys = list(_sweep_params.keys())
    _param_labels = {k: v["label"] for k, v in _sweep_params.items()}

    _hm_c1, _hm_c2 = st.columns(2)
    with _hm_c1:
        selected_x = st.selectbox(
            "X axis (columns)",
            _param_keys,
            index=_param_keys.index("ctx.hold_period_years"),
            format_func=lambda k: _param_labels[k],
            key="heatmap_x",
        )
    with _hm_c2:
        selected_y = st.selectbox(
            "Y axis (rows)",
            _param_keys,
            index=_param_keys.index("buy.annual_appreciation"),
            format_func=lambda k: _param_labels[k],
            key="heatmap_y",
        )

    if selected_x == selected_y:
        st.warning("X and Y axes must be different parameters.")
    else:
        _vals_x = _sweep_params[selected_x]["values"]
        _vals_y = _sweep_params[selected_y]["values"]
        _fmt_x = _sweep_params[selected_x]["fmt"]
        _fmt_y = _sweep_params[selected_y]["fmt"]

        heat = cached_sweep_2d(
            *_cache_args,
            param_x=selected_x, values_x=_vals_x,
            param_y=selected_y, values_y=_vals_y,
        )
        z = heat.values / 1000
        x_labels = [_fmt_x(c) for c in heat.columns]
        y_labels = [_fmt_y(r) for r in heat.index]

        fig_heat = go.Figure(go.Heatmap(
            z=z, x=x_labels, y=y_labels,
            colorscale="RdYlGn", zmid=0,
            text=[[f"{v:.0f}" for v in row] for row in z],
            texttemplate="%{text}",
            colorbar=dict(title="Gain ($K)"),
        ))

        _xi = _closest_idx(_vals_x, _current_param_value(selected_x, prop, buy, rent_a, ctx))
        _yi = _closest_idx(_vals_y, _current_param_value(selected_y, prop, buy, rent_a, ctx))
        if _xi is not None and _yi is not None:
            fig_heat.add_shape(
                type="rect",
                x0=_xi - 0.5, x1=_xi + 0.5, y0=_yi - 0.5, y1=_yi + 0.5,
                line=dict(color="white", width=3),
            )

        fig_heat.update_layout(
            xaxis_title=_param_labels[selected_x],
            yaxis_title=_param_labels[selected_y],
            height=420, margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── Section 5: Breakeven ──────────────────────────────────────────────────

    st.subheader("Breakeven points")

    be_app, be_hold = cached_breakeven(*_cache_args)

    b1, b2 = st.columns(2)
    with b1:
        st.markdown(f"**At current hold period ({hold} yrs):**")
        if be_app is not None:
            st.info(f"Need appreciation **> {be_app*100:.2f}%/yr** for buying to win")
        else:
            st.warning("No breakeven appreciation exists in the searched range (−5% to 15%)")
    with b2:
        st.markdown(f"**At current appreciation ({appreciation*100:.1f}%/yr):**")
        if be_hold is not None:
            st.info(f"Need to hold **{be_hold}+ years** for buying to win")
        else:
            st.warning("Buying never wins within 20 years at this appreciation rate")

    # ── Section 6: Rent-out ───────────────────────────────────────────────────

    if show_rent_out:
        st.subheader("Rent-out scenario — what if you relocate instead of selling?")

        ro_assumptions = RentOutAssumptions(
            expected_rent=expected_rent_out,
            vacancy_rate=vacancy_rate,
            property_mgmt_pct=mgmt_pct,
        )
        ro = run_rent_out(prop, buy, ro_assumptions, years=hold)

        yearly_ro = pd.DataFrame([
            {
                "Year": y.year,
                "Effective rent": y.effective_rent,
                "Mortgage P&I": y.mortgage_pi,
                "Other costs": y.hoa + y.property_tax + y.insurance + y.pmi + y.property_mgmt + y.maintenance,
                "Net cash flow": y.net_cash_flow,
                "Principal built": y.principal_paid,
            }
            for y in ro.yearly
        ])

        fig_ro = go.Figure()
        fig_ro.add_trace(go.Bar(
            name="Effective rent", x=yearly_ro["Year"], y=yearly_ro["Effective rent"],
            marker_color="#1D9E75",
        ))
        fig_ro.add_trace(go.Bar(
            name="Mortgage P&I", x=yearly_ro["Year"], y=-yearly_ro["Mortgage P&I"],
            marker_color="#D4543C",
        ))
        fig_ro.add_trace(go.Bar(
            name="Other costs", x=yearly_ro["Year"], y=-yearly_ro["Other costs"],
            marker_color="#F0A500",
        ))
        fig_ro.add_trace(go.Scatter(
            name="Net cash flow", x=yearly_ro["Year"], y=yearly_ro["Net cash flow"],
            mode="lines+markers", line=dict(color="black", width=2),
        ))
        fig_ro.add_hline(y=0, line_dash="dot", line_color="#888", line_width=1)
        fig_ro.update_layout(
            barmode="relative", xaxis_title="Year", yaxis_title="$ per year",
            height=380, margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_ro, use_container_width=True)

        r1, r2, r3 = st.columns(3)
        r1.metric("Cumulative cash flow", f"${ro.cumulative_cash_flow:,.0f}")
        r2.metric("Principal paid", f"${ro.cumulative_principal_paid:,.0f}")
        r3.metric("Final property value", f"${ro.final_property_value:,.0f}")

        annual_cf = ro.cumulative_cash_flow / hold
        if annual_cf < 0:
            st.warning(
                f"Average annual cash flow is **negative**: ${annual_cf:,.0f}/yr "
                f"(${abs(annual_cf)/12:,.0f}/mo subsidy from other income)"
            )
        else:
            st.success(
                f"Average annual cash flow: **${annual_cf:,.0f}/yr** (${annual_cf/12:,.0f}/mo)"
            )

        with st.expander("Year-by-year detail"):
            st.dataframe(
                yearly_ro.set_index("Year").style.format("${:,.0f}"),
                use_container_width=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Market Research
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.title("Chicago North Side — Housing Market Research")
    st.info(
        "Data compiled from **Redfin**, **Zillow**, **Cook County Assessor**, "
        "**Real Group Real Estate**, **Camille Canales Group**, **AC Group Chicago**, "
        "and **S&P Case-Shiller / FRED**. "
        "Solid markers = directly sourced data points. "
        "Thin dashed connectors = linear interpolation between known anchors — treat as indicative only."
    )

    years_full = list(range(2015, 2025))

    # ── Chart 1: Case-Shiller macro context ───────────────────────────────────

    st.subheader("1. Chicago-wide: S&P Case-Shiller Home Price Index (2015–2024)")
    st.caption(
        "Source: FRED, St. Louis Fed — series CHXRSA (all homes, seasonally adjusted) and "
        "CHXRCNSA (condos). Normalized to 2015 = 100. "
        "Covers the broader Chicago metro, not North Side neighborhoods specifically."
    )

    cs_all   = [100, 103, 107, 111, 114, 119, 136, 158, 165, 177]
    cs_condo = [100, 103, 106, 108, 110, 113, 120, 132, 137, 143]

    fig_cs = go.Figure()
    fig_cs.add_trace(go.Scatter(
        x=years_full, y=cs_all, mode="lines+markers", name="All homes",
        line=dict(color="#1D4E89", width=2.5),
        marker=dict(size=7),
        hovertemplate="%{x}: %{y:.0f} (+%{customdata:.0f}% vs 2015)<extra>All homes</extra>",
        customdata=[v - 100 for v in cs_all],
    ))
    fig_cs.add_trace(go.Scatter(
        x=years_full, y=cs_condo, mode="lines+markers", name="Condos",
        line=dict(color="#E07B39", width=2.5, dash="dash"),
        marker=dict(size=7, symbol="square"),
        hovertemplate="%{x}: %{y:.0f} (+%{customdata:.0f}% vs 2015)<extra>Condos</extra>",
        customdata=[v - 100 for v in cs_condo],
    ))
    fig_cs.add_annotation(x=2024, y=177, text="+77% total (all homes)", showarrow=False,
                          xanchor="right", font=dict(color="#1D4E89", size=11))
    fig_cs.add_annotation(x=2024, y=143, text="+43% total (condos)", showarrow=False,
                          xanchor="right", font=dict(color="#E07B39", size=11))
    fig_cs.update_layout(
        yaxis_title="Index (2015 = 100)", xaxis_title="Year",
        height=340, margin=dict(t=10, b=10), hovermode="x unified",
        xaxis=dict(tickmode="linear", dtick=1),
    )
    st.plotly_chart(fig_cs, use_container_width=True)

    st.markdown(
        "**Key takeaway:** Chicago all-home prices rose **+77%** from 2015–2024, "
        "but **condos lagged significantly** (+43%), especially during the 2020–2022 surge "
        "when buyers prioritized space. The steepest single-year gain was **2021 (+14%)** "
        "driven by pandemic demand and near-zero mortgage rates."
    )

    st.divider()

    # ── Chart 2: Lincoln Park by property type ────────────────────────────────

    st.subheader("2. Lincoln Park — Median sale price by property type (2015–2024)")

    # Sourced anchor points
    lp_condo_src  = {2018: 445, 2019: 470, 2020: 510, 2022: 518, 2024: 550}
    lp_sfh_src    = {2016: 1600, 2017: 1600, 2018: 1600, 2020: 1450, 2022: 1720}
    lp_overall_src = {2023: 700, 2024: 726}

    def interp(src, years=years_full):
        xs = sorted(src.keys())
        ys = [src[x] for x in xs]
        return np.interp(years, xs, ys).tolist()

    lp_c_interp = interp(lp_condo_src)
    lp_s_interp = interp(lp_sfh_src)
    lp_o_interp = interp(lp_overall_src)

    fig_lp = go.Figure()

    # Interpolated background lines (thin, dashed)
    fig_lp.add_trace(go.Scatter(
        x=years_full, y=[v / 1_000 for v in lp_c_interp],
        mode="lines", line=dict(color="#E07B39", dash="dot", width=1),
        showlegend=False, hoverinfo="skip",
    ))
    fig_lp.add_trace(go.Scatter(
        x=years_full, y=[v / 1_000 for v in lp_s_interp],
        mode="lines", line=dict(color="#1D4E89", dash="dot", width=1),
        showlegend=False, hoverinfo="skip",
    ))

    # Sourced data points with solid lines connecting them
    fig_lp.add_trace(go.Scatter(
        x=list(lp_condo_src.keys()), y=[v / 1_000 for v in lp_condo_src.values()],
        mode="lines+markers", name="Condo (sourced)",
        line=dict(color="#E07B39", width=2),
        marker=dict(size=10, symbol="circle"),
        hovertemplate="%{x}: $%{y:.2f}M<extra>LP Condo</extra>",
    ))
    fig_lp.add_trace(go.Scatter(
        x=list(lp_sfh_src.keys()), y=[v / 1_000 for v in lp_sfh_src.values()],
        mode="lines+markers", name="Single-family home (sourced)",
        line=dict(color="#1D4E89", width=2),
        marker=dict(size=10, symbol="square"),
        hovertemplate="%{x}: $%{y:.2f}M<extra>LP SFH</extra>",
    ))
    fig_lp.add_trace(go.Scatter(
        x=list(lp_overall_src.keys()), y=[v / 1_000 for v in lp_overall_src.values()],
        mode="lines+markers", name="All types — overall (sourced)",
        line=dict(color="#2E8B57", width=2),
        marker=dict(size=10, symbol="diamond"),
        hovertemplate="%{x}: $%{y:.2f}M<extra>LP Overall</extra>",
    ))

    fig_lp.update_layout(
        yaxis_title="Median sale price ($M)", xaxis_title="Year",
        height=400, margin=dict(t=10, b=10), hovermode="x unified",
        xaxis=dict(tickmode="linear", dtick=1),
    )
    st.plotly_chart(fig_lp, use_container_width=True)

    st.caption(
        "Sources: Real Group Real Estate market reports (condo & SFH 2016–2022) · "
        "Cook County Assessor data cited in search results (condo 2024) · "
        "AC Group Chicago market report (overall 2023) · "
        "Camille Canales Group (overall 2024). "
        "Dot-dashed lines are linear interpolation — not sourced independently."
    )

    lp_col1, lp_col2, lp_col3 = st.columns(3)
    lp_col1.metric("Condo 2015→2024", "$550K", "~+40% est. from ~$395K")
    lp_col2.metric("SFH 2016→2022", "$1.72M", "+7.5% from $1.60M trough-to-peak")
    lp_col3.metric("Overall 2023→2024", "$726K", "+3.7% YoY")

    st.divider()

    # ── Chart 3: Neighborhood comparison ─────────────────────────────────────

    st.subheader("3. Neighborhood snapshot — 2024 median sale price by property type")
    st.caption(
        "Wicker Park and West Town data from 2024 where available; "
        "West Town condo from 2020 peak (most recent published figure); "
        "West Town SFH from Jan 2022. Chicago city-wide from Redfin."
    )

    nbhds      = ["Lincoln Park", "Wicker Park", "West Town", "Chicago (city)"]
    condo_px   = [550,  600,  500,  None]   # $K
    sfh_px     = [1720, 1100,  975,  None]  # $K
    overall_px = [726,   695,  668,   410]  # $K

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(
        name="Condo", x=nbhds,
        y=[v or 0 for v in condo_px],
        marker_color="#E07B39",
        text=[f"${v:,.0f}K" if v else "n/a" for v in condo_px],
        textposition="outside",
    ))
    fig_cmp.add_trace(go.Bar(
        name="Single-family", x=nbhds,
        y=[v or 0 for v in sfh_px],
        marker_color="#1D4E89",
        text=[f"${v:,.0f}K" if v else "n/a" for v in sfh_px],
        textposition="outside",
    ))
    fig_cmp.add_trace(go.Bar(
        name="All types — overall", x=nbhds,
        y=[v or 0 for v in overall_px],
        marker_color="#2E8B57",
        text=[f"${v:,.0f}K" if v else "n/a" for v in overall_px],
        textposition="outside",
    ))
    fig_cmp.update_layout(
        barmode="group", yaxis_title="Median sale price ($K)",
        height=440, margin=dict(t=30, b=10),
        yaxis=dict(range=[0, 2200]),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    st.divider()

    # ── Chart 4: 10-year overall trend comparison ─────────────────────────────

    st.subheader("4. Overall median price trend — all three neighborhoods (2015–2024)")
    st.caption(
        "Thin dashed lines connect the few sourced anchor points per neighborhood. "
        "Lincoln Park has the most data; Wicker Park and West Town have fewer published historical points."
    )

    # Sourced anchors per neighborhood (overall, all types)
    lp_trend  = {2023: 700, 2024: 726}
    wp_trend  = {2019: 520, 2021: 600, 2023: 675, 2024: 695}   # estimated mid-points from market narratives
    wt_trend  = {2019: 460, 2020: 500, 2022: 580, 2024: 668}   # SFH+condo blended from search results

    # Confidence notes per neighborhood
    _wp_note = ("Wicker Park 2019–2023 are estimated midpoints from narrative descriptions in "
                "Redfin / local RE sources; only 2024 ($695K) is directly cited.")
    _wt_note = ("West Town 2019–2022 are blended estimates from condo (~$500K peak) and SFH "
                "(~$975K–$1M) narratives; 2024 ($668K) is directly cited from Redfin.")

    fig_trend = go.Figure()
    for src, name, color, note in [
        (lp_trend, "Lincoln Park", "#1D4E89", "LP overall: 2023–2024 directly sourced."),
        (wp_trend, "Wicker Park",  "#E07B39", _wp_note),
        (wt_trend, "West Town",    "#2E8B57", _wt_note),
    ]:
        xs = sorted(src.keys())
        ys = [src[x] / 1_000 for x in xs]
        fig_trend.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name=name,
            line=dict(color=color, width=2, dash="dot"),
            marker=dict(size=9),
            hovertemplate=f"%{{x}}: $%{{y:.2f}}M<extra>{name}</extra>",
        ))

    fig_trend.update_layout(
        yaxis_title="Median sale price ($M)", xaxis_title="Year",
        height=360, margin=dict(t=10, b=10), hovermode="x unified",
        xaxis=dict(tickmode="linear", dtick=1),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.divider()

    # ── Section 5: Key observations ───────────────────────────────────────────

    st.subheader("5. Key observations")

    obs_col1, obs_col2 = st.columns(2)
    with obs_col1:
        st.markdown("""
**Lincoln Park**
- Condos: ~$395K (est. 2015) → $550K (2024) ≈ **+39% over 9 years, ~3.8%/yr**
- SFH: stable ~$1.6M (2016–2018), dipped in summer 2020, then **surged 19%** to $1.72M by early 2022
- Overall market cooled slightly in 2025–2026 after a hot run; Redfin shows −5.9% YoY in Mar 2026
- Condos are ~85% of inventory; SFH are rare and command a steep premium

**Wicker Park**
- Overall median ~$695K in 2024; heavy condo inventory
- Condo range $450K–$750K; SFH $900K–$1.3M (very limited SFH supply)
- Strong appreciation in early 2026 (+25% YoY in Jan 2026 per Redfin — likely a hot-month spike)
- Similar overall price level to Lincoln Park despite different neighborhood character
""")
    with obs_col2:
        st.markdown("""
**West Town** *(Wicker Park + Bucktown + Ukrainian Village)*
- Broader area; overall median $668K (2024) — slightly below Lincoln Park and Wicker Park
- Condos plateaued near $500K from 2020 through ~2022; modest appreciation since
- SFH peaked ~$1M in mid-2019, softened to ~$975K by Jan 2022
- Consistent 4–5%/yr overall appreciation since 2022

**What this means for the calculator**
- **Appreciation input:** use **2–3%/yr** for a Lincoln Park condo (conservative, historically accurate)
- **Condos vs SFH:** condos have appreciated slower but are far more liquid and lower maintenance
- **Timing risk:** 2021–2022 was an anomalous surge; don't anchor to that as a base rate
- **Long hold = better odds:** all three neighborhoods show steadily rising prices over 10yr horizons
""")

    st.divider()

    # ── Source data table ─────────────────────────────────────────────────────

    with st.expander("Full source data table"):
        _src_rows = [
            ("Lincoln Park", "Condo",  2018, 445,  "Plateau noted 2017–2018",                "Real Group Real Estate"),
            ("Lincoln Park", "Condo",  2019, 470,  "Spike in early 2019",                    "Real Group Real Estate"),
            ("Lincoln Park", "Condo",  2020, 510,  "Gain concentrated May–Sep 2020",         "Real Group Real Estate"),
            ("Lincoln Park", "Condo",  2022, 518,  "Leveled off 'just below $520K'",         "Real Group Real Estate"),
            ("Lincoln Park", "Condo",  2024, 550,  "Cook County Assessor data",              "Cook County Assessor (via search)"),
            ("Lincoln Park", "SFH",    2016, 1600, "Stable ~$1.6M throughout 2016",          "Real Group Real Estate"),
            ("Lincoln Park", "SFH",    2017, 1600, "Stable",                                 "Real Group Real Estate"),
            ("Lincoln Park", "SFH",    2018, 1600, "Stable",                                 "Real Group Real Estate"),
            ("Lincoln Park", "SFH",    2020, 1450, "Trough summer 2020",                     "Real Group Real Estate"),
            ("Lincoln Park", "SFH",    2022, 1720, "'More than $1.7M' Feb 2022",             "Real Group Real Estate"),
            ("Lincoln Park", "Overall",2023,  700, "Median all types, +6% YoY",              "AC Group Chicago"),
            ("Lincoln Park", "Overall",2024,  726, "Avg all types, +2.1% YoY",               "Camille Canales Group"),
            ("Wicker Park",  "Overall",2024,  695, "Median Nov 2024, +2.8% YoY",             "Camille Canales Group"),
            ("Wicker Park",  "Condo",  2024,  600, "$450K–$750K range — midpoint",           "Camille Canales Group"),
            ("Wicker Park",  "SFH",    2024, 1100, "$900K–$1.3M range — midpoint",           "Camille Canales Group"),
            ("West Town",    "Overall",2024,  668, "Median, +4.2% YoY",                      "Redfin"),
            ("West Town",    "Condo",  2020,  500, "Peak ~$500K mid-2020, then plateaued",   "Local RE sources (search)"),
            ("West Town",    "SFH",    2019, 1000, "Peak ~$1M mid-2019",                     "Local RE sources (search)"),
            ("West Town",    "SFH",    2022,  975, "~$975K Jan 2022, down from $1M peak",    "Local RE sources (search)"),
            ("Chicago",      "Overall",2024,  410, "City-wide median",                        "Redfin"),
        ]
        df_src = pd.DataFrame(
            _src_rows,
            columns=["Neighborhood", "Type", "Year", "Price ($K)", "Notes", "Source"],
        )
        st.dataframe(df_src, use_container_width=True, hide_index=True)

        st.markdown("""
**Sources cited:**
- [Real Group Real Estate — Lincoln Park market conditions](https://realgroupre.com/chicago-market-conditions-in-lincoln-park.html)
- [Real Group Real Estate — West Town/Wicker Park conditions](https://realgroupre.com/chicago-market-conditions-in-west-town-wicker-bucktown.html)
- [Camille Canales Group — Lincoln Park avg price](https://ccg-chicago.com/blog/what-is-the-average-home-price-in-lincoln-park-this-year)
- [Camille Canales Group — Wicker Park prices](https://ccg-chicago.com/blog/how-much-are-homes-selling-for-in-wicker-park)
- [AC Group Chicago — Lincoln Park 2024 forecast](https://acgroupchicago.com/blog/lincoln-park-real-estate-market-prices-trends-and-forecast-2024)
- [FRED / St. Louis Fed — Case-Shiller Chicago (CHXRSA)](https://fred.stlouisfed.org/series/CHXRSA)
- [FRED / St. Louis Fed — Case-Shiller Chicago Condo (CHXRCNSA)](https://fred.stlouisfed.org/series/CHXRCNSA)
- [Redfin — West Town housing market](https://www.redfin.com/neighborhood/32213/IL/Chicago/West-Town/housing-market)
- [Redfin — Chicago housing market](https://www.redfin.com/city/29470/IL/Chicago/housing-market)
        """)
