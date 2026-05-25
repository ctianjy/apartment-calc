# apartment-calc

Rent-vs-buy decision analysis for Chicago condo hunting. Python-first, notebook-driven, version-controlled assumptions.

## What this does

- **Models a specific property** with all real costs: P&I, HOA, taxes, insurance, PMI, closing costs, selling costs, opportunity cost of cash
- **Compares buying vs. renting** an equivalent unit over a chosen hold period
- **Sensitivity analysis** — tornado charts, 2D heatmaps, breakeven solvers
- **Monte Carlo** — distribution of outcomes when inputs are uncertain
- **Rent-out scenario** — what happens if you keep the property and lease it after relocating

## What this does NOT do

- No live Zillow / Redfin / MLS data (those APIs are gated; manual property entry is the trade-off)
- No tax advice (consult a CPA on actual tax benefits)
- No investment recommendation (a model is only as good as its assumptions)

## Quick start

```bash
# Clone
git clone git@github.com:<your-username>/apartment-calc.git
cd apartment-calc

# Set up
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run the analysis
jupyter lab notebooks/
```

Open `notebooks/01_single_property_analysis.ipynb` first.

## GitHub Pages setup (one-time)

The included workflow renders notebooks to HTML and deploys to GitHub Pages on every push to `main`. To enable:

1. Go to your repo on GitHub → **Settings** → **Pages**
2. Under **"Build and deployment"**, set **Source** to **"GitHub Actions"**
3. Push to `main` — the workflow will run, and your rendered notebooks will be live at `https://<your-username>.github.io/apartment-calc/`

The first run may fail until Pages is enabled. After enabling, re-run the workflow from the Actions tab.

## Project structure

```
apartment-calc/
├── mortgage_calc/          # Core math (pure functions, testable)
│   ├── mortgage.py         # Amortization, PMI
│   ├── scenarios.py        # Rent-vs-buy engine
│   ├── sensitivity.py      # Sweeps, tornado, breakevens
│   ├── rent_out.py         # Landlord cash-flow model
│   └── loaders.py          # YAML loaders
├── notebooks/
│   ├── 01_single_property_analysis.ipynb
│   ├── 02_multi_property_comparison.ipynb
│   └── 03_monte_carlo.ipynb
├── properties/             # One YAML per property under consideration
│   ├── sample_lincoln_park.yaml
│   └── sample_old_town.yaml
├── scenarios/              # Assumption sets
│   ├── base.yaml
│   ├── optimistic.yaml
│   └── pessimistic.yaml
└── tests/                  # pytest sanity checks on the math
```

## Workflow

1. **Find a listing** (Zillow, Redfin, your realtor — the search part stays manual)
2. **Add a YAML** to `properties/` with price, HOA, taxes, etc.
3. **Open notebook 01** and point it at the new property
4. **Read the verdict**, check the tornado for which assumptions matter most
5. **Run Monte Carlo** to see the distribution
6. **Iterate on assumptions** by editing `scenarios/*.yaml` — version control your beliefs

## Key concepts

| Concept | What it means |
|---|---|
| **True economic gain** | Sale profit + rent savings − opportunity cost of cash. The honest rent-vs-buy number |
| **Cash-on-cash return** | Profit divided by total cash invested (down + closing). Captures leverage |
| **Breakeven appreciation** | The annual appreciation rate at which buying ties renting |
| **Sensitivity tornado** | Which input perturbation moves the answer the most |

## Caveats

- **Property tax in Cook County** can fluctuate — the YAMLs use monthly estimates; verify against actual assessments
- **HOA assessments** are not modeled — a $10K special assessment in year 2 isn't in the math
- **Tax benefit is off by default** — most filers don't itemize post-TCJA; turn on in scenario YAML if applicable
- **Appreciation is assumed constant** — real markets are lumpy; Monte Carlo captures this better than the point estimates

## Tests

```bash
pytest tests/
```

## Roadmap

Phase 1 (this repo): calculator + sensitivity + Monte Carlo
Phase 2: pull historical Chicago appreciation data automatically
Phase 3 (maybe): listing scraping with Zillow URL paste-in

## License

Private — for personal use.
