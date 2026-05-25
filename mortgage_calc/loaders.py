"""Load properties and assumption scenarios from YAML files.

Keeping these as YAML in version control means:
- Easy to diff changes to your assumptions over time
- Easy to share with Jasmine (or anyone else) for review
- No magic numbers buried in notebooks
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .scenarios import BuyAssumptions, FinancialContext, Property, RentAssumptions
from .rent_out import RentOutAssumptions


def load_property(path: str | Path) -> Property:
    """Load a single property from YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return Property(**{k: v for k, v in data.items() if k in Property.__dataclass_fields__})


def load_all_properties(directory: str | Path) -> dict[str, Property]:
    """Load every YAML in a directory as a Property, keyed by filename stem."""
    directory = Path(directory)
    properties = {}
    for path in sorted(directory.glob("*.yaml")):
        prop = load_property(path)
        properties[path.stem] = prop
    return properties


def load_scenario(path: str | Path) -> dict[str, Any]:
    """Load an assumption set. Returns dict of {buy, rent, ctx, rent_out}.

    `rent` and `rent_out` may be None if the scenario doesn't define them.
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    buy = BuyAssumptions(**data.get("buy", {}))
    ctx = FinancialContext(**data.get("ctx", {}))

    rent_data = data.get("rent")
    rent = RentAssumptions(**rent_data) if rent_data else None

    rent_out_data = data.get("rent_out")
    rent_out = RentOutAssumptions(**rent_out_data) if rent_out_data else None

    return {"buy": buy, "rent": rent, "ctx": ctx, "rent_out": rent_out}
