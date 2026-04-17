from dataclasses import dataclass
from typing import Optional


@dataclass
class Plant:
    """One fuel plant (customer node)."""
    name: str
    cap: float
    init_stock: float
    cons_rate: float
    deadline: Optional[float] = None   # if None, derived as init_stock / cons_rate (natural stockout)


@dataclass
class Ship:
    """Vessel parameters."""
    empty_weight: float
    pump_rate: float
    prep_time: float
    charter_rate: float
    fuel_cost: float
    speed: float
