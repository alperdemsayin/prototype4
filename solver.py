"""MILP solver for the Maritime Inventory Routing Problem.

Integrates the original soft-deadline penalty model with the new
dashboard features: open/closed route, input validation, structured output.
"""

import time
from typing import Dict, List, Optional, Tuple

from ortools.linear_solver import pywraplp
from structures import Plant, Ship


BIG_M = 9999


# ==============================================================================
# PRE-PROCESSING
# ==============================================================================

def preprocess(
    plants: List[Plant],
    ship: Ship,
    dist,
    penalty: float = 1_000_000,
    return_to_depot: bool = False,
) -> Dict:
    """
    Compute all derived quantities used by the MILP model.

    Node numbering
    --------------
        depot    = 0
        plant i  ->  node index i  (1-based, matching plants list position)
        terminal = n + 1           (dummy end node, or depot in closed mode)
    """
    n = len(plants)
    depot = 0
    terminal = n + 1
    customers = list(range(1, n + 1))
    nodes = [depot] + customers + [terminal]

    # travel time matrix
    t = [[dist[i][j] / ship.speed for j in range(n + 2)] for i in range(n + 2)]

    gap, alpha, beta, nat_dl, eff_l, L = {}, {}, {}, {}, {}, {}
    for idx, p in enumerate(plants, start=1):
        gap[idx]   = p.cap - p.init_stock
        alpha[idx] = ship.prep_time + gap[idx] / ship.pump_rate
        beta[idx]  = p.cons_rate / ship.pump_rate
        nat_dl[idx] = p.init_stock / p.cons_rate          # natural stockout deadline
        # effective deadline: user-supplied or natural, whichever is tighter
        user_dl = p.deadline
        eff_l[idx] = min(user_dl, nat_dl[idx]) if user_dl is not None else nat_dl[idx]
        # penalty cost per hour of lateness (original formulation: P * CAP_i)
        L[idx] = penalty * p.cap

    # worst-case cargo capacity
    Q = sum(gap[i] + plants[i - 1].cons_rate * eff_l[i] for i in customers)

    return {
        "n": n,
        "depot": depot,
        "terminal": terminal,
        "C": customers,
        "V": nodes,
        "t": t,
        "gap": gap,
        "alpha": alpha,
        "beta": beta,
        "nat_dl": nat_dl,
        "eff_l": eff_l,
        "L": L,
        "Q": Q,
        "M": BIG_M,
        "return_to_depot": return_to_depot,
        "terminal_label": "Depot (return)" if return_to_depot else "End of service",
        "penalty": penalty,
    }


# ==============================================================================
# DIAGNOSTICS
# ==============================================================================

def quick_diagnostics(
    plants: List[Plant],
    ship: Ship,
    dist,
    return_to_depot: bool = False,
) -> Dict:
    """Fast, user-facing checks before optimization."""
    issues = []
    warnings = []

    if ship.speed <= 0:
        issues.append("Ship speed must be greater than 0.")
    if ship.pump_rate <= 0:
        issues.append("Pump rate must be greater than 0.")
    if ship.empty_weight < 0 or ship.charter_rate < 0 or ship.fuel_cost < 0 or ship.prep_time < 0:
        issues.append("Ship parameters cannot be negative.")

    names = [p.name.strip() for p in plants]
    if any(not n for n in names):
        issues.append("Every plant must have a name.")
    if len(set(names)) != len(names):
        issues.append("Plant names must be unique.")

    n = len(plants)
    if len(dist) != n + 2 or any(len(row) != n + 2 for row in dist):
        issues.append("Distance matrix size is inconsistent with the number of plants.")
        return {"valid": False, "issues": issues, "warnings": warnings, "plant_checks": []}

    plant_checks = []
    for idx, plant in enumerate(plants, start=1):
        nat_dl = plant.init_stock / plant.cons_rate if plant.cons_rate > 0 else None
        user_dl = plant.deadline
        eff_dl = min(user_dl, nat_dl) if (user_dl is not None and nat_dl is not None) else nat_dl

        entry = {
            "Plant": plant.name,
            "Natural dl (hr)": round(nat_dl, 2) if nat_dl is not None else None,
            "User deadline (hr)": round(user_dl, 2) if user_dl is not None else "Auto",
            "Effective dl (hr)": round(eff_dl, 2) if eff_dl is not None else None,
            "Nearest travel from depot (hr)": None,
            "Min service start slack (hr)": None,
            "Likely feasible from depot": None,
        }

        if plant.cap <= 0:
            issues.append(f"{plant.name}: capacity must be greater than 0.")
        if plant.init_stock < 0:
            issues.append(f"{plant.name}: initial stock cannot be negative.")
        if plant.cons_rate <= 0:
            issues.append(f"{plant.name}: consumption rate must be greater than 0.")
        if plant.init_stock > plant.cap:
            issues.append(f"{plant.name}: initial stock cannot exceed capacity.")
        if user_dl is not None and user_dl <= 0:
            issues.append(f"{plant.name}: deadline must be greater than 0.")

        if idx < len(dist) and dist[0][idx] < 0:
            issues.append(f"{plant.name}: distance from depot cannot be negative.")

        if plant.cons_rate > 0 and ship.speed > 0 and ship.pump_rate > 0 and eff_dl is not None:
            travel = dist[0][idx] / ship.speed
            slack = eff_dl - travel
            entry["Nearest travel from depot (hr)"] = round(travel, 2)
            entry["Min service start slack (hr)"] = round(slack, 2)
            entry["Likely feasible from depot"] = slack >= 0
            if slack < 0:
                warnings.append(
                    f"{plant.name}: cannot be reached from the depot before its effective deadline. "
                    f"The penalty model will still try to minimise lateness."
                )
            elif slack < 4:
                warnings.append(f"{plant.name} has very low timing slack ({slack:.2f} hr).")

        plant_checks.append(entry)

    if return_to_depot:
        warnings.append("Closed-route mode: final return to depot is included in time and cost.")
    else:
        warnings.append("Open-route mode: the voyage ends after the last delivery.")

    return {"valid": not issues, "issues": issues, "warnings": warnings, "plant_checks": plant_checks}


# ==============================================================================
# ARC FILTERING
# ==============================================================================

def _valid_arcs(
    nodes: List[int], depot: int, terminal: int, customers: List[int]
) -> List[Tuple[int, int]]:
    arcs = []
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            if j == depot and i != terminal:
                continue
            if i == terminal:
                continue
            if i == depot and j == terminal:
                continue
            if j == terminal and i not in customers:
                continue
            arcs.append((i, j))
    return arcs


# ==============================================================================
# SOLVER
# ==============================================================================

def run_solver(
    plants: List[Plant],
    ship: Ship,
    dist,
    penalty: float = 1_000_000,
    return_to_depot: bool = False,
    top_n: int = 1,
):
    """
    Build and solve the MILP.

    Uses the original soft-deadline formulation:
      - Lateness slack variable σ_i per plant (allows late delivery with penalty).
      - Penalty cost = penalty * CAP_i per hour of lateness.
      - No hard deadline constraint — the solver trades off lateness vs. cost.

    Parameters
    ----------
    plants         : list of Plant objects
    ship           : Ship object
    dist           : distance matrix (n+2 × n+2)
    penalty        : penalty coefficient P ($/hr per unit capacity).  Default 1_000_000.
    return_to_depot: if True, terminal node is the depot (closed route).
    top_n          : number of top solutions to return (uses NextSolution()).
    """
    diagnostics = quick_diagnostics(plants, ship, dist, return_to_depot=return_to_depot)
    if not diagnostics["valid"]:
        return {"kind": "validation_error", "diagnostics": diagnostics}

    pre = preprocess(plants, ship, dist, penalty=penalty, return_to_depot=return_to_depot)

    V, C      = pre["V"], pre["C"]
    depot     = pre["depot"]
    terminal  = pre["terminal"]
    t         = pre["t"]
    alpha     = pre["alpha"]
    beta      = pre["beta"]
    gap       = pre["gap"]
    eff_l     = pre["eff_l"]
    L         = pre["L"]
    Q         = pre["Q"]
    M         = pre["M"]

    solver = pywraplp.Solver.CreateSolver("SCIP") or pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        return "Could not initialise OR-Tools solver (SCIP/CBC unavailable)."

    INF  = solver.infinity()
    arcs = _valid_arcs(V, depot, terminal, C)

    # ── decision variables ────────────────────────────────────────────
    x     = {(i, j): solver.BoolVar(f"x_{i}_{j}") for i, j in arcs}
    f     = {(i, j): solver.NumVar(0, INF, f"f_{i}_{j}") for i, j in arcs}
    u     = {i: solver.NumVar(0, INF, f"u_{i}") for i in V}
    sigma = {i: solver.NumVar(0, INF, f"sigma_{i}") for i in C}   # lateness slack

    # ── objective: charter + empty fuel + cargo fuel + lateness penalty ──
    charter_cost  = ship.charter_rate * u[terminal]
    empty_fuel    = solver.Sum(
        ship.fuel_cost * dist[i][j] * ship.empty_weight * x[i, j] for i, j in arcs
    )
    cargo_fuel    = solver.Sum(
        ship.fuel_cost * dist[i][j] * f[i, j] for i, j in arcs
    )
    lateness_cost = solver.Sum(L[i] * sigma[i] for i in C)
    solver.Minimize(charter_cost + empty_fuel + cargo_fuel + lateness_cost)

    # ── (1) depart depot exactly once ────────────────────────────────
    solver.Add(solver.Sum(x[depot, j] for j in C) == 1)

    # ── (2) arrive at terminal exactly once ──────────────────────────
    solver.Add(solver.Sum(x[i, terminal] for i in C) == 1)

    # ── (3) flow conservation at each plant ──────────────────────────
    for j in C:
        incoming = solver.Sum(x[i, j] for i, k in arcs if k == j)
        outgoing = solver.Sum(x[j, k] for i, k in arcs if i == j)
        solver.Add(incoming == outgoing)
        solver.Add(incoming <= 1)
        solver.Add(outgoing <= 1)

    # ── (4) time propagation + MTZ subtour elimination ────────────────
    for i, j in arcs:
        a = alpha.get(i, 0.0)
        b = beta.get(i, 0.0)
        solver.Add((1 + b) * u[i] + a + t[i][j] - M * (1 - x[i, j]) <= u[j])

    # ── (5) inventory balance ─────────────────────────────────────────
    for i in C:
        inflow  = solver.Sum(f[k, i] for k, j in arcs if j == i)
        outflow = solver.Sum(f[i, j] for k, j in arcs if k == i)
        solver.Add(inflow - outflow - plants[i - 1].cons_rate * u[i] == gap[i])

    # ── (6) cargo capacity: f_ij ≤ Q * x_ij ─────────────────────────
    for i, j in arcs:
        solver.Add(f[i, j] <= Q * x[i, j])

    # ── (7) soft deadline: u_i ≤ eff_l_i + sigma_i ───────────────────
    #    (original formulation — allows lateness, penalises it)
    for i in C:
        solver.Add(u[i] <= eff_l[i] + sigma[i])

    # ── (8) clock starts at depot ────────────────────────────────────
    solver.Add(u[depot] == 0)

    solver.SetTimeLimit(60_000)
    t0     = time.time()
    status = solver.Solve()
    elapsed = round(time.time() - t0, 2)

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return {
            "kind": "infeasible",
            "status_code": status,
            "message": f"No feasible solution found (solver status code {status}).",
            "diagnostics": diagnostics,
            "elapsed": elapsed,
        }

    # ── helper ────────────────────────────────────────────────────────
    def label(i: int) -> str:
        if i == depot:    return "Depot"
        if i == terminal: return pre["terminal_label"]
        return plants[i - 1].name

    def extract_solution(sol_index: int):
        sol_tag = "OPTIMAL" if (sol_index == 0 and status == pywraplp.Solver.OPTIMAL) \
                  else "FEASIBLE"

        route_nodes, cur = [depot], depot
        safety = 0
        while cur != terminal and safety < len(V) + 5:
            nxt = next((j for i, j in arcs if i == cur and x[i, j].solution_value() > 0.5), None)
            if nxt is None:
                break
            route_nodes.append(nxt)
            cur = nxt
            safety += 1
        route_labels = [label(n) for n in route_nodes]

        deliveries = []
        total_lateness_penalty = 0.0
        for i in C:
            p = plants[i - 1]
            arr  = u[i].solution_value()
            late = sigma[i].solution_value()
            consumed  = p.cons_rate * arr
            delivered = gap[i] + consumed
            stock_at_arrival = p.init_stock - consumed
            pen = L[i] * late
            total_lateness_penalty += pen
            deliveries.append({
                "Plant":              p.name,
                "Arrival (hr)":       round(arr, 3),
                "Eff. Deadline (hr)": round(eff_l[i], 3),
                "Lateness (hr)":      round(late, 3),
                "Init Stock (T)":     p.init_stock,
                "Consumed (T)":       round(consumed, 3),
                "Stock at arrival (T)": round(stock_at_arrival, 3),
                "Delivered (T)":      round(delivered, 3),
                "Final Stock (T)":    p.cap,
                "Slack vs eff dl (hr)": round(eff_l[i] - arr, 3),
                "On Time":            late < 1e-6,
            })

        charter_val  = ship.charter_rate * u[terminal].solution_value()
        empty_f_val  = sum(
            ship.fuel_cost * dist[i][j] * ship.empty_weight * x[i, j].solution_value()
            for i, j in arcs
        )
        cargo_f_val  = sum(
            ship.fuel_cost * dist[i][j] * f[i, j].solution_value()
            for i, j in arcs
        )
        total_cost = charter_val + empty_f_val + cargo_f_val + total_lateness_penalty

        active_arcs = [
            {
                "From":            label(i),
                "To":              label(j),
                "Dist (NM)":       round(dist[i][j], 2),
                "Travel (hr)":     round(t[i][j], 3),
                "Fuel on Board (T)": round(f[i, j].solution_value(), 2),
            }
            for i, j in arcs if x[i, j].solution_value() > 0.5
        ]

        return {
            "kind":          "solution",
            "solution_rank": sol_index + 1,
            "status":        sol_tag,
            "elapsed":       elapsed,
            "total_cost":    total_cost,
            "charter":       charter_val,
            "empty_fuel":    empty_f_val,
            "cargo_fuel":    cargo_f_val,
            "lateness_penalty": total_lateness_penalty,
            "voyage_time":   u[terminal].solution_value(),
            "route_nodes":   route_nodes,
            "route_labels":  route_labels,
            "deliveries":    deliveries,
            "arcs":          active_arcs,
            "n_vars":        solver.NumVariables(),
            "n_cons":        solver.NumConstraints(),
            "Q":             Q,
            "pre":           pre,
            "diagnostics":   diagnostics,
            "return_to_depot": return_to_depot,
        }

    # ── collect up to top_n solutions ────────────────────────────────
    solutions = [extract_solution(0)]
    for rank in range(1, top_n):
        if not solver.NextSolution():
            break
        solutions.append(extract_solution(rank))

    return {
        "kind":      "multi_solution",
        "solutions": solutions,
        "n_found":   len(solutions),
        "elapsed":   elapsed,
        "diagnostics": diagnostics,
    }
