"""Streamlit app for integrated maritime inventory routing and fuel cost optimization."""

import json
import math
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from structures import Plant, Ship
from solver import quick_diagnostics, run_solver


st.set_page_config(
    page_title="Maritime MIRP DSS",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = (
    "A DECISION SUPPORT SYSTEM FOR INTEGRATED MARITIME INVENTORY "
    "ROUTING AND FUEL CONSUMPTION COST OPTIMIZATION"
)

FIXED_SCENARIO = {
    "depot": {"name": "Istanbul Depot", "lat": 41.0082, "lon": 28.9784},
    "plants": [
        {"name": "Antalya",    "lat": 36.8969, "lon": 30.7133, "cap": 500.0, "init_stock": 400.0, "cons_rate": 5.0,  "deadline": 120.0},
        {"name": "Iskenderun", "lat": 36.5872, "lon": 36.1735, "cap": 420.0, "init_stock": 330.0, "cons_rate": 4.0,  "deadline": 110.0},
        {"name": "Mersin",     "lat": 36.8000, "lon": 34.6333, "cap": 600.0, "init_stock": 520.0, "cons_rate": 6.0,  "deadline": 120.0},
        {"name": "Canakkale",  "lat": 40.1553, "lon": 26.4142, "cap": 350.0, "init_stock": 300.0, "cons_rate": 3.0,  "deadline":  90.0},
        {"name": "Izmir",      "lat": 38.4237, "lon": 27.1428, "cap": 480.0, "init_stock": 360.0, "cons_rate": 4.5,  "deadline": 100.0},
        {"name": "Samsun",     "lat": 41.2867, "lon": 36.3300, "cap": 390.0, "init_stock": 300.0, "cons_rate": 3.8,  "deadline": 105.0},
    ],
}

DEFAULT_SHIP = {
    "empty_weight": 2000.0,
    "pump_rate":      50.0,
    "prep_time":       0.5,
    "charter_rate":  500.0,
    "fuel_cost":      0.02,
    "speed":          15.0,
}


def haversine_nm(lat1, lon1, lat2, lon2):
    r = 3440.065
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return r * 2 * math.asin(math.sqrt(a))


@st.cache_data(show_spinner=False)
def compute_distance_matrix(depot_lat: float, depot_lon: float, plant_rows: List[Dict]):
    n = len(plant_rows)
    dist = [[0.0] * (n + 2) for _ in range(n + 2)]
    points = [(depot_lat, depot_lon)] + [(p["lat"], p["lon"]) for p in plant_rows]
    for i in range(n + 1):
        for j in range(n + 1):
            if i != j:
                dist[i][j] = round(
                    haversine_nm(points[i][0], points[i][1], points[j][0], points[j][1]), 1
                )
    return dist


def make_active_plant_rows() -> List[Dict]:
    rows = []
    for i, item in enumerate(st.session_state.fixed_plants):
        if item["enabled"]:
            rows.append({
                "id":         i + 1,
                "name":       item["name"],
                "lat":        float(item["lat"]),
                "lon":        float(item["lon"]),
                "cap":        float(item["cap"]),
                "init_stock": float(item["init_stock"]),
                "cons_rate":  float(item["cons_rate"]),
                "deadline":   float(item["deadline"]),
            })
    return rows


def make_plants(rows: List[Dict]) -> List[Plant]:
    return [
        Plant(name=r["name"], cap=r["cap"], init_stock=r["init_stock"],
              cons_rate=r["cons_rate"], deadline=r["deadline"])
        for r in rows
    ]


def build_bundle(result: Dict) -> bytes:
    return json.dumps({
        "status":       result.get("status"),
        "voyage_time":  result.get("voyage_time"),
        "total_cost":   result.get("total_cost"),
        "route_labels": result.get("route_labels"),
        "deliveries":   result.get("deliveries"),
        "arcs":         result.get("arcs"),
    }, indent=2).encode("utf-8")


def show_top_summary(ship: Ship, active_rows: List[Dict], route_mode: str):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Active plants",  len(active_rows))
    m2.metric("Route type",     route_mode)
    m3.metric("Vessel speed",   f"{ship.speed:.1f} NM/hr")
    m4.metric("Pump rate",      f"{ship.pump_rate:.1f} T/hr")


def render_map(result: Dict, active_rows: List[Dict], depot: Dict, rank: int = 1):
    coord_map = {"Depot": (depot["lat"], depot["lon"]), depot["name"]: (depot["lat"], depot["lon"])}
    for row in active_rows:
        coord_map[row["name"]] = (row["lat"], row["lon"])

    visit_order, order = {}, 1
    for lbl in result["route_labels"]:
        if lbl in {"Depot", "End of service", "Depot (return)"}:
            continue
        if lbl in coord_map and lbl not in visit_order:
            visit_order[lbl] = order
            order += 1

    fig = go.Figure()
    fig.add_trace(go.Scattermapbox(
        lat=[depot["lat"]], lon=[depot["lon"]],
        mode="markers+text",
        marker=dict(size=18, color="#1d4ed8"),
        text=["Depot"], textposition="top right",
        textfont=dict(size=14, color="#1d4ed8"),
        name="Depot",
        hovertemplate=f"<b>{depot['name']}</b><br>Starting depot<extra></extra>",
    ))

    uv_lat, uv_lon, uv_hov = [], [], []
    v_lat, v_lon, v_hov, v_txt = [], [], [], []

    for row in active_rows:
        dlv = next((d for d in result["deliveries"] if d["Plant"] == row["name"]), None)
        hover = f"<b>{row['name']}</b>"
        if row["name"] in visit_order:
            hover += f"<br>Visit order: {visit_order[row['name']]}"
        if dlv:
            late = dlv.get("Lateness (hr)", 0)
            late_str = f" ⚠️ +{late:.2f} hr" if late > 1e-6 else " ✅"
            hover += (
                f"<br>Arrival: {dlv['Arrival (hr)']} hr"
                f"<br>Deadline: {dlv['Eff. Deadline (hr)']} hr{late_str}"
                f"<br>Delivered: {dlv['Delivered (T)']} T"
                f"<br>Slack: {dlv['Slack vs eff dl (hr)']} hr"
            )
        hover += "<extra></extra>"
        if row["name"] in visit_order:
            v_lat.append(row["lat"]); v_lon.append(row["lon"])
            v_hov.append(hover); v_txt.append(str(visit_order[row["name"]]))
        else:
            uv_lat.append(row["lat"]); uv_lon.append(row["lon"]); uv_hov.append(hover)

    if uv_lat:
        fig.add_trace(go.Scattermapbox(lat=uv_lat, lon=uv_lon, mode="markers",
            marker=dict(size=14, color="#f59e0b"), name="Plants (unvisited)",
            hovertemplate=uv_hov, showlegend=False))
    if v_lat:
        fig.add_trace(go.Scattermapbox(
            lat=v_lat, lon=v_lon, mode="markers",
            marker=dict(size=14, color="#ef4444"),
            name="Visited plants", hovertemplate=v_hov, showlegend=False))

        fig.add_trace(go.Scattermapbox(
            lat=v_lat, lon=v_lon,
            mode="text",
            text=[str(x) for x in v_txt],
            textposition="top center",
            textfont=dict(size=16, color="#111111", family="Arial Black"),
            hoverinfo="skip", showlegend=False))

    fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0), height=520)
    st.plotly_chart(fig, use_container_width=True, key=f"map_chart_{rank}")


def render_one_solution(result: Dict, active_rows: List[Dict], depot: Dict, rank: int = 1):
    on_time   = sum(1 for d in result["deliveries"] if d["On Time"])
    total_pl  = len(result["deliveries"])
    late_cnt  = total_pl - on_time

    a, b, c, d = st.columns(4)
    a.metric("Total cost",         f"${result['total_cost']:,.0f}")
    b.metric("Voyage time",        f"{result['voyage_time']:.2f} hr")
    c.metric("On-time deliveries", f"{on_time} / {total_pl}")
    if late_cnt > 0:
        d.metric("Lateness penalty", f"${result['lateness_penalty']:,.0f}",
                 delta=f"{late_cnt} plant(s) late", delta_color="inverse")
    else:
        d.metric("Lateness penalty", "$0", delta="All on time ✅", delta_color="normal")

    st.markdown("**Route:** " + " → ".join(result["route_labels"]))

    t1, t2, t3, t4 = st.tabs(["Map", "Deliveries", "Costs", "Technical details"])

    with t1:
        render_map(result, active_rows, depot, rank=rank)

    with t2:
        df = pd.DataFrame(result["deliveries"])
        if not df.empty:
            bar_df = df[["Plant", "Delivered (T)"]].copy()
            bar_df["Delivered (T)"] = pd.to_numeric(bar_df["Delivered (T)"], errors="coerce").fillna(0)
            st.plotly_chart(
                go.Figure(data=[go.Bar(x=bar_df["Plant"], y=bar_df["Delivered (T)"],
                    text=bar_df["Delivered (T)"], textposition="outside")]
                ).update_layout(title="Delivered quantity by plant",
                    xaxis_title="Plant", yaxis_title="Delivered (T)"),
                use_container_width=True, key=f"bar_delivered_{rank}")
            pie_src = bar_df[bar_df["Delivered (T)"] > 0]
            if not pie_src.empty:
                st.plotly_chart(
                    go.Figure(data=[go.Pie(labels=pie_src["Plant"],
                        values=pie_src["Delivered (T)"], hole=0.35)]
                    ).update_layout(title="Delivery share by plant"),
                    use_container_width=True, key=f"pie_delivered_{rank}")

        def highlight_late(row):
            if not row.get("On Time", True):
                return ["background-color: #fef2f2"] * len(row)
            return [""] * len(row)

        st.dataframe(df.style.apply(highlight_late, axis=1), use_container_width=True, hide_index=True, key=f"df_deliveries_{rank}")
        st.download_button("Download deliveries CSV",
            df.to_csv(index=False).encode("utf-8"), f"deliveries_sol{rank}.csv", "text/csv", key=f"dl_csv_{rank}")

    with t3:
        costs = pd.DataFrame([
            {"Component": "Charter cost",         "Value ($)": round(result["charter"], 2)},
            {"Component": "Empty-ship fuel cost", "Value ($)": round(result["empty_fuel"], 2)},
            {"Component": "Cargo fuel cost",      "Value ($)": round(result["cargo_fuel"], 2)},
            {"Component": "Lateness penalty",     "Value ($)": round(result["lateness_penalty"], 2)},
        ])
        st.plotly_chart(
            go.Figure(data=[go.Bar(x=costs["Component"], y=costs["Value ($)"],
                text=costs["Value ($)"].round(2), textposition="outside")]
            ).update_layout(title="Cost breakdown", xaxis_title="Component", yaxis_title="Cost ($)"),
            use_container_width=True, key=f"bar_costs_{rank}")
        pie_costs = costs[costs["Value ($)"] > 0]
        if not pie_costs.empty:
            st.plotly_chart(
                go.Figure(data=[go.Pie(labels=pie_costs["Component"],
                    values=pie_costs["Value ($)"], hole=0.35)]
                ).update_layout(title="Cost share"),
                use_container_width=True, key=f"pie_costs_{rank}")
        total_row = pd.DataFrame([{"Component": "TOTAL", "Value ($)": round(result["total_cost"], 2)}])
        st.dataframe(pd.concat([costs, total_row], ignore_index=True),
            use_container_width=True, hide_index=True, key=f"df_costs_{rank}")
        st.download_button("Download result JSON", build_bundle(result),
            f"mirp_result_sol{rank}.json", "application/json", key=f"dl_json_{rank}")

    with t4:
        st.markdown("#### Active arcs")
        st.dataframe(pd.DataFrame(result["arcs"]), use_container_width=True, hide_index=True, key=f"df_arcs_{rank}")
        pre = result.get("pre", {})
        if pre:
            st.markdown("#### Model coefficients")
            st.write({
                "worst_case_cargo_Q":  round(pre.get("Q", 0.0), 3),
                "penalty_coefficient": pre.get("penalty"),
                "alpha":  {i: round(v, 3) for i, v in pre.get("alpha", {}).items()},
                "beta":   {i: round(v, 4) for i, v in pre.get("beta", {}).items()},
                "eff_l":  {i: round(v, 2) for i, v in pre.get("eff_l", {}).items()},
                "L_i":    pre.get("L", {}),
                "terminal_label": pre.get("terminal_label"),
            })
        st.caption(
            f"OR-Tools SCIP | Variables: {result['n_vars']} | "
            f"Constraints: {result['n_cons']} | Solve time: {result['elapsed']} s"
        )


def render_results(multi: Dict, active_rows: List[Dict], depot: Dict):
    if isinstance(multi, str):
        st.error(multi); return
    if multi.get("kind") == "validation_error":
        st.error("Input validation failed.")
        for issue in multi["diagnostics"]["issues"]:
            st.write(f"- {issue}")
        return
    if multi.get("kind") == "infeasible":
        st.error(multi["message"])
        checks = pd.DataFrame(multi["diagnostics"].get("plant_checks", []))
        if not checks.empty:
            st.dataframe(checks, use_container_width=True, hide_index=True)
        return

    solutions = multi.get("solutions", [])
    st.caption(f"Found {multi.get('n_found', len(solutions))} solution(s) | Solve time: {multi['elapsed']} s")

    for w in multi.get("diagnostics", {}).get("warnings", []):
        st.warning(w)

    if len(solutions) == 1:
        sol = solutions[0]
        st.markdown(f"### Solution #1 — {sol['status']}")
        render_one_solution(sol, active_rows, depot, rank=1)
    else:
        tabs = st.tabs([f"Solution #{s['solution_rank']} — {s['status']}" for s in solutions])
        for tab, sol in zip(tabs, solutions):
            with tab:
                render_one_solution(sol, active_rows, depot, rank=sol["solution_rank"])


# ── session state ─────────────────────────────────────────────────────────────

if "fixed_plants" not in st.session_state:
    st.session_state.fixed_plants = [dict(item, enabled=True) for item in FIXED_SCENARIO["plants"]]
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = None

# ── page layout ───────────────────────────────────────────────────────────────

st.title(APP_TITLE)
st.divider()

setup_tab, results_tab = st.tabs(["Setup", "Results"])

with setup_tab:
    left, right = st.columns([1.1, 0.9])

    with left:
        st.markdown("### Select active plants")
        for idx, plant in enumerate(st.session_state.fixed_plants):
            with st.container(border=True):
                c1, c2 = st.columns([0.9, 2.4])
                with c1:
                    plant["enabled"] = st.toggle(
                        f"Use {plant['name']}", value=plant["enabled"], key=f"enabled_{idx}")
                with c2:
                    st.write(f"**{plant['name']}**")
                    e1, e2, e3, e4 = st.columns(4)
                    plant["cap"]        = e1.number_input(f"Cap {plant['name']}",      min_value=0.0,  value=float(plant["cap"]),        step=10.0,  key=f"cap_{idx}")
                    plant["init_stock"] = e2.number_input(f"Init {plant['name']}",     min_value=0.0,  value=float(plant["init_stock"]),  step=10.0,  key=f"init_{idx}")
                    plant["cons_rate"]  = e3.number_input(f"Cons {plant['name']}",     min_value=0.01, value=float(plant["cons_rate"]),   step=0.1,   key=f"cons_{idx}")
                    plant["deadline"]   = e4.number_input(f"Deadline {plant['name']}", min_value=0.1,
                        value=float(plant.get("deadline") or plant["init_stock"] / plant["cons_rate"]),
                        step=1.0, key=f"ddl_{idx}")
                    st.write(f"Lat {plant['lat']}, Lon {plant['lon']}")

    with right:
        st.markdown("### Vessel inputs")
        with st.container(border=True):
            c1, c2 = st.columns(2)
            empty_weight = c1.number_input("Empty weight (T)",     min_value=0.0, value=DEFAULT_SHIP["empty_weight"], step=100.0)
            pump_rate    = c2.number_input("Pump rate (T/hr)",     min_value=0.1, value=DEFAULT_SHIP["pump_rate"],    step=5.0)
            prep_time    = c1.number_input("Preparation time (hr)",min_value=0.0, value=DEFAULT_SHIP["prep_time"],    step=0.1)
            charter_rate = c2.number_input("Charter rate ($/hr)",  min_value=0.0, value=DEFAULT_SHIP["charter_rate"], step=50.0)
            fuel_cost    = c1.number_input("Fuel cost ($/Ton-NM)", min_value=0.0, value=DEFAULT_SHIP["fuel_cost"],    step=0.01, format="%.4f")
            speed        = c2.number_input("Speed (NM/hr)",        min_value=0.1, value=DEFAULT_SHIP["speed"],        step=1.0)

        st.markdown("### Solver options")
        with st.container(border=True):
            o1, o2, o3 = st.columns(3)
            return_to_depot = o1.toggle("Closed route (return to depot)", value=False)
            top_n           = o2.number_input("Top N solutions", min_value=1, max_value=10, value=1, step=1)
            penalty         = o3.number_input(
                "Lateness penalty coefficient (P)",
                min_value=0.0, value=1_000_000.0, step=100_000.0, format="%.0f",
                help="Penalty per hour of lateness × tank capacity. "
                     "Higher = avoids lateness. Lower = may trade lateness for cost savings.",
            )
            st.caption(
                "**Soft-deadline model** (original formulation): late delivery is *allowed* but "
                "penalised at P × CAP_i $/hr. Set P very high to approximate a hard deadline."
            )

        depot       = FIXED_SCENARIO["depot"]
        active_rows = make_active_plant_rows()
        route_mode  = "Closed route" if return_to_depot else "Open route"
        ship = Ship(empty_weight=empty_weight, pump_rate=pump_rate, prep_time=prep_time,
                    charter_rate=charter_rate, fuel_cost=fuel_cost, speed=speed)

        if active_rows:
            plants      = make_plants(active_rows)
            dist        = compute_distance_matrix(depot["lat"], depot["lon"], active_rows)
            diagnostics = quick_diagnostics(plants, ship, dist, return_to_depot=return_to_depot)
            show_top_summary(ship, active_rows, route_mode)
            for w in diagnostics.get("warnings", []):
                st.warning(w)
            for issue in diagnostics.get("issues", []):
                st.error(issue)
            st.markdown("### Active plant table")
            st.dataframe(pd.DataFrame(active_rows), use_container_width=True, hide_index=True)
            if diagnostics.get("plant_checks"):
                with st.expander("Pre-solve diagnostics"):
                    st.dataframe(pd.DataFrame(diagnostics["plant_checks"]),
                        use_container_width=True, hide_index=True)
        else:
            st.error("Select at least one plant to create a scenario.")

        run_clicked = st.button("Run optimization", type="primary", use_container_width=True)
        if run_clicked:
            if not active_rows:
                st.session_state.last_result = {
                    "kind": "validation_error",
                    "diagnostics": {"issues": ["Select at least one plant."], "warnings": []},
                }
            else:
                with st.spinner("Solving…"):
                    result = run_solver(plants, ship, dist,
                        penalty=penalty, return_to_depot=return_to_depot, top_n=int(top_n))
                st.session_state.last_result = result
                st.session_state.last_inputs = {"active_rows": active_rows, "depot": depot}
                st.success("Optimization complete — see the Results tab.")

with results_tab:
    st.markdown("### Optimization results")
    if st.session_state.last_result is None:
        st.write("Run the model from the Setup tab to view results.")
    else:
        render_results(st.session_state.last_result,
            st.session_state.last_inputs["active_rows"],
            st.session_state.last_inputs["depot"])
