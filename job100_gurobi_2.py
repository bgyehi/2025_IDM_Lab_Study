"""
job100_gurobi_restart.py
Single-machine scheduling (minimize total tardiness) for n=100 jobs.
Features:
 - instance generator with "tightness" tuning to create '아슬아슬' due-dates
 - SPT / EDD baselines
 - simple local search (optional) to improve SPT
 - Gurobi MILP with SPT-based MIP start, tightened Big-M, solver parameters
 - prints summary and optionally saves CSV with order & times
"""

import random
import math
import time
import csv
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as e:
    raise ImportError("gurobipy import failed. Install Gurobi and gurobipy. Error: " + str(e))


# -----------------------
# Instance generation
# -----------------------
def generate_instance(n: int = 100,
                      p_low: int = 1,
                      p_high: int = 100,
                      tightness: float = 0.45,
                      seed: int = None) -> Tuple[List[int], List[int]]:
    """
    Return p (processing times) and d (due dates).
    tightness: fraction controlling spread of due dates (smaller -> tighter cluster)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    p = [random.randint(p_low, p_high) for _ in range(n)]
    sumP = sum(p)
    center = sumP / 2.0
    spread = sumP * tightness
    # create due dates around center, add p_j to avoid too-small d_j
    d = [int(round(random.uniform(center - spread/2, center + spread/2))) + pj for pj in p]
    return p, d


def tune_tightness_for_target(n=100, target_frac=0.5, tol=0.06, max_iters=20, seed=123):
    """
    Heuristic: adjust tightness so that SPT gives approx target_frac tardy fraction.
    Returns p,d,tightness,observed_frac
    """
    low, high = 0.05, 1.0
    last_frac = None
    for it in range(max_iters):
        mid = (low + high) / 2.0
        p, d = generate_instance(n=n, tightness=mid, seed=seed + it)
        frac = compute_tardy_fraction_spt(p, d)
        last_frac = frac
        if abs(frac - target_frac) <= tol:
            return p, d, mid, frac
        if frac > target_frac:
            # too many tardy -> increase spread (loosen) -> increase tightness
            low = mid
        else:
            high = mid
    return p, d, mid, last_frac


# -----------------------
# Baselines & utils
# -----------------------
def schedule_by_order(p: List[int], d: List[int], order: List[int]):
    time_acc = 0
    S = [0.0] * len(p)
    C = [0.0] * len(p)
    T = [0.0] * len(p)
    for idx in order:
        S[idx] = time_acc
        time_acc += p[idx]
        C[idx] = S[idx] + p[idx]
        T[idx] = max(0.0, C[idx] - d[idx])
    total_tardy = sum(T)
    return total_tardy, S, C, T


def spt_total_tardiness(p: List[int], d: List[int]):
    order = sorted(range(len(p)), key=lambda j: p[j])
    return schedule_by_order(p, d, order)


def edd_total_tardiness(p: List[int], d: List[int]):
    order = sorted(range(len(p)), key=lambda j: d[j])
    return schedule_by_order(p, d, order)


def compute_tardy_fraction_spt(p, d):
    total_tardy, S, C, T = spt_total_tardiness(p, d)
    tardy_count = sum(1 for t in T if t > 1e-9)
    return tardy_count / len(p)


# Simple local search: try inserting a job from position i to j if it improves total tardiness.
def local_search_insertion(p, d, base_order, max_iters=500):
    best_order = base_order.copy()
    best_obj, _, _, _ = schedule_by_order(p, d, best_order)
    n = len(p)
    it = 0
    improved = True
    while improved and it < max_iters:
        improved = False
        it += 1
        # try random pairwise insert attempts
        i = random.randrange(n)
        j = random.randrange(n)
        if i == j:
            continue
        order = best_order.copy()
        job = order.pop(i)
        order.insert(j, job)
        obj, _, _, _ = schedule_by_order(p, d, order)
        if obj + 1e-9 < best_obj:
            best_obj = obj
            best_order = order
            improved = True
    return best_order, best_obj


# -----------------------
# Gurobi model builder + solve
# -----------------------
def build_and_solve_gurobi(p: List[int],
                           d: List[int],
                           time_limit: int = 300,
                           mip_gap: float = 0.01,
                           threads: int = 0,
                           use_mip_start: bool = True,
                           use_local_search_start: bool = True,
                           verbose: bool = True):
    n = len(p)
    sumP = sum(p)
    # Tight Big-M: using sumP is acceptable here (smaller than sumP + max(d))
    M = float(sumP)

    model = gp.Model("single_machine_total_tardiness")
    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPGap', mip_gap)
    model.setParam('MIPFocus', 1)    # focus on finding good feasible solutions
    model.setParam('Heuristics', 0.5)
    model.setParam('Presolve', 2)
    if threads and threads > 0:
        model.setParam('Threads', threads)
    # Silence output if not verbose
    if not verbose:
        model.setParam('OutputFlag', 0)

    # Variables
    S = model.addVars(n, lb=0.0, name="S")
    T = model.addVars(n, lb=0.0, name="T")
    y = {}
    for i in range(n):
        for j in range(i + 1, n):
            y[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")
    model.update()

    # Disjunctive ordering constraints
    for i in range(n):
        for j in range(i + 1, n):
            model.addConstr(S[i] + p[i] <= S[j] + M * (1 - y[(i, j)]), name=f"ord1_{i}_{j}")
            model.addConstr(S[j] + p[j] <= S[i] + M * (y[(i, j)]), name=f"ord2_{i}_{j}")

    # Tardiness constraints
    for i in range(n):
        model.addConstr(T[i] >= S[i] + p[i] - d[i], name=f"tardy_{i}")

    # Objective
    model.setObjective(gp.quicksum(T[i] for i in range(n)), GRB.MINIMIZE)

    # Build SPT start
    spt_order = sorted(range(n), key=lambda j: p[j])
    if use_local_search_start:
        # try to improve SPT by insertion local search
        spt_order, spt_obj = local_search_insertion(p, d, spt_order, max_iters=1000)
    # prepare position map
    pos = [0] * n
    for idx, job in enumerate(spt_order):
        pos[job] = idx

    # Set MIP start for y, S, T
    if use_mip_start:
        for i in range(n):
            for j in range(i + 1, n):
                # y_ij = 1 if i before j in spt_order
                y[(i, j)].Start = 1.0 if pos[i] < pos[j] else 0.0
        # S and T starts
        time_acc = 0.0
        for job in spt_order:
            S[job].Start = time_acc
            time_acc += p[job]
            T[job].Start = max(0.0, S[job].Start + p[job] - d[job])

    # Optimize
    t0 = time.time()
    model.optimize()
    t_elapsed = time.time() - t0

    status = model.Status
    result = {
        "status": status,
        "time": t_elapsed,
        "model": model
    }

    if status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL, GRB.INTERRUPTED):
        # Extract solution if available
        if model.SolCount > 0:
            S_sol = [S[i].X for i in range(n)]
            T_sol = [T[i].X for i in range(n)]
            order = sorted(range(n), key=lambda i: S_sol[i])
            total_tardy = sum(T_sol)
            result.update({
                "S": S_sol,
                "T": T_sol,
                "order": order,
                "total_tardy": total_tardy,
                "obj": model.ObjVal
            })
    return result


# -----------------------
# Main execution
# -----------------------
def main():
    # Parameters - 바꿔써도 됨
    n = 100
    seed = 2025
    # target tardy fraction for SPT ~ 0.45 (아슬아슬)
    target_frac = 0.45
    time_limit = 300        # seconds (바꾸려면 여기서 변경; 교수님은 1800 권장)
    mip_gap = 0.01
    threads = 0             # 0 => Gurobi auto choose; set >0 to restrict

    # Generate instance with tuned tightness
    p, d, tightness, observed_frac = tune_tightness_for_target(n=n, target_frac=target_frac, tol=0.08, max_iters=25, seed=seed)
    print(f"Generated instance: n={n}, tightness={tightness:.3f}, SPT tardy_frac≈{observed_frac:.3f}")
    print(f"sum p = {sum(p)}, sample d = {d[:8]}")

    # Baselines
    spt_obj, spt_S, spt_C, spt_T = spt_total_tardiness(p, d)
    edd_obj, edd_S, edd_C, edd_T = edd_total_tardiness(p, d)
    print(f"SPT total tardiness = {int(spt_obj)}")
    print(f"EDD total tardiness = {int(edd_obj)}")

    # Solve with Gurobi (with SPT MIP start + local search start)
    sol = build_and_solve_gurobi(p, d, time_limit=time_limit, mip_gap=mip_gap, threads=threads,
                                 use_mip_start=True, use_local_search_start=True, verbose=True)

    print("Gurobi status:", sol.get("status"))
    if "total_tardy" in sol:
        print(f"Gurobi solution total tardiness = {sol['total_tardy']:.2f}")
        print("First 20 jobs in Gurobi order:", sol['order'][:20])
    else:
        print("Gurobi did not return a solution (or no feasible solution found). Check logs.")

    # Save results CSV for the best solution (if found)
    if "order" in sol:
        order = sol["order"]
        # Recompute schedule details for that order
        _, S_sol, C_sol, T_sol = schedule_by_order(p, d, order)
        df = pd.DataFrame({
            "job": order,
            "p": [p[j] for j in order],
            "d": [d[j] for j in order],
            "S": [S_sol[j] for j in order],
            "C": [C_sol[j] for j in order],
            "T": [T_sol[j] for j in order]
        })
        csv_name = "gurobi_solution_order.csv"
        df.to_csv(csv_name, index=False)
        print(f"Solution saved to {csv_name}")
    else:
        # Save instance only
        inst_df = pd.DataFrame({"job": list(range(n)), "p": p, "d": d})
        inst_name = "instance_p_d.csv"
        inst_df.to_csv(inst_name, index=False)
        print(f"Instance saved to {inst_name}")


if __name__ == "__main__":
    main()
