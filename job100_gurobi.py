import random
import math
import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    raise ImportError("gurobipy가 설치되어 있지 않습니다. Gurobi + Python API 설치 및 라이선스 필요.")

def generate_instance(n=100, p_low=1, p_high=100, tightness=0.45, seed=None):
    """
    n: job 수
    p_low, p_high: p_j의 균등분포 범위
    tightness: tau (0..1) — 클수록 due-date 퍼짐이 큼(=문제가 느슨해짐)
               보통 0.25~0.6 범위를 실험
    반환: p (list), d (list)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    p = [random.randint(p_low, p_high) for _ in range(n)]
    sumP = sum(p)
    center = sumP / 2.0
    spread = sumP * tightness  # 조절 가능
    d = [int(round(random.uniform(center - spread/2, center + spread/2))) + pj for pj in p]
    return p, d

def tune_tightness_for_target(n=100, target_frac=0.5, tol=0.05, max_iters=20, seed=123):
    """
    SPT 기준으로 tardy 비율이 target_frac에 가깝도록 tightness를 조절하는 간단한 이분탐색.
    결과는 (p, d, tightness)
    """
    low, high = 0.05, 1.0
    for it in range(max_iters):
        mid = (low + high) / 2.0
        p, d = generate_instance(n=n, tightness=mid, seed=seed+it)
        # SPT schedule
        order = sorted(range(n), key=lambda j: p[j])
        time = 0
        tardy_count = 0
        for j in order:
            time += p[j]
            if time > d[j]:
                tardy_count += 1
        frac = tardy_count / n
        # print(it, mid, frac)
        if abs(frac - target_frac) <= tol:
            return p, d, mid, frac
        if frac > target_frac:
            # too many tardy => make due-dates looser => increase tightness
            low = mid
        else:
            high = mid
    return p, d, mid, frac

def spt_total_tardiness(p, d):
    order = sorted(range(len(p)), key=lambda j: p[j])
    time = 0
    total_tardy = 0
    for j in order:
        time += p[j]
        total_tardy += max(0, time - d[j])
    return total_tardy, order

def edd_total_tardiness(p, d):
    order = sorted(range(len(p)), key=lambda j: d[j])
    time = 0
    total_tardy = 0
    for j in order:
        time += p[j]
        total_tardy += max(0, time - d[j])
    return total_tardy, order

def build_and_solve_gurobi(p, d, time_limit=1800, mip_gap=0.01, threads=0):
    """
    p, d: lists length n
    time_limit: seconds (e.g., 1800 = 30 minutes)
    returns: dict with model, solution S, T, objective, status
    """
    n = len(p)
    sumP = sum(p)
    M = sumP + max(d)  # 안전한 big-M (간단히 sumP + max duedate)
    model = gp.Model("single_machine_total_tardiness")
    # parameters
    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPGap', mip_gap)
    if threads > 0:
        model.setParam('Threads', threads)
    # variables
    S = model.addVars(n, lb=0.0, name="S")
    T = model.addVars(n, lb=0.0, name="T")
    y = {}  # y[i,j] for i<j
    for i in range(n):
        for j in range(i+1, n):
            y[(i,j)] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")
    model.update()
    # constraints: disjunctive ordering
    for i in range(n):
        for j in range(i+1, n):
            # if y_ij == 1 => i before j
            model.addConstr(S[i] + p[i] <= S[j] + M*(1 - y[(i,j)]), name=f"ord1_{i}_{j}")
            model.addConstr(S[j] + p[j] <= S[i] + M*(y[(i,j)]), name=f"ord2_{i}_{j}")
    # tardiness constraints
    for i in range(n):
        model.addConstr(T[i] >= S[i] + p[i] - d[i], name=f"tardy_{i}")
    # objective
    model.setObjective(gp.quicksum(T[i] for i in range(n)), GRB.MINIMIZE)
    # optimize
    model.update()
    model.optimize()
    status = model.Status
    sol = {}
    if status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL, GRB.INTERRUPTED):
        # extract start times and tardiness
        S_sol = [S[i].X for i in range(n)]
        T_sol = [T[i].X for i in range(n)]
        # derive sequence by sorting S
        order = sorted(range(n), key=lambda i: S_sol[i])
        total_tardy = sum(T_sol)
        sol = {
            "status": status,
            "obj": model.ObjVal if model.SolCount > 0 else None,
            "S": S_sol,
            "T": T_sol,
            "order": order,
            "total_tardy": total_tardy,
            "model": model
        }
    else:
        sol = {"status": status, "message": "No feasible solution found or solver error."}
    return sol

# === 실행 예제 ===
if __name__ == "__main__":
    n = 100
    # 자동 튜닝으로 '아슬아슬' 인스턴스 만들기 (목표 tardy 비율 약 0.5)
    p, d, tightness, frac = tune_tightness_for_target(n=n, target_frac=0.5, tol=0.08, max_iters=25, seed=2025)
    print(f"generated instance: n={n}, tightness={tightness:.3f}, SPT tardy_frac≈{frac:.3f}")
    print(f"sum p = {sum(p)}, d sample = {d[:8]}")

    # SPT / EDD 비교 (빠른 baseline)
    spt_obj, spt_order = spt_total_tardiness(p, d)
    edd_obj, edd_order = edd_total_tardiness(p, d)
    print(f"SPT total tardiness = {spt_obj}")
    print(f"EDD total tardiness = {edd_obj}")

    # Gurobi로 정식 최적화 (TimeLimit=1800초 권장 — 여기선 테스트로 300초로 둠)
    sol = build_and_solve_gurobi(p, d, time_limit=300, mip_gap=0.01, threads=0)
    print("Gurobi status:", sol.get("status"))
    if "total_tardy" in sol:
        print(f"Gurobi found solution total tardiness = {sol['total_tardy']:.2f}")
        # 간단한 순서 출력 (앞쪽 20개)
        print("first 20 jobs in Gurobi order:", sol['order'][:20])
    else:
        print(sol)
