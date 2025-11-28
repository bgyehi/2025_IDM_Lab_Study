#5차과제

import random
import itertools
import time
import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    raise ImportError("gurobipy가 설치되어 있지 않습니다. Gurobi + Python API 설치 및 라이선스 필요.")


# ==============================================
# 인스턴스 생성 함수
# ==============================================
def generate_instance(n=100, p_low=1, p_high=100, tightness=0.45, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    p = [random.randint(p_low, p_high) for _ in range(n)]
    sumP = sum(p)
    center = sumP / 2.0
    spread = sumP * tightness
    d = [int(round(random.uniform(center - spread / 2, center + spread / 2))) + pj for pj in p]
    return p, d


def tune_tightness_for_target(n=100, target_frac=0.5, tol=0.05, max_iters=20, seed=123):
    low, high = 0.05, 1.0
    for it in range(max_iters):
        mid = (low + high) / 2.0
        p, d = generate_instance(n=n, tightness=mid, seed=seed + it)
        order = sorted(range(n), key=lambda j: p[j])
        time_sum = 0
        tardy_count = 0
        for j in order:
            time_sum += p[j]
            if time_sum > d[j]:
                tardy_count += 1
        frac = tardy_count / n
        if abs(frac - target_frac) <= tol:
            return p, d, mid, frac
        if frac > target_frac:
            low = mid
        else:
            high = mid
    return p, d, mid, frac


# ==============================================
# 휴리스틱 평가 함수
# ==============================================
def total_tardiness(sequence, p, d):
    time_sum = 0
    tardiness = 0
    for job in sequence:
        time_sum += p[job]
        tardiness += max(0, time_sum - d[job])
    return tardiness


# ==============================================
# 로컬서치 연산자들
# ==============================================
def swap(seq, i, j):
    new_seq = seq[:]
    new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
    return new_seq


def swap_full_search(seq, p, d):
    best_seq = seq[:]
    best_tardiness = total_tardiness(best_seq, p, d)
    for i, j in itertools.combinations(range(len(seq)), 2):
        new_seq = swap(best_seq, i, j)
        new_tardiness = total_tardiness(new_seq, p, d)
        if new_tardiness < best_tardiness:
            best_seq = new_seq
            best_tardiness = new_tardiness
    return best_seq, best_tardiness


def insert(seq, i, j):
    new_seq = seq[:]
    job = new_seq.pop(i)
    new_seq.insert(j, job)
    return new_seq


def insert_full_search(seq, p, d):
    best_seq = seq[:]
    best_tardiness = total_tardiness(best_seq, p, d)
    for i in range(len(seq)):
        for j in range(len(seq)):
            if i != j:
                new_seq = insert(best_seq, i, j)
                new_tardiness = total_tardiness(new_seq, p, d)
                if new_tardiness < best_tardiness:
                    best_seq = new_seq
                    best_tardiness = new_tardiness
    return best_seq, best_tardiness


def block_reverse(seq, i, j):
    new_seq = seq[:]
    new_seq[i:j + 1] = reversed(new_seq[i:j + 1])
    return new_seq


def block_full_search(seq, p, d):
    best_seq = seq[:]
    best_tardiness = total_tardiness(best_seq, p, d)
    for i in range(len(seq) - 1):
        for j in range(i + 1, len(seq)):
            new_seq = block_reverse(best_seq, i, j)
            new_tardiness = total_tardiness(new_seq, p, d)
            if new_tardiness < best_tardiness:
                best_seq = new_seq
                best_tardiness = new_tardiness
    return best_seq, best_tardiness


# ==============================================
# Gurobi 모델 (강화버전)
# ==============================================
def build_and_solve_gurobi(p, d, time_limit=1800, mip_gap=0.01, threads=8):
    n = len(p)
    sumP = sum(p)
    M = sumP + max(d)
    model = gp.Model("single_machine_total_tardiness")

    # ---- 파라미터 설정 (성능 강화) ----
    model.setParam("TimeLimit", time_limit)
    model.setParam("MIPGap", mip_gap)
    model.setParam("MIPFocus", 3)        # Bound 개선 중심
    model.setParam("Heuristics", 0.2)    # 휴리스틱 탐색 강화
    model.setParam("Cuts", 2)             # 절단평면 강화
    model.setParam("Presolve", 2)         # 최대 사전 단순화
    model.setParam("Threads", threads)    # 멀티코어 병렬 처리

    # ---- 변수 정의 ----
    S = model.addVars(n, lb=0.0, name="S")   # 시작시간
    T = model.addVars(n, lb=0.0, name="T")   # 지각시간
    y = {}
    for i in range(n):
        for j in range(i + 1, n):
            y[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")

    model.update()

    # ---- 제약조건 ----
    for i in range(n):
        for j in range(i + 1, n):
            model.addConstr(S[i] + p[i] <= S[j] + M * (1 - y[(i, j)]))
            model.addConstr(S[j] + p[j] <= S[i] + M * (y[(i, j)]))

    for i in range(n):
        model.addConstr(T[i] >= S[i] + p[i] - d[i])

    # ---- 목적함수 ----
    model.setObjective(gp.quicksum(T[i] for i in range(n)), GRB.MINIMIZE)

    # ---- 최적화 ----
    model.optimize()

    status = model.Status
    sol = {}
    if status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL, GRB.INTERRUPTED):
        S_sol = [S[i].X for i in range(n)]
        T_sol = [T[i].X for i in range(n)]
        order = sorted(range(n), key=lambda i: S_sol[i])
        total_tardy = sum(T_sol)
        sol = {
            "status": status,
            "obj": model.ObjVal if model.SolCount > 0 else None,
            "S": S_sol,
            "T": T_sol,
            "order": order,
            "total_tardy": total_tardy,
            "best_bound": model.ObjBound,
            "gap": model.MIPGap * 100,
            "runtime": model.Runtime,
        }
    else:
        sol = {"status": status, "message": "No feasible solution found or solver error."}
    return sol


# ==============================================
# 실행부
# ==============================================
if __name__ == "__main__":
    n = 100
    p, d, tightness, frac = tune_tightness_for_target(n=n, target_frac=0.5, tol=0.08, max_iters=25, seed=2025)
    print(f"\n[인스턴스 생성 완료] n={n}, tightness={tightness:.3f}, SPT tardy_frac≈{frac:.3f}\n")

    jobs = list(range(n))
    random.shuffle(jobs)
    print("초기 랜덤 시퀀스 생성 완료.")

    # --- Swap Local Search ---
    start = time.time()
    best_swap_seq, best_swap_tardiness = swap_full_search(jobs, p, d)
    swap_time = time.time() - start
    print(f"[Swap] Best Tardiness = {best_swap_tardiness}, Time = {swap_time:.2f} sec")

    # --- Insert Local Search ---
    start = time.time()
    best_insert_seq, best_insert_tardiness = insert_full_search(jobs, p, d)
    insert_time = time.time() - start
    print(f"[Insert] Best Tardiness = {best_insert_tardiness}, Time = {insert_time:.2f} sec")

    # --- Block Reverse Local Search ---
    start = time.time()
    best_block_seq, best_block_tardiness = block_full_search(jobs, p, d)
    block_time = time.time() - start
    print(f"[Block Reverse] Best Tardiness = {best_block_tardiness}, Time = {block_time:.2f} sec")

    # --- Gurobi Solver ---
    print("\n[Gurobi 최적화 시작...]")
    start = time.time()
    sol = build_and_solve_gurobi(p, d, time_limit=1800, mip_gap=0.01, threads=8)
    gurobi_time = time.time() - start

    print("\n================ Gurobi Summary ================")
    print(f"Status        : {sol.get('status')}")
    if "total_tardy" in sol:
        print(f"Objective     : {sol['obj']:.2f}")
        print(f"Best Bound    : {sol['best_bound']:.2f}")
        print(f"Gap           : {sol['gap']:.2f}%")
        print(f"Total Tardiness : {sol['total_tardy']:.2f}")
        print(f"Runtime       : {sol['runtime']:.2f} sec (Measured: {gurobi_time:.2f} sec)")
    else:
        print(sol)

    print("================================================")
