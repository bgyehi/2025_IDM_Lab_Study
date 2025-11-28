# ==========================================================
# Single Machine Total Tardiness Problem (100 jobs)
# Gurobi + Heuristics + Local Search (Swap / Insert / Block)
# ==========================================================

import random
import time
import itertools
import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    raise ImportError("⚠️ gurobipy가 설치되어 있지 않습니다. Gurobi 라이선스와 Python API를 설치하세요.")


# ----------------------------------------------------------
# 1️⃣ 인스턴스 생성 함수
# ----------------------------------------------------------
def generate_instance(n=100, p_low=1, p_high=20, tightness=0.5, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    p = [random.randint(p_low, p_high) for _ in range(n)]
    total_p = sum(p)
    spread = int(total_p * tightness)
    d = [random.randint(spread // 3, spread) + pj for pj in p]
    return p, d


# ----------------------------------------------------------
# 2️⃣ Tardiness 계산 함수
# ----------------------------------------------------------
def total_tardiness(order, p, d):
    time_now = 0
    total = 0
    for j in order:
        time_now += p[j]
        total += max(0, time_now - d[j])
    return total


# ----------------------------------------------------------
# 3️⃣ Gurobi 최적화 모델
# ----------------------------------------------------------
def solve_gurobi(p, d, time_limit=600, mip_gap=0.01):
    n = len(p)
    M = sum(p) + max(d)
    m = gp.Model("SingleMachine_Tardiness")
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = mip_gap
    m.Params.OutputFlag = 1

    # 변수
    S = m.addVars(n, lb=0.0, name="Start")
    T = m.addVars(n, lb=0.0, name="Tardy")
    y = {}
    for i in range(n):
        for j in range(i + 1, n):
            y[(i, j)] = m.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")

    m.update()

    # 제약식
    for i in range(n):
        for j in range(i + 1, n):
            m.addConstr(S[i] + p[i] <= S[j] + M * (1 - y[(i, j)]))
            m.addConstr(S[j] + p[j] <= S[i] + M * (y[(i, j)]))
        m.addConstr(T[i] >= S[i] + p[i] - d[i])

    # 목적함수
    m.setObjective(gp.quicksum(T[i] for i in range(n)), GRB.MINIMIZE)

    start = time.time()
    m.optimize()
    elapsed = time.time() - start

    if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        S_sol = [S[i].X for i in range(n)]
        order = sorted(range(n), key=lambda i: S_sol[i])
        tardiness = sum(T[i].X for i in range(n))
        return {"status": m.status, "obj": tardiness, "order": order, "time": elapsed}
    else:
        return {"status": m.status, "obj": None, "order": None, "time": elapsed}


# ----------------------------------------------------------
# 4️⃣ Heuristic Rules (SPT / EDD)
# ----------------------------------------------------------
def spt_rule(p, d):
    order = sorted(range(len(p)), key=lambda j: p[j])
    return order, total_tardiness(order, p, d)

def edd_rule(p, d):
    order = sorted(range(len(p)), key=lambda j: d[j])
    return order, total_tardiness(order, p, d)


# ----------------------------------------------------------
# 5️⃣ Local Search Operators
# ----------------------------------------------------------
def swap(seq, i, j):
    new_seq = seq[:]
    new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
    return new_seq

def insert(seq, i, j):
    new_seq = seq[:]
    job = new_seq.pop(i)
    new_seq.insert(j, job)
    return new_seq

def block_reverse(seq, i, j):
    new_seq = seq[:]
    new_seq[i:j+1] = reversed(new_seq[i:j+1])
    return new_seq


# ----------------------------------------------------------
# 6️⃣ Local Search Methods
# ----------------------------------------------------------
def swap_local_search(seq, p, d):
    best_seq = seq[:]
    best_val = total_tardiness(best_seq, p, d)
    improved = True

    while improved:
        improved = False
        for i, j in itertools.combinations(range(len(seq)), 2):
            new_seq = swap(best_seq, i, j)
            new_val = total_tardiness(new_seq, p, d)
            if new_val < best_val:
                best_seq, best_val = new_seq, new_val
                improved = True
    return best_seq, best_val


def insert_local_search(seq, p, d):
    best_seq = seq[:]
    best_val = total_tardiness(best_seq, p, d)
    improved = True

    while improved:
        improved = False
        for i in range(len(seq)):
            for j in range(len(seq)):
                if i != j:
                    new_seq = insert(best_seq, i, j)
                    new_val = total_tardiness(new_seq, p, d)
                    if new_val < best_val:
                        best_seq, best_val = new_seq, new_val
                        improved = True
    return best_seq, best_val


def block_reverse_local_search(seq, p, d, min_block=2, max_block=10):
    best_seq = seq[:]
    best_val = total_tardiness(best_seq, p, d)
    improved = True

    while improved:
        improved = False
        for i in range(len(seq)):
            for l in range(min_block, max_block + 1):
                j = i + l - 1
                if j >= len(seq):
                    break
                new_seq = block_reverse(best_seq, i, j)
                new_val = total_tardiness(new_seq, p, d)
                if new_val < best_val:
                    best_seq, best_val = new_seq, new_val
                    improved = True
    return best_seq, best_val


# ----------------------------------------------------------
# 7️⃣ 메인 실행
# ----------------------------------------------------------
if __name__ == "__main__":
    # 인스턴스 생성
    n = 100
    p, d = generate_instance(n, tightness=0.6, seed=42)
    print(f"=== Generated 100-Job Instance ===")
    print(f"sum(p)={sum(p)}, sample due={d[:5]}\n")

    # Gurobi 실행
    print("=== Gurobi Optimization ===")
    gurobi_result = solve_gurobi(p, d, time_limit=600)
    print(f"Status = {gurobi_result['status']}, Obj = {gurobi_result['obj']}, Time = {gurobi_result['time']:.2f}s\n")

    # Heuristic
    print("=== Heuristic Rules ===")
    spt_order, spt_val = spt_rule(p, d)
    edd_order, edd_val = edd_rule(p, d)
    print(f"SPT: {spt_val}, EDD: {edd_val}\n")

    # Local Search
    print("=== Local Search (from EDD) ===")
    start = time.time()
    swap_order, swap_val = swap_local_search(edd_order, p, d)
    print(f"Swap  → Tardiness = {swap_val},  Time = {time.time() - start:.2f}s")

    start = time.time()
    insert_order, insert_val = insert_local_search(edd_order, p, d)
    print(f"Insert → Tardiness = {insert_val}, Time = {time.time() - start:.2f}s")

    start = time.time()
    block_order, block_val = block_reverse_local_search(edd_order, p, d)
    print(f"Block → Tardiness = {block_val},  Time = {time.time() - start:.2f}s")

    # Summary
    print("\n================ Summary ================")
    print(f"Gurobi       : {gurobi_result['obj']}")
    print(f"SPT          : {spt_val}")
    print(f"EDD          : {edd_val}")
    print(f"Swap (EDD)   : {swap_val}")
    print(f"Insert (EDD) : {insert_val}")
    print(f"Block (EDD)  : {block_val}")
    print("========================================")
