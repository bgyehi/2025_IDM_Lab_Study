# test_scheduling.py
# PyCharm에 붙여넣고 실행하세요.
# 주의: 파일명을 gurobi.py로 하면 안 됩니다.

from gurobipy import Model, GRB
import itertools
import random
import math
import matplotlib.pyplot as plt

# ---------------------------
# 헬퍼: 총 tardiness 계산
# ---------------------------
def compute_tardiness_from_sequence(seq, P, d):
    """seq: list of job indices (0-based). P, d: lists"""
    time = 0
    total_tard = 0
    C = {}
    for j in seq:
        time += P[j]
        C[j] = time
        tard = max(0, C[j] - d[j])
        total_tard += tard
    return total_tard, C

# ---------------------------
# Gurobi 모델 생성 함수
# ---------------------------
def build_and_solve_gurobi(P, d, time_limit=None, verbose=True):
    n = len(P)
    M = sum(P)
    jobs = list(range(n))

    m = Model("single_machine_tardiness")
    if not verbose:
        m.Params.LogToConsole = 0

    # 변수
    C = {j: m.addVar(lb=0.0, ub=M, vtype=GRB.CONTINUOUS, name=f"C_{j}") for j in jobs}
    T = {j: m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"T_{j}") for j in jobs}
    y = {}
    for i in jobs:
        for j in jobs:
            if i == j:
                continue
            y[i, j] = m.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")

    m.update()

    # 목적: sum T_j
    m.setObjective(sum(T[j] for j in jobs), GRB.MINIMIZE)

    # 제약: 쌍마다 한쪽이 먼저
    for i in jobs:
        for j in jobs:
            if i == j:
                continue
            m.addConstr(y[i, j] + y[j, i] == 1, name=f"order_{i}_{j}")

    # 완료시간 관계: C_j >= C_i + P_i  (if y[i,j]==1)
    for i in jobs:
        for j in jobs:
            if i == j:
                continue
            # C_j >= C_i + P_i - M*(1 - y[i,j])
            m.addConstr(C[j] >= C[i] + P[i] - M * (1 - y[i, j]),
                        name=f"comp_rel_{i}_{j}")

    # tardiness 정의
    for j in jobs:
        m.addConstr(T[j] >= C[j] - d[j], name=f"tard_def_{j}")
        m.addConstr(T[j] >= 0, name=f"tard_nonneg_{j}")

    # optional: C_j >= P_j (각 job 최소 완료시간)
    for j in jobs:
        m.addConstr(C[j] >= P[j], name=f"comp_lb_{j}")

    # 시간제한이 필요하면 설정
    if time_limit is not None:
        m.Params.TimeLimit = time_limit

    m.optimize()

    if m.Status == GRB.OPTIMAL or m.Status == GRB.TIME_LIMIT:
        # 완료시간 기준으로 순서 추출 (작업의 실제 순서는 C값 오름차순)
        Cvals = {j: C[j].X for j in jobs}
        sorted_seq = sorted(jobs, key=lambda j: Cvals[j])
        obj = sum(T[j].X for j in jobs)
        return {
            "model": m,
            "objective": obj,
            "C": Cvals,
            "sequence": sorted_seq,
            "T": {j: T[j].X for j in jobs},
            "status": m.Status
        }
    else:
        return {"model": m, "status": m.Status}

# ---------------------------
# 완전탐색(enumeration)
# ---------------------------
def enumerate_optimal(P, d, max_n_for_enum=10):
    n = len(P)
    if n > max_n_for_enum:
        raise ValueError(f"n={n} -> enumeration may be too slow (limit {max_n_for_enum}).")
    best_seq = None
    best_val = math.inf
    for perm in itertools.permutations(range(n)):
        val, _ = compute_tardiness_from_sequence(perm, P, d)
        if val < best_val:
            best_val = val
            best_seq = perm
    return best_val, list(best_seq)

# ---------------------------
# Heuristics: EDD, SPT
# ---------------------------
def edd_rule(P, d):
    # EDD: sort by due date ascending
    seq = sorted(range(len(P)), key=lambda j: d[j])
    val, C = compute_tardiness_from_sequence(seq, P, d)
    return {"sequence": seq, "tardiness": val, "C": C}

def spt_rule(P, d):
    # SPT: sort by processing time ascending
    seq = sorted(range(len(P)), key=lambda j: P[j])
    val, C = compute_tardiness_from_sequence(seq, P, d)
    return {"sequence": seq, "tardiness": val, "C": C}

# ---------------------------
# Gantt chart 그리기 (선택)
# ---------------------------
def plot_gantt(sequence, P, d, Cvals=None, filename="gantt.png"):
    """sequence: list of job indices; P: processing times; d: due dates"""
    fig, ax = plt.subplots(figsize=(8, 2 + 0.3 * len(sequence)))
    y = 10
    start = 0
    labels = []
    for j in sequence:
        ax.barh(0, P[j], left=start, height=0.4, align='center')
        ax.text(start + P[j] / 2, 0, f"Job{j}", va='center', ha='center', color='white', fontsize=9)
        start += P[j]
    # due dates 표시
    for j in sequence:
        ax.axvline(d[j], color='k', linestyle='--', linewidth=0.8)
    ax.set_yticks([])
    ax.set_xlabel("Time")
    ax.set_title("Gantt chart (sequence order left->right)")
    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    print(f"Gantt chart saved to {filename}")

# ---------------------------
# 메인: 임의 데이터 생성 및 실행 예시
# ---------------------------
def main():
    # 랜덤 시드 고정 (재현성)
    random.seed(1)

    # === 임의 데이터 생성 ===
    n = 6  # 적당한 n: enumeration도 해보려면 n <= 9 권장
    P = [random.randint(1, 10) for _ in range(n)]
    # due date는 처리시간 총합의 일부 범위로 설정 (더 어렵게도 만들 수 있음)
    totalP = sum(P)
    d = [random.randint(0, totalP) for _ in range(n)]

    print("Jobs (index):", list(range(n)))
    print("Processing times P:", P)
    print("Due dates d:", d)
    print()

    # === Gurobi로 풀기 ===
    print("---- Solving with Gurobi (MIP) ----")
    res = build_and_solve_gurobi(P, d, verbose=True)
    if res.get("status") in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        print("Gurobi objective (sum T):", res["objective"])
        print("Sequence by completion time:", res["sequence"])
        print("Completion times C:", res["C"])
        print("Tardiness per job:", res["T"])
    else:
        print("Gurobi failed, status:", res.get("status"))

    print()

    # === Enumeration (완전탐색) 비교 ===
    try:
        best_val, best_seq = enumerate_optimal(P, d, max_n_for_enum=9)
        print("---- Enumeration (optimal) ----")
        print("Best enumeration tardiness:", best_val)
        print("Best enumeration sequence:", best_seq)
    except ValueError as e:
        print("Enumeration skipped:", e)

    print()

    # === Heuristics: EDD, SPT ===
    edd = edd_rule(P, d)
    spt = spt_rule(P, d)
    print("---- Heuristics ----")
    print("EDD tardiness:", edd["tardiness"], "seq:", edd["sequence"])
    print("SPT tardiness:", spt["tardiness"], "seq:", spt["sequence"])

    print()

    # === Gantt chart (선택) ===
    # 그리려면 matplotlib 설치 필요 (대부분 환경에 이미 설치됨).
    # 차트는 Gurobi가 반환한 sequence로 그림.
    seq_for_gantt = res["sequence"] if res.get("sequence") else edd["sequence"]
    plot_gantt(seq_for_gantt, P, d, filename="gantt.png")

if __name__ == "__main__":
    main()
