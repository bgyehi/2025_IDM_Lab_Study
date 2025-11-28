#7차과제

import random
import time
from gurobipy import Model, GRB

# ==========================================================
# 0. 파라미터 & 고정 Job 데이터
# ==========================================================
job_num = 100
seed = 42
TIME_LIMIT_IG = 1800
TIME_LIMIT_LS = 1800
TIME_LIMIT_GUROBI = 1800

random.seed(seed)

jobs_data = [
    (1, 45, 532), (2, 12, 986), (3, 6, 994), (4, 22, 257),
    (5, 20, 434), (6, 19, 232), (7, 13, 523), (8, 11, 610),
    (9, 48, 474), (10, 39, 267), (11, 10, 416), (12, 42, 780),
    (13, 32, 935), (14, 7, 522), (15, 6, 417), (16, 10, 871),
    (17, 18, 711), (18, 19, 605), (19, 37, 858), (20, 43, 669),
    (21, 6, 346), (22, 40, 471), (23, 17, 342), (24, 50, 452),
    (25, 46, 962), (26, 49, 774), (27, 39, 751), (28, 31, 469),
    (29, 19, 964), (30, 33, 798), (31, 42, 638), (32, 22, 797),
    (33, 5, 608), (34, 15, 570), (35, 49, 424), (36, 32, 341),
    (37, 26, 721), (38, 22, 705), (39, 14, 293), (40, 18, 973),
    (41, 26, 248), (42, 11, 312), (43, 10, 356), (44, 29, 842),
    (45, 11, 363), (46, 27, 896), (47, 27, 632), (48, 43, 810),
    (49, 21, 265), (50, 7, 594), (51, 34, 590), (52, 39, 810),
    (53, 12, 679), (54, 29, 741), (55, 10, 457), (56, 40, 766),
    (57, 23, 211), (58, 45, 896), (59, 44, 938), (60, 28, 317),
    (61, 41, 898), (62, 17, 749), (63, 50, 968), (64, 9, 473),
    (65, 7, 987), (66, 47, 856), (67, 19, 548), (68, 23, 314),
    (69, 10, 500), (70, 19, 645), (71, 11, 361), (72, 29, 664),
    (73, 22, 203), (74, 34, 939), (75, 45, 936), (76, 28, 469),
    (77, 15, 712), (78, 28, 980), (79, 27, 382), (80, 18, 719),
    (81, 47, 308), (82, 22, 840), (83, 49, 505), (84, 48, 854),
    (85, 46, 719), (86, 9, 823), (87, 43, 403), (88, 45, 356),
    (89, 15, 582), (90, 39, 980), (91, 20, 365), (92, 15, 752),
    (93, 34, 997), (94, 29, 743), (95, 22, 200), (96, 45, 813),
    (97, 49, 531), (98, 40, 700), (99, 19, 219), (100, 48, 314)
]

# 문자열 Job ID, dict p, d (언니 코드와 같은 형태)
jobs = [f"J{i:03d}" for i in range(1, job_num + 1)]
p = {jobs[i]: jobs_data[i][1] for i in range(job_num)}
d = {jobs[i]: jobs_data[i][2] for i in range(job_num)}

# ==========================================================
# 유틸: 총 지연 계산
# ==========================================================
def compute_total_tardiness(order, p, d):
    t = 0
    tot = 0
    for j in order:
        t += p[j]
        tot += max(0, t - d[j])
    return tot

# ==========================================================
# 초기해: SPT / EDD
# ==========================================================
def initial_solution(p, d, mode="SPT"):
    job_list = list(p.keys())
    if mode == "SPT":
        return sorted(job_list, key=lambda j: p[j])
    elif mode == "EDD":
        return sorted(job_list, key=lambda j: d[j])
    else:
        return job_list

# ==========================================================
# IG components (destruction / construction / local search)
# ==========================================================
def destruction_phase(sequence, d_size):
    seq = sequence.copy()
    idxs = sorted(random.sample(range(len(seq)), d_size), reverse=True)
    removed = [seq[i] for i in idxs]
    for i in idxs:
        seq.pop(i)
    return seq, removed

def construction_phase(partial_seq, jobs_to_insert, p, d):
    seq = partial_seq.copy()
    for job in jobs_to_insert:
        best_seq = None
        best_cost = float('inf')
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [job] + seq[pos:]
            c = compute_total_tardiness(new_seq, p, d)
            if c < best_cost:
                best_cost = c
                best_seq = new_seq
        seq = best_seq
    return seq

def local_search(sequence, p, d, time_limit_sec=10):  # 빠른 LS (IG 내부용)
    start = time.time()
    seq = sequence.copy()
    best_cost = compute_total_tardiness(seq, p, d)
    while time.time() - start < time_limit_sec:
        improved = False
        for i in range(len(seq)):
            for j in range(len(seq)):
                if i == j:
                    continue
                new_seq = seq.copy()
                job = new_seq.pop(i)
                new_seq.insert(j, job)
                c = compute_total_tardiness(new_seq, p, d)
                if c < best_cost:
                    seq = new_seq
                    best_cost = c
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return seq, best_cost

def iterated_greedy(p, d, destruction_size=5, init_mode="SPT", time_limit=TIME_LIMIT_IG):
    start = time.time()
    current_seq = initial_solution(p, d, init_mode)
    best_seq = current_seq[:]
    best_cost = compute_total_tardiness(best_seq, p, d)
    while time.time() - start < time_limit:
        partial, removed = destruction_phase(current_seq, destruction_size)
        new_seq = construction_phase(partial, removed, p, d)
        # 간단 LS 적용 (빠르게)
        new_seq, new_cost = local_search(new_seq, p, d, time_limit_sec=1)
        if new_cost < best_cost:
            best_seq = new_seq[:]
            best_cost = new_cost
            current_seq = new_seq[:]
        else:
            if random.random() < 0.1:
                current_seq = new_seq[:]
    return best_seq, best_cost, time.time() - start

# ==========================================================
# Gurobi 모델 (문자열 Job ID 기반) — gap 출력 포함
# ==========================================================
def solve_with_gurobi(p, d, time_limit_sec=TIME_LIMIT_GUROBI):
    job_ids = list(p.keys())

    model = Model("single_machine_tardiness")

    # 옵션: gap 개선에 도움되는 설정들
    model.setParam("TimeLimit", time_limit_sec)
    model.setParam("MIPFocus", 1)      # bound 개선 집중
    model.setParam("Cuts", 2)
    model.setParam("Presolve", 2)
    model.setParam("Heuristics", 0.2)
    # model.setParam("Threads", 4)     # 필요시 설정 (환경에 따라)

    # 변수
    C = model.addVars(job_ids, lb=0.0, vtype=GRB.CONTINUOUS, name="C")
    T = model.addVars(job_ids, lb=0.0, vtype=GRB.CONTINUOUS, name="T")
    y = model.addVars(job_ids, job_ids, vtype=GRB.BINARY, name="y")

    # 제약: 자기자신 y[i,i] = 0
    for j in job_ids:
        y[j, j].ub = 0

    # 완료시간 정의: Cj = pj + sum(pi * y[i,j])  (이 식은 완전 순서 표현)
    for j in job_ids:
        model.addConstr(C[j] == p[j] + sum(p[i] * y[i, j] for i in job_ids if i != j), name=f"completion[{j}]")

    # 순서 쌍대성: for each i != j, y[i,j] + y[j,i] == 1
    for i in job_ids:
        for j in job_ids:
            if i != j:
                model.addConstr(y[i, j] + y[j, i] == 1, name=f"precedence[{i},{j}]")

    # Big-M 비겹침 (완료시간 기반)
    M = sum(p.values())
    for i in job_ids:
        for j in job_ids:
            if i == j:
                continue
            model.addConstr(C[i] <= C[j] - p[j] + M * (1 - y[i, j]), name=f"nonoverlap1[{i},{j}]")
            model.addConstr(C[j] <= C[i] - p[i] + M * (y[i, j]),     name=f"nonoverlap2[{i},{j}]")

    # tardiness 정의
    for j in job_ids:
        model.addConstr(T[j] >= C[j] - d[j], name=f"tard1[{j}]")
        model.addConstr(T[j] >= 0,           name=f"tard2[{j}]")

    # 목적
    model.setObjective(sum(T[j] for j in job_ids), GRB.MINIMIZE)

    # 최적화
    model.optimize()

    # 결과 수집
    status = model.Status
    obj = model.ObjVal if model.SolCount > 0 else None
    bound = model.ObjBound if hasattr(model, "ObjBound") else None
    gap = model.MIPGap if hasattr(model, "MIPGap") else None
    runtime = model.Runtime

    # 완료시간 기반 순서 복원 (정수해가 아닐 경우에도 C.X로 정렬)
    order = []
    if model.SolCount > 0:
        order = sorted(job_ids, key=lambda j: C[j].X)

    return {
        "model": model,
        "status": status,
        "obj": obj,
        "bound": bound,
        "gap": gap,
        "runtime": runtime,
        "order": order
    }
# ==========================================================
# 실행 및 출력 포맷
# ==========================================================
if __name__ == "__main__":
    # IG
    print("\n=== Iterated Greedy (IG) ===")
    ig_seq, ig_tard, ig_time = iterated_greedy(p, d, destruction_size=5, init_mode="SPT", time_limit=5)  # 데모용 5초
    print(f"IG Tardiness = {ig_tard}, time = {ig_time:.2f} 초")

    # Local Search (from SPT)
    print("\n=== Local Search (from SPT) ===")
    base = initial_solution(p, d, "SPT")
    ls_seq, ls_tard = local_search(base, p, d, time_limit_sec=5)  # 데모용 5초
    print(f"Local Search Tardiness = {ls_tard}")

    # SPT / EDD
    spt_order = initial_solution(p, d, "SPT")
    edd_order = initial_solution(p, d, "EDD")
    spt_tard = compute_total_tardiness(spt_order, p, d)
    edd_tard = compute_total_tardiness(edd_order, p, d)

    # Gurobi
    print("\n=== Gurobi ===")
    res = solve_with_gurobi(p, d, time_limit_sec=TIME_LIMIT_GUROBI)

    # Gurobi 출력
    print("\n=== Gurobi Result ===")
    if res["status"] == GRB.OPTIMAL:
        print("최적해를 찾았습니다.")
    elif res["status"] == GRB.TIME_LIMIT:
        print("Time limit reached")
    else:
        print("Gurobi status:", res["status"])

    if res["obj"] is not None:
        print(f"Gurobi 총 Tardiness = {res['obj']}")
        if res["bound"] is not None:
            print(f"best bound = {res['bound']}")
        if res["gap"] is not None:
            print(f"gap = {res['gap']*100:.4f}%")
        print(f"Gurobi 계산 시간 = {res['runtime']:.2f} 초")
        # 순서(앞 20개)
        print("Gurobi 순서 (앞 20개):", " → ".join(res["order"][:20]))
    else:
        print("최적해(또는 feasible 해)를 찾지 못했습니다.")

    # Heuristics
    print("\n=== Heuristic Rules (Same Instance) ===")
    print(f"SPT 총 Tardiness = {spt_tard},   시간 = 0.0000 초")
    print(f"EDD 총 Tardiness = {edd_tard},   시간 = 0.0000 초")

    # 요약
    print("\n================ Summary ================")
    print(f"SPT Heuristic   : Tardiness = {spt_tard}")
    print(f"EDD Heuristic   : Tardiness = {edd_tard}")
    print(f"Local Search    : Tardiness = {ls_tard}")
    print(f"IG              : Tardiness = {ig_tard}")
    print(f"Gurobi          : Tardiness = {res['obj']}, runtime = {res['runtime']:.2f} sec, status = {res['status']}")
    print("=========================================")
