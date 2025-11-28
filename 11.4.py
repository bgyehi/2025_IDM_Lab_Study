import random
import time
from gurobipy import Model, GRB
import numpy as np

# ==========================================================
# 0. 공통 파라미터 & 고정 Job 데이터
# ==========================================================
job_num = 100
seed = 42
TIME_LIMIT_IG = 1800
TIME_LIMIT_LS = 1800
TIME_LIMIT_GUROBI = 1800

random.seed(seed)

# -------------------------
# 고정된 100개 작업 데이터
# (job_id, processing time p, due date d)
# -------------------------
jobs_data = [
    (1, 45, 532), (2, 12, 986), (3, 6, 994), (4, 22, 257), (5, 20, 434),
    (6, 19, 232), (7, 13, 523), (8, 11, 610), (9, 48, 474), (10, 39, 267),
    (11, 10, 416), (12, 42, 780), (13, 32, 935), (14, 7, 522), (15, 6, 417),
    (16, 10, 871), (17, 18, 711), (18, 19, 605), (19, 37, 858), (20, 43, 669),
    (21, 6, 346), (22, 40, 471), (23, 17, 342), (24, 50, 452), (25, 46, 962),
    (26, 49, 774), (27, 39, 751), (28, 31, 469), (29, 19, 964), (30, 33, 798),
    (31, 42, 638), (32, 22, 797), (33, 5, 608), (34, 15, 570), (35, 49, 424),
    (36, 32, 341), (37, 26, 721), (38, 22, 705), (39, 14, 293), (40, 18, 973),
    (41, 26, 248), (42, 11, 312), (43, 10, 356), (44, 29, 842), (45, 11, 363),
    (46, 27, 896), (47, 27, 632), (48, 43, 810), (49, 21, 265), (50, 7, 594),
    (51, 34, 590), (52, 39, 810), (53, 12, 679), (54, 29, 741), (55, 10, 457),
    (56, 40, 766), (57, 23, 211), (58, 45, 896), (59, 44, 938), (60, 28, 317),
    (61, 41, 898), (62, 17, 749), (63, 50, 968), (64, 9, 473), (65, 7, 987),
    (66, 47, 856), (67, 19, 548), (68, 23, 314), (69, 10, 500), (70, 19, 645),
    (71, 11, 361), (72, 29, 664), (73, 22, 203), (74, 34, 939), (75, 45, 936),
    (76, 28, 469), (77, 15, 712), (78, 28, 980), (79, 27, 382), (80, 18, 719),
    (81, 47, 308), (82, 22, 840), (83, 49, 505), (84, 48, 854), (85, 46, 719),
    (86, 9, 823), (87, 43, 403), (88, 45, 356), (89, 15, 582), (90, 39, 980),
    (91, 20, 365), (92, 15, 752), (93, 34, 997), (94, 29, 743), (95, 22, 200),
    (96, 45, 813), (97, 49, 531), (98, 40, 700), (99, 19, 219), (100, 48, 314)
]

processing_times = [p for (_, p, _) in jobs_data]
due_dates = [d for (_, _, d) in jobs_data]


# ==========================================================
# 1. 평가 함수
# ==========================================================
def compute_total_tardiness(sequence, p, d):
    current_time = 0
    total_tardiness = 0
    for job in sequence:
        current_time += p[job - 1]
        total_tardiness += max(0, current_time - d[job - 1])
    return total_tardiness


# ==========================================================
# 2. 초기해 생성 (SPT, EDD)
# ==========================================================
def initial_solution(p, d, mode="SPT"):
    jobs = list(range(1, len(p) + 1))
    if mode == "SPT":
        jobs.sort(key=lambda j: p[j - 1])
    elif mode == "EDD":
        jobs.sort(key=lambda j: d[j - 1])
    return jobs


# ==========================================================
# 3. Destruction
# ==========================================================
def destruction_phase(sequence, d_size):
    seq = sequence.copy()
    selected_indices = sorted(random.sample(range(len(seq)), d_size), reverse=True)
    selected_jobs = [seq[i] for i in selected_indices]
    for idx in selected_indices:
        seq.pop(idx)
    return seq, selected_jobs


# ==========================================================
# 4. Construction
# ==========================================================
def construction_phase(partial_seq, jobs_to_insert, p, d):
    seq = partial_seq.copy()
    for job in jobs_to_insert:
        best_seq = None
        best_tardiness = float('inf')
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [job] + seq[pos:]
            tardiness = compute_total_tardiness(new_seq, p, d)
            if tardiness < best_tardiness:
                best_tardiness = tardiness
                best_seq = new_seq
        seq = best_seq
    return seq


# ==========================================================
# 5. Local Search
# ==========================================================
def local_search(sequence, p, d, time_limit_sec=TIME_LIMIT_LS):
    start = time.time()
    seq = sequence.copy()
    best_tardiness = compute_total_tardiness(seq, p, d)

    while time.time() - start < time_limit_sec:
        improved = False
        for i in range(len(seq)):
            for j in range(len(seq)):
                if time.time() - start >= time_limit_sec:
                    return seq, best_tardiness

                if i == j:
                    continue
                new_seq = seq.copy()
                job = new_seq.pop(i)
                new_seq.insert(j, job)
                new_tardiness = compute_total_tardiness(new_seq, p, d)
                if new_tardiness < best_tardiness:
                    seq = new_seq
                    best_tardiness = new_tardiness
                    improved = True
                    break
            if improved:
                break

        if not improved:
            break

    return seq, best_tardiness


# ==========================================================
# 6. Iterated Greedy
# ==========================================================
def iterated_greedy(p, d, destruction_size=5, init_mode="SPT", time_limit=TIME_LIMIT_IG):
    start = time.time()
    current_seq = initial_solution(p, d, init_mode)
    best_seq = current_seq[:]
    best_tardiness = compute_total_tardiness(best_seq, p, d)

    iteration = 0

    while time.time() - start < time_limit:
        iteration += 1

        # Destruction
        partial_seq, removed_jobs = destruction_phase(current_seq, destruction_size)

        # Construction
        new_seq = construction_phase(partial_seq, removed_jobs, p, d)
        new_tardiness = compute_total_tardiness(new_seq, p, d)

        # Acceptance
        if new_tardiness < best_tardiness:
            best_seq = new_seq[:]
            best_tardiness = new_tardiness
            current_seq = new_seq[:]
        else:
            if random.random() < 0.1:
                current_seq = new_seq[:]

    return best_seq, best_tardiness, time.time() - start


# ==========================================================
# 7. Gurobi 최적해 계산 (1||ΣTᵢ)
# ==========================================================
def solve_with_gurobi(p, d, time_limit_sec=TIME_LIMIT_GUROBI):
    n = len(p)
    model = Model("single_machine_tardiness")
    model.setParam("TimeLimit", time_limit_sec)
    model.setParam("OutputFlag", 0)

    # variables
    start_time = model.addVars(n, lb=0, vtype=GRB.CONTINUOUS, name="S")
    completion = model.addVars(n, lb=0, vtype=GRB.CONTINUOUS, name="C")
    tardiness = model.addVars(n, lb=0, vtype=GRB.CONTINUOUS, name="T")

    # sequencing variables
    y = model.addVars(n, n, vtype=GRB.BINARY, name="y")

    # constraints
    for i in range(n):
        model.addConstr(completion[i] == start_time[i] + p[i])
        model.addConstr(tardiness[i] >= completion[i] - d[i])

    M = sum(p) + max(d)

    for i in range(n):
        for j in range(n):
            if i != j:
                model.addConstr(start_time[j] >= completion[i] - M * (1 - y[i, j]))

    for i in range(n):
        for j in range(i + 1, n):
            model.addConstr(y[i, j] + y[j, i] == 1)

    model.setObjective(sum(tardiness[i] for i in range(n)), GRB.MINIMIZE)
    model.optimize()

    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        obj = model.ObjVal
    else:
        obj = None

    return obj, model.Runtime


# ==========================================================
# 8. 전체 실행
# ==========================================================
if __name__ == "__main__":

    print("\n========== Running IG ==========")
    ig_seq, ig_tard, ig_time = iterated_greedy(processing_times, due_dates)

    print("\n========== Running Local Search ==========")
    ls_seq, ls_tard = local_search(
        initial_solution(processing_times, due_dates, "SPT"),
        processing_times,
        due_dates,
        TIME_LIMIT_LS
    )
    ls_time = TIME_LIMIT_LS

    # SPT / EDD 평가
    seq_spt = initial_solution(processing_times, due_dates, "SPT")
    seq_edd = initial_solution(processing_times, due_dates, "EDD")

    tard_spt = compute_total_tardiness(seq_spt, processing_times, due_dates)
    tard_edd = compute_total_tardiness(seq_edd, processing_times, due_dates)

    # Gurobi
    print("\n========== Running Gurobi ==========")
    gurobi_obj, gurobi_time = solve_with_gurobi(processing_times, due_dates)

    # 결과 요약
    print("\n=====================================================")
    print("                 최종 결과 비교")
    print("=====================================================")
    print(f"SPT               : tardiness={tard_spt}, time=0")
    print(f"EDD               : tardiness={tard_edd}, time=0")
    print(f"Local Search      : tardiness={ls_tard}, time={ls_time:.2f}")
    print(f"IG                : tardiness={ig_tard}, time={ig_time:.2f}")
    print(f"Gurobi Optimal    : tardiness={gurobi_obj}, time={gurobi_time:.2f}")
    print("=====================================================")
