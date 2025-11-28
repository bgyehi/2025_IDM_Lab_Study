from gurobipy import *
import random
import time

# =======================
# 파라미터
# =======================
job_num         = 100
seed            = 42
pt_min, pt_max  = 1, 20   # 처리시간 p 범위
due_tightness   = 0.6     # due 생성 조임 정도
TIME_LIMIT_SEC  = 1800    # Gurobi 30분 제한
TIME_LIMIT_LS   = 1800    # 로컬 서치용 30분 시간제한(구로비와 동등한 비교위해)
# =======================

random.seed(seed)

# =======================
# 1. 공통 Job 데이터 생성 (고정 인스턴스)
# =======================
# 고정된 100개 작업 (Job i: p, d)
jobs_data = [
    (1, 45, 532),
    (2, 12, 986),
    (3, 6, 994),
    (4, 22, 257),
    (5, 20, 434),
    (6, 19, 232),
    (7, 13, 523),
    (8, 11, 610),
    (9, 48, 474),
    (10, 39, 267),
    (11, 10, 416),
    (12, 42, 780),
    (13, 32, 935),
    (14, 7, 522),
    (15, 6, 417),
    (16, 10, 871),
    (17, 18, 711),
    (18, 19, 605),
    (19, 37, 858),
    (20, 43, 669),
    (21, 6, 346),
    (22, 40, 471),
    (23, 17, 342),
    (24, 50, 452),
    (25, 46, 962),
    (26, 49, 774),
    (27, 39, 751),
    (28, 31, 469),
    (29, 19, 964),
    (30, 33, 798),
    (31, 42, 638),
    (32, 22, 797),
    (33, 5, 608),
    (34, 15, 570),
    (35, 49, 424),
    (36, 32, 341),
    (37, 26, 721),
    (38, 22, 705),
    (39, 14, 293),
    (40, 18, 973),
    (41, 26, 248),
    (42, 11, 312),
    (43, 10, 356),
    (44, 29, 842),
    (45, 11, 363),
    (46, 27, 896),
    (47, 27, 632),
    (48, 43, 810),
    (49, 21, 265),
    (50, 7, 594),
    (51, 34, 590),
    (52, 39, 810),
    (53, 12, 679),
    (54, 29, 741),
    (55, 10, 457),
    (56, 40, 766),
    (57, 23, 211),
    (58, 45, 896),
    (59, 44, 938),
    (60, 28, 317),
    (61, 41, 898),
    (62, 17, 749),
    (63, 50, 968),
    (64, 9, 473),
    (65, 7, 987),
    (66, 47, 856),
    (67, 19, 548),
    (68, 23, 314),
    (69, 10, 500),
    (70, 19, 645),
    (71, 11, 361),
    (72, 29, 664),
    (73, 22, 203),
    (74, 34, 939),
    (75, 45, 936),
    (76, 28, 469),
    (77, 15, 712),
    (78, 28, 980),
    (79, 27, 382),
    (80, 18, 719),
    (81, 47, 308),
    (82, 22, 840),
    (83, 49, 505),
    (84, 48, 854),
    (85, 46, 719),
    (86, 9, 823),
    (87, 43, 403),
    (88, 45, 356),
    (89, 15, 582),
    (90, 39, 980),
    (91, 20, 365),
    (92, 15, 752),
    (93, 34, 997),
    (94, 29, 743),
    (95, 22, 200),
    (96, 45, 813),
    (97, 49, 531),
    (98, 40, 700),
    (99, 19, 219),
    (100, 48, 314),
]

# Job ID 리스트 (J001 ~ J100)
jobs = [f"J{idx:03d}" for idx in range(1, len(jobs_data) + 1)]

# 처리시간 p_j, 마감시간 d_j 딕셔너리로 매핑
p = {f"J{i:03d}": pt for i, pt, dd in jobs_data}
d = {f"J{i:03d}": dd for i, pt, dd in jobs_data}

# Big-M 계산용 총 처리시간
sum_p = sum(p.values())
M = sum_p

# 생성된 Job 리스트 출력
print("=== Generated Jobs (고정 공통 데이터) ===")
for i, (idx, pt, dd) in enumerate(jobs_data, start=1):
    print(f"Job {idx}: p={pt}, d={dd}")
print()


# =======================
# 유틸 함수: 총 Tardiness 계산
# =======================
def total_tardiness(order):
    """order: 작업 ID 리스트 (예: ['J001', 'J010', ...])"""
    time_now = 0
    total = 0
    for j in order:
        time_now += p[j]
        tard = max(0, time_now - d[j])
        total += tard
    return total


# =======================
# 2. Gurobi 모델 (Single Machine Tardiness)
# =======================
m = Model("Single_Machine_Tardiness")
m.Params.TimeLimit = TIME_LIMIT_SEC

# 변수 생성
y = m.addVars(jobs, jobs, vtype=GRB.BINARY, name="y")   # i가 j보다 먼저면 1
for j in jobs:
    y[j, j].ub = 0  # 자기 자신 금지

C = m.addVars(jobs, vtype=GRB.CONTINUOUS, name="C")     # 완료시간
T = m.addVars(jobs, vtype=GRB.CONTINUOUS, name="T")     # 지연시간

# 목적함수: 총 tardiness 최소화
m.setObjective(quicksum(T[j] for j in jobs), GRB.MINIMIZE)

# 제약조건
# 1) 완료시간 (선행 작업 처리시간의 합으로 C 정의, idle time 없음)
for j in jobs:
    m.addConstr(
        C[j] == p[j] + quicksum(p[i] * y[i, j] for i in jobs if i != j),
        name=f"completion[{j}]"
    )

# 2) 쌍별 순서 (i가 j보다 먼저 또는 그 반대: 둘 중 하나)
for i in jobs:
    for j in jobs:
        if i != j:
            m.addConstr(y[i, j] + y[j, i] == 1, name=f"precedence[{i},{j}]")

# 3) Big-M 비겹침 제약 (완료시간 관점에서 겹치지 않도록)
for i in jobs:
    for j in jobs:
        if i == j:
            continue
        m.addConstr(
            C[i] <= C[j] - p[j] + M * (1 - y[i, j]),
            name=f"nonoverlap1[{i},{j}]"
        )
        m.addConstr(
            C[j] <= C[i] - p[i] + M * (y[i, j]),
            name=f"nonoverlap2[{i},{j}]"
        )

# 4) Tardiness 정의
for j in jobs:
    m.addConstr(T[j] >= C[j] - d[j], name=f"tard1[{j}]")
    m.addConstr(T[j] >= 0,         name=f"tard2[{j}]")


# 최적화 실행 및 시간 측정
start_time_gurobi = time.time()
m.optimize()
elapsed_gurobi = time.time() - start_time_gurobi

gurobi_tt = None
gurobi_order = None

print("\n=== Gurobi Result ===")
if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
    if m.status == GRB.OPTIMAL:
        print("최적해를 찾았습니다.")
    elif m.status == GRB.TIME_LIMIT:
        print("시간 제한 내에서 최선해를 찾았습니다.")
    else:
        print("부분 최적해(또는 허용 가능한 해)를 찾았습니다.")

    gurobi_tt = m.ObjVal
    print(f"Gurobi 총 Tardiness = {gurobi_tt}")
    print(f"Gurobi 계산 시간 = {elapsed_gurobi:.2f} 초")

    # 완료시간 기준 작업 순서
    gurobi_order = sorted(jobs, key=lambda j: C[j].X)
    print("Gurobi 순서 (앞 20개):", " → ".join(gurobi_order[:20]))
else:
    print(" 최적해를 찾지 못했거나 문제가 발생했습니다.")
    print("Gurobi status:", m.status)


# =======================
# 3. 규칙기반: SPT / EDD
# =======================
# SPT
start_time_spt = time.time()
spt_order = sorted(jobs, key=lambda j: p[j])
spt_tt = total_tardiness(spt_order)
elapsed_spt = time.time() - start_time_spt

# EDD
start_time_edd = time.time()
edd_order = sorted(jobs, key=lambda j: d[j])
edd_tt = total_tardiness(edd_order)
elapsed_edd = time.time() - start_time_edd

print("\n=== Heuristic Rules (Same Instance) ===")
print(f"SPT 총 Tardiness = {spt_tt},   시간 = {elapsed_spt:.4f} 초")
print(f"EDD 총 Tardiness = {edd_tt},   시간 = {elapsed_edd:.4f} 초")