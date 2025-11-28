import random
import gurobipy as gp
from gurobipy import GRB

# -----------------------------
# 1. 문제 인스턴스 생성
# -----------------------------
n_jobs = 100
p_low, p_high = 1, 100
tightness = 0.05  # 듀데이트 분산
center_ratio = 0.45  # 아슬아슬하게 지각 발생 목표
seed = 2025
random.seed(seed)

# 프로세싱 타임 생성
processing_times = [random.randint(p_low, p_high) for _ in range(n_jobs)]
sum_p = sum(processing_times)

# 아슬아슬한 듀데이트 생성
center = sum_p * center_ratio
spread = sum_p * tightness
due_dates = [int(round(random.uniform(center - spread/2, center + spread/2))) + p
             for p in processing_times]

print("sum p =", sum_p, "sample d =", due_dates[:8])

# -----------------------------
# 2. SPT / EDD 간단 계산
# -----------------------------
def total_tardiness(order):
    time_acc = 0
    tardy_sum = 0
    for j in order:
        time_acc += processing_times[j]
        tardy_sum += max(0, time_acc - due_dates[j])
    return tardy_sum

SPT_order = sorted(range(n_jobs), key=lambda j: processing_times[j])
EDD_order = sorted(range(n_jobs), key=lambda j: due_dates[j])

print("SPT total tardiness =", total_tardiness(SPT_order))
print("EDD total tardiness =", total_tardiness(EDD_order))

# -----------------------------
# 3. Gurobi 최적화 모델
# -----------------------------
m = gp.Model("tardiness_minimization")

# 변수: x[i,j] = 1 if job i is in position j
x = m.addVars(n_jobs, n_jobs, vtype=GRB.BINARY, name="x")

# 변수: C_j = completion time of job at position j
C = m.addVars(n_jobs, vtype=GRB.CONTINUOUS, name="C")

# 변수: T_i = tardiness of job i
T = m.addVars(n_jobs, vtype=GRB.CONTINUOUS, name="T")

# -----------------------------
# 4. 제약조건
# -----------------------------
# 각 job이 정확히 하나 위치에 배치
for i in range(n_jobs):
    m.addConstr(gp.quicksum(x[i,j] for j in range(n_jobs)) == 1)

# 각 위치에 정확히 하나 job 배치
for j in range(n_jobs):
    m.addConstr(gp.quicksum(x[i,j] for i in range(n_jobs)) == 1)

# 완료시간 정의
for j in range(n_jobs):
    m.addConstr(C[j] == gp.quicksum(processing_times[i]*x[i,j] for i in range(n_jobs)) + (C[j-1] if j>0 else 0))

# 지각 정의
for i in range(n_jobs):
    m.addConstr(T[i] >= gp.quicksum(C[j]*x[i,j] for j in range(n_jobs)) - due_dates[i])
    m.addConstr(T[i] >= 0)

# -----------------------------
# 5. 목적함수
# -----------------------------
m.setObjective(gp.quicksum(T[i] for i in range(n_jobs)), GRB.MINIMIZE)

# -----------------------------
# 6. Gurobi 파라미터 설정
# -----------------------------
m.setParam("TimeLimit", 1800)  # 30분
m.setParam("MIPGap", 0.01)
m.setParam("MIPFocus", 1)
m.setParam("Heuristics", 0.5)
m.setParam("Presolve", 2)

# -----------------------------
# 7. 최적화 실행
# -----------------------------
m.optimize()

# -----------------------------
# 8. 결과 확인
# -----------------------------
if m.status == GRB.OPTIMAL or m.status == GRB.TIME_LIMIT:
    print("Gurobi solution total tardiness =", m.objVal)
    # 첫 20 jobs 출력
    job_order = []
    for j in range(n_jobs):
        for i in range(n_jobs):
            if x[i,j].X > 0.5:
                job_order.append(i)
                break
    print("First 20 jobs in Gurobi order:", job_order[:20])
else:
    print("Gurobi did not find a solution")
