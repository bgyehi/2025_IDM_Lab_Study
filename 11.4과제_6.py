import random
import time

# Gurobi optional import
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except:
    GUROBI_AVAILABLE = False

# ==========================================================
# 설정: 재현성(무작위성), 목표 tardiness (있으면 조기종료)
# ==========================================================
random.seed(42)
TARGET_TARDINESS = 46653   # IG가 이 값을 만들면 조기 종료(없으면 None)
TIME_LIMIT_IG = 1800       # 초 단위 (테스트할 땐 10 또는 30으로 줄여도 됨)
DESTRUCTION_SIZE = 5
INIT_MODE = "SPT"          # "SPT" 또는 "EDD"

# ==========================================================
# 1) 사용자 제공: processing_times, due_dates (직접 붙여넣음)
# ==========================================================
processing_times = [
45,12,6,22,20,19,13,11,48,39,10,42,32,7,6,10,18,19,37,43,
6,40,17,50,46,49,39,31,19,33,42,22,5,15,49,32,26,22,14,18,
26,11,10,29,11,27,27,43,21,7,34,39,12,29,10,40,23,45,44,28,
41,17,50,9,7,47,19,23,10,19,11,29,22,34,45,28,15,28,27,18,
47,22,49,48,46,9,43,45,15,39,20,15,34,29,22,45,49,40,19,48
]

due_dates = [
532,986,994,257,434,232,523,610,474,267,416,780,935,522,417,871,711,605,858,669,
346,471,342,452,962,774,751,469,964,798,638,797,608,570,424,341,721,705,293,973,
248,312,356,842,363,896,632,810,265,594,590,810,679,741,457,766,211,896,938,317,
898,749,968,473,987,856,548,314,500,645,361,664,203,939,936,469,712,980,382,719,
308,840,505,854,719,823,403,356,582,980,365,752,997,743,200,813,531,700,219,314
]

NUM_JOBS = len(processing_times)
assert NUM_JOBS == len(due_dates), "processing_times와 due_dates 길이가 달라요."

# ==========================================================
# 평가 함수
# ==========================================================
def compute_total_tardiness(sequence, p, d):
    current_time = 0
    total_tardiness = 0
    for job in sequence:
        # sequence uses 1-based job ids
        current_time += p[job - 1]
        total_tardiness += max(0, current_time - d[job - 1])
    return total_tardiness

# ==========================================================
# 초기해 (SPT / EDD)
# ==========================================================
def initial_solution(p, d, mode="SPT"):
    jobs = list(range(1, len(p) + 1))
    if mode == "SPT":
        jobs.sort(key=lambda j: p[j - 1])
    elif mode == "EDD":
        jobs.sort(key=lambda j: d[j - 1])
    return jobs

# ==========================================================
# Destruction / Construction / Local Search
# ==========================================================
def destruction_phase(sequence, d_size):
    seq = sequence.copy()
    selected_indices = sorted(random.sample(range(len(seq)), d_size), reverse=True)
    selected_jobs = [seq[i] for i in selected_indices]
    for idx in selected_indices:
        seq.pop(idx)
    return seq, selected_jobs

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

def local_search(sequence, p, d):
    improved = True
    seq = sequence.copy()
    best_tardiness = compute_total_tardiness(seq, p, d)
    while improved:
        improved = False
        for i in range(len(seq)):
            for j in range(len(seq)):
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
    return seq, best_tardiness

# ==========================================================
# IG 알고리즘 (조기종료: TARGET_TARDINESS 도달 시)
# ==========================================================
def iterated_greedy(p, d, destruction_size=5, init_mode="SPT", time_limit=1800, target=None):
    print(f"\n🚀 IG 시작 (init={init_mode}, destruction_size={destruction_size}, time_limit={time_limit}s)")
    start_time = time.time()
    current_seq = initial_solution(p, d, init_mode)
    best_seq = current_seq[:]
    best_tardiness = compute_total_tardiness(best_seq, p, d)
    iteration = 0

    # 즉시 타깃 체크
    if target is not None and best_tardiness <= target:
        print(f"초기해가 이미 target({target}) 이하입니다: {best_tardiness}")
        return best_seq, best_tardiness, 0.0

    while time.time() - start_time < time_limit:
        iteration += 1
        partial_seq, removed_jobs = destruction_phase(current_seq, destruction_size)
        new_seq = construction_phase(partial_seq, removed_jobs, p, d)
        new_tardiness = compute_total_tardiness(new_seq, p, d)

        if new_tardiness < best_tardiness:
            best_tardiness = new_tardiness
            best_seq = new_seq[:]
            current_seq = new_seq[:]
        else:
            if random.random() < 0.1:
                current_seq = new_seq[:]

        # 로그 (빈도 낮게)
        if iteration % 100 == 0:
            elapsed = time.time() - start_time
            print(f"[Iter {iteration}] best_tardiness={best_tardiness} elapsed={elapsed:.2f}s")

        # 타깃 도달 시 조기종료
        if target is not None and best_tardiness <= target:
            elapsed = time.time() - start_time
            print(f"🎯 TARGET 달성: iteration={iteration}, tardiness={best_tardiness}, time={elapsed:.2f}s")
            return best_seq, best_tardiness, elapsed

    elapsed = time.time() - start_time
    print(f"✅ IG 종료: iterations={iteration}, best_tardiness={best_tardiness}, time={elapsed:.2f}s")
    return best_seq, best_tardiness, elapsed

# ==========================================================
# Gurobi (선택적): 간단한 순서변수 x_ij (주의: n=100이면 변수 O(n^2)로 무거움)
# ==========================================================
def solve_with_gurobi(p, d, time_limit=300):
    if not GUROBI_AVAILABLE:
        print("⚠️ Gurobi 미설치 — skip")
        return None
    n = len(p)
    model = gp.Model("single_machine_tardiness")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)

    # decision: x[i,j] = 1 if i 바로 다음에 j (i,j are 0-based job indices)
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
    C = model.addVars(n, vtype=GRB.CONTINUOUS, name="C")
    T = model.addVars(n, vtype=GRB.CONTINUOUS, name="T")

    # objective
    model.setObjective(gp.quicksum(T[i] for i in range(n)), GRB.MINIMIZE)

    # tardiness def
    for i in range(n):
        model.addConstr(T[i] >= C[i] - d[i])
        model.addConstr(T[i] >= 0)

    # sequencing: big-M formulation (경계값)
    M = sum(p) + max(d)
    for i in range(n):
        for j in range(n):
            if i == j:
                # optional: x[i,i]=0
                model.addConstr(x[i, j] == 0)
            else:
                model.addConstr(C[j] >= C[i] + p[j] - M * (1 - x[i, j]))

    # each job has exactly one predecessor (except a virtual start) -> simpler relaxation:
    # enforce exactly one incoming arc per job
    for j in range(n):
        model.addConstr(gp.quicksum(x[i, j] for i in range(n) if i != j) == 1)

    # Optimize (may be heavy)
    model.optimize()
    if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
        print("Gurobi 종료, status:", model.Status)
        return model.ObjVal
    else:
        print("Gurobi 비정상 종료, status:", model.Status)
        return None

# ==========================================================
# 실행: IG, Local Search, SPT, EDD, (선택)Gurobi
# ==========================================================
if __name__ == "__main__":
    # 1) SPT / EDD baseline
    spt_seq = initial_solution(processing_times, due_dates, "SPT")
    edd_seq = initial_solution(processing_times, due_dates, "EDD")
    spt_tardiness = compute_total_tardiness(spt_seq, processing_times, due_dates)
    edd_tardiness = compute_total_tardiness(edd_seq, processing_times, due_dates)

    print("\n📌 Baseline")
    print(f"SPT tardiness = {spt_tardiness}")
    print(f"EDD tardiness = {edd_tardiness}")

    # 2) Local Search (from SPT)
    t0 = time.time()
    ls_seq, ls_tardiness = local_search(spt_seq, processing_times, due_dates)
    t_ls = time.time() - t0
    print(f"\n🔍 Local Search (from SPT): tardiness = {ls_tardiness} (time {t_ls:.2f}s)")

    # 3) IG
    ig_seq, ig_tardiness, ig_time = iterated_greedy(
        processing_times, due_dates,
        destruction_size=DESTRUCTION_SIZE,
        init_mode=INIT_MODE,
        time_limit=TIME_LIMIT_IG,
        target=TARGET_TARDINESS
    )

    print(f"\n🚩 IG 결과: tardiness = {ig_tardiness}, time = {ig_time:.2f}s")

    # 4) Gurobi (optional, 오래 걸림)
    if GUROBI_AVAILABLE:
        print("\n🔐 Gurobi로 최적화 시도 (무거움: n^2 변수)")
        gurobi_obj = solve_with_gurobi(processing_times, due_dates, time_limit=1800)
    else:
        gurobi_obj = None

    # 5) 종합 출력
    print("\n==============================")
    print("최종 비교 요약")
    print("==============================")
    print(f"SPT: {spt_tardiness}")
    print(f"EDD: {edd_tardiness}")
    print(f"Local Search (from SPT): {ls_tardiness} (time {t_ls:.2f}s)")
    print(f"IG: {ig_tardiness} (time {ig_time:.2f}s)")
    if gurobi_obj is not None:
        print(f"Gurobi objective (ΣTj) : {gurobi_obj}")
    else:
        print("Gurobi: 미실행 or 미설치")
