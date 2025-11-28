import random
import time

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except:
    GUROBI_AVAILABLE = False


# ==========================================================
# 1️⃣ 문제 세팅
# ==========================================================
NUM_JOBS = 100
random.seed(42)

processing_times = [random.randint(5, 50) for _ in range(NUM_JOBS)]
due_dates = [random.randint(200, 1000) for _ in range(NUM_JOBS)]

print("\n==============================")
print("📘 [노션 붙여넣기용 입력 리스트]")
print("==============================")
print("\nprocessing_times = ", processing_times)
print("\ndue_dates = ", due_dates)
print("\n")


# ==========================================================
# 2️⃣ 평가 함수 (총 지연합)
# ==========================================================
def compute_total_tardiness(sequence, p, d):
    current_time = 0
    total_tardiness = 0
    for job in sequence:
        current_time += p[job - 1]
        total_tardiness += max(0, current_time - d[job - 1])
    return total_tardiness


# ==========================================================
# 3️⃣ 초기해 생성 (SPT / EDD)
# ==========================================================
def initial_solution(p, d, mode="SPT"):
    jobs = list(range(1, len(p) + 1))
    if mode == "SPT":
        jobs.sort(key=lambda j: p[j - 1])
    elif mode == "EDD":
        jobs.sort(key=lambda j: d[j - 1])
    return jobs


# ==========================================================
# 4️⃣ Destruction Phase
# ==========================================================
def destruction_phase(sequence, d_size):
    seq = sequence.copy()
    selected_indices = sorted(random.sample(range(len(seq)), d_size), reverse=True)
    selected_jobs = [seq[i] for i in selected_indices]
    for idx in selected_indices:
        seq.pop(idx)
    return seq, selected_jobs


# ==========================================================
# 5️⃣ Construction Phase
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
# 6️⃣ Local Search (Insertion 기반)
# ==========================================================
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
# 7️⃣ IG 알고리즘
# ==========================================================
def iterated_greedy(p, d, destruction_size=5, init_mode="SPT", time_limit=1800):
    print(f"🚀 [IG 알고리즘 시작 - 초기해: {init_mode}, 시간제한: {time_limit}초]\n")
    start_time = time.time()

    current_seq = initial_solution(p, d, init_mode)
    best_seq = current_seq[:]
    best_tardiness = compute_total_tardiness(best_seq, p, d)

    iteration = 0

    while time.time() - start_time < time_limit:
        iteration += 1

        partial_seq, removed_jobs = destruction_phase(current_seq, destruction_size)
        new_seq = construction_phase(partial_seq, removed_jobs, p, d)

        new_tardiness = compute_total_tardiness(new_seq, p, d)

        if new_tardiness < best_tardiness:
            best_seq = new_seq[:]
            best_tardiness = new_tardiness
            current_seq = new_seq[:]
        else:
            if random.random() < 0.1:
                current_seq = new_seq[:]

    elapsed = time.time() - start_time
    print(f"\n✅ IG 종료 (총 반복: {iteration}, 총 시간: {elapsed:.2f}s)\n")
    return best_seq, best_tardiness, elapsed


# ==========================================================
# 8️⃣ Gurobi 최적해 계산 (1||ΣTj)
# ==========================================================
def solve_with_gurobi(p, d):
    if not GUROBI_AVAILABLE:
        print("⚠️ Gurobi 미설치: 최적해 계산 skip")
        return None, None

    n = len(p)
    model = gp.Model("single_machine_tardiness")
    model.setParam('OutputFlag', 0)

    C = model.addVars(n, vtype=GRB.CONTINUOUS)
    T = model.addVars(n, vtype=GRB.CONTINUOUS)
    x = model.addVars(n, n, vtype=GRB.BINARY)

    model.setObjective(gp.quicksum(T[i] for i in range(n)), GRB.MINIMIZE)

    for i in range(n):
        model.addConstr(T[i] >= C[i] - d[i])
        model.addConstr(T[i] >= 0)

    M = sum(p) + max(d)
    for i in range(n):
        for j in range(n):
            if i != j:
                model.addConstr(C[j] >= C[i] + p[j] - M*(1 - x[i, j]))

    for j in range(n):
        model.addConstr(gp.quicksum(x[i, j] for i in range(n) if i != j) == 1)

    model.optimize()

    best_tardiness = model.objVal
    return None, best_tardiness


# ==========================================================
# 9️⃣ 실행 및 비교
# ==========================================================
if __name__ == "__main__":
    # IG 실행
    ig_seq, ig_tardiness, ig_time = iterated_greedy(
        processing_times, due_dates,
        destruction_size=5,
        init_mode="SPT",
        time_limit=10   # 테스트용. 실제 1800초 가능
    )

    # Local Search
    start_local = time.time()
    ls_seq, ls_tardiness = local_search(initial_solution(processing_times, due_dates, "SPT"), processing_times, due_dates)
    end_local = time.time()

    # SPT / EDD 값
    spt_seq = initial_solution(processing_times, due_dates, "SPT")
    edd_seq = initial_solution(processing_times, due_dates, "EDD")
    spt_tardiness = compute_total_tardiness(spt_seq, processing_times, due_dates)
    edd_tardiness = compute_total_tardiness(edd_seq, processing_times, due_dates)

    # Gurobi
    _, gurobi_tardiness = solve_with_gurobi(processing_times, due_dates)

    print("\n==============================")
    print("📊 [최종 결과 비교]")
    print("==============================")
    print(f"SPT: tardiness = {spt_tardiness}")
    print(f"EDD: tardiness = {edd_tardiness}")
    print(f"Local Search: tardiness = {ls_tardiness}, time = {end_local - start_local:.2f}s")
    print(f"IG: tardiness = {ig_tardiness}, time = {ig_time:.2f}s")
    if gurobi_tardiness is not None:
        print(f"Gurobi 최적해: tardiness = {gurobi_tardiness}")
    else:
        print("Gurobi 최적해: (미실행)")
