import random
import time

# ============================================
# 1️⃣ 문제 세팅 (싱글머신 스케줄링)
# ============================================

NUM_JOBS = 100
random.seed(42)

processing_times = [random.randint(5, 30) for _ in range(NUM_JOBS)]
due_dates = [random.randint(40, 200) for _ in range(NUM_JOBS)]

print("📘 [문제 세팅]")
print(f"총 Job 수: {NUM_JOBS}")
print(f"처리시간 예시(앞 10개): {processing_times[:10]}")
print(f"납기일 예시(앞 10개): {due_dates[:10]}\n")

# ============================================
# 2️⃣ 평가 함수 (총 지연합, Total Tardiness)
# ============================================
def compute_total_tardiness(sequence, p, d):
    current_time = 0
    total_tardiness = 0
    for job in sequence:
        current_time += p[job - 1]
        total_tardiness += max(0, current_time - d[job - 1])
    return total_tardiness

# ============================================
# 3️⃣ 초기 해 생성 (SPT or EDD)
# ============================================
def initial_solution(p, d, mode="SPT"):
    jobs = list(range(1, len(p) + 1))
    if mode == "SPT":
        jobs.sort(key=lambda j: p[j - 1])
    elif mode == "EDD":
        jobs.sort(key=lambda j: d[j - 1])
    else:
        random.shuffle(jobs)
    return jobs

# ============================================
# 4️⃣ Destruction Phase
# ============================================
def destruction_phase(sequence, d_size):
    seq = sequence.copy()
    selected_indices = sorted(random.sample(range(len(seq)), d_size), reverse=True)
    selected_jobs = [seq[i] for i in selected_indices]
    for idx in selected_indices:
        seq.pop(idx)
    return seq, selected_jobs

# ============================================
# 5️⃣ Construction Phase
# ============================================
def construction_phase(partial_seq, jobs_to_insert, p, d):
    seq = partial_seq.copy()
    for job in jobs_to_insert:
        best_seq = None
        best_tardiness = float("inf")
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [job] + seq[pos:]
            tardiness = compute_total_tardiness(new_seq, p, d)
            if tardiness < best_tardiness:
                best_tardiness = tardiness
                best_seq = new_seq
        seq = best_seq
    return seq

# ============================================
# 6️⃣ 간단한 Local Search (2-Exchange)
# ============================================
def local_search(sequence, p, d):
    best_seq = sequence[:]
    best_val = compute_total_tardiness(best_seq, p, d)
    improved = True

    while improved:
        improved = False
        for i in range(len(sequence) - 1):
            for j in range(i + 1, len(sequence)):
                new_seq = best_seq[:]
                new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                new_val = compute_total_tardiness(new_seq, p, d)
                if new_val < best_val:
                    best_seq = new_seq
                    best_val = new_val
                    improved = True
                    break
            if improved:
                break
    return best_seq, best_val

# ============================================
# 7️⃣ IG 알고리즘 (단일 iteration)
# ============================================
def iterated_greedy(p, d, destruction_size=3, init_mode="SPT", with_local_search=False):
    current_seq = initial_solution(p, d, init_mode)
    current_val = compute_total_tardiness(current_seq, p, d)
    best_seq, best_val = current_seq[:], current_val

    # Destruction & Construction
    partial_seq, removed_jobs = destruction_phase(current_seq, destruction_size)
    new_seq = construction_phase(partial_seq, removed_jobs, p, d)
    new_val = compute_total_tardiness(new_seq, p, d)

    if new_val < best_val:
        best_seq, best_val = new_seq, new_val

    # Local Search 선택적으로 적용
    if with_local_search:
        ls_seq, ls_val = local_search(best_seq, p, d)
        if ls_val < best_val:
            best_seq, best_val = ls_seq, ls_val

    return best_seq, best_val

# ============================================
# 8️⃣ 시간제한 기반 IG 반복 실행
# ============================================
if __name__ == "__main__":
    DESTRUCTION_SIZE = 3
    INIT_MODE = "SPT"
    USE_LOCAL_SEARCH = True
    TIME_LIMIT = 1800  # 30분 (초 단위)

    start_time = time.time()
    best_seq = None
    best_val = float("inf")
    iteration = 0

    print("🚀 [IG 시간제한 반복 시작]")
    print(f"제한시간: {TIME_LIMIT}초, Local Search: {USE_LOCAL_SEARCH}\n")

    while time.time() - start_time < TIME_LIMIT:
        iteration += 1
        seq, val = iterated_greedy(
            processing_times,
            due_dates,
            destruction_size=DESTRUCTION_SIZE,
            init_mode=INIT_MODE,
            with_local_search=USE_LOCAL_SEARCH,
        )
        if val < best_val:
            best_val = val
            best_seq = seq[:]

        # 1분마다 진행 상황 출력
        if iteration % 10 == 0 or (time.time() - start_time) % 60 < 0.5:
            elapsed = time.time() - start_time
            print(f"[Iter {iteration:04d}] Best Tardiness = {best_val}  (Elapsed: {elapsed:.1f}s)")

    total_elapsed = time.time() - start_time

    print("\n✅ [실행 완료]")
    print(f"총 반복 횟수: {iteration}")
    print(f"최종 최고 해의 총 지연합: {best_val}")
    print(f"총 실행 시간: {total_elapsed:.2f} 초")
    print(f"최적 시퀀스 (앞 30개): {best_seq[:30]}")
