import random
import time

# ==========================================================
# 1️⃣ 문제 세팅
# ==========================================================
NUM_JOBS = 100
random.seed(42)

# (p: 처리시간, d: 납기시간)
processing_times = [random.randint(5, 50) for _ in range(NUM_JOBS)]
due_dates = [random.randint(200, 1000) for _ in range(NUM_JOBS)]

print("📘 [문제 세팅: Job 100개 무작위 생성]")
for i in range(NUM_JOBS):
    print(f"Job {i + 1}: p={processing_times[i]}, d={due_dates[i]}")
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
# 7️⃣ IG 알고리즘 (시간제약 1800초)
# ==========================================================
def iterated_greedy(p, d, destruction_size=5, init_mode="SPT", time_limit=1800):
    print(f"🚀 [IG 알고리즘 시작 - 초기해: {init_mode}, 시간제한: {time_limit}초]\n")
    start_time = time.time()

    # 초기 솔루션
    current_seq = initial_solution(p, d, init_mode)
    best_seq = current_seq[:]
    best_tardiness = compute_total_tardiness(best_seq, p, d)
    print(f"초기해: {best_seq}")
    print(f"초기 tardiness: {best_tardiness}\n")

    iteration = 0

    while time.time() - start_time < time_limit:
        iteration += 1

        # --- Destruction ---
        destruction_start = time.time()
        partial_seq, removed_jobs = destruction_phase(current_seq, destruction_size)
        destruction_end = time.time()

        # --- Construction ---
        construction_start = time.time()
        new_seq = construction_phase(partial_seq, removed_jobs, p, d)
        construction_end = time.time()

        new_tardiness = compute_total_tardiness(new_seq, p, d)

        if new_tardiness < best_tardiness:
            best_tardiness = new_tardiness
            best_seq = new_seq[:]
            current_seq = new_seq[:]
        else:
            # bad move 허용 (탐색 다양성)
            if random.random() < 0.1:
                current_seq = new_seq[:]

        if iteration % 10 == 0:
            print(f"[Iter {iteration}] Best Tardiness: {best_tardiness:.2f} | Time: {time.time() - start_time:.2f}s")

    elapsed = time.time() - start_time
    print(f"\n✅ IG 종료 (총 반복: {iteration}, 총 시간: {elapsed:.2f}s)")
    return best_seq, best_tardiness, elapsed


# ==========================================================
# 8️⃣ 실행 및 로컬서치 비교
# ==========================================================
if __name__ == "__main__":
    # IG 실행
    ig_seq, ig_tardiness, ig_time = iterated_greedy(
        processing_times, due_dates,
        destruction_size=5,
        init_mode="SPT",  # or "EDD"
        time_limit=1800
    )

    # Local Search
    print("\n🔍 [로컬 서치 실행 중...]\n")
    start_local = time.time()
    ls_seq, ls_tardiness = local_search(initial_solution(processing_times, due_dates, "SPT"), processing_times,
                                        due_dates)
    end_local = time.time()

    # 결과 비교
    print("\n📊 [결과 비교 요약]")
    print(f"IG  : tardiness={ig_tardiness}, time={ig_time:.2f}s")
    print(f"Local Search : tardiness={ls_tardiness}, time={end_local - start_local:.2f}s")

    print(f"\n🧩 IG 최종 시퀀스:\n{ig_seq}")
    print(f"🧩 Local Search 최종 시퀀스:\n{ls_seq}")

