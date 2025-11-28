# iterated_greedy_100jobs_1800s.py
import random
import time
from typing import List, Tuple

# ---------------------------
# 설정
# ---------------------------
NUM_JOBS = 100
TIME_LIMIT = 1800            # seconds (30 minutes)
RANDOM_SEED = 42             # 재현성을 위해 출력하고 고정
INIT_RULE = 'SPT'            # 'SPT' 또는 'EDD'
D_FIXED = 3                  # 제거할 job 개수. None이면 매 반복마다 랜덤(1..NUM_JOBS-1)

# 로그 출력 빈도 (몇 회마다 진행상황 요약 출력)
PRINT_PROGRESS_EVERY = 100

# ---------------------------
# 데이터 생성 (랜덤)
# ---------------------------
random.seed(RANDOM_SEED)
processing_times = [random.randint(1, 100) for _ in range(NUM_JOBS)]
due_dates = [random.randint(100, 500) for _ in range(NUM_JOBS)]
jobs = list(range(NUM_JOBS))

# 출력 : 랜덤으로 생성된 원본 데이터(요청 따라 기록)
print("=== Problem data (random seed = {}) ===".format(RANDOM_SEED))
print("NUM_JOBS =", NUM_JOBS)
print("Processing times (first 20 shown):", processing_times[:20], "...")
print("Due dates (first 20 shown):      ", due_dates[:20], "...")
print()

# ---------------------------
# 유틸: 총 타디니스 계산
# ---------------------------
def calc_total_tardiness(sequence: List[int]) -> int:
    t = 0
    total = 0
    for j in sequence:
        t += processing_times[j]
        total += max(0, t - due_dates[j])
    return total

# ---------------------------
# IG 알고리즘
# ---------------------------
def iterated_greedy(jobs: List[int],
                    time_limit: int = TIME_LIMIT,
                    d_fixed: int = D_FIXED,
                    init_rule: str = INIT_RULE) -> Tuple[List[int], int, List[dict]]:
    start_time = time.time()

    # 초기해 생성
    if init_rule == 'SPT':
        current = sorted(jobs, key=lambda j: processing_times[j])
    elif init_rule == 'EDD':
        current = sorted(jobs, key=lambda j: due_dates[j])
    else:
        current = random.sample(jobs, len(jobs))

    best = current[:]
    best_tard = calc_total_tardiness(best)
    initial_tard = best_tard

    print(f"Initial solution ({init_rule}) computed. Initial total tardiness = {initial_tard}")
    print("Initial solution (first 30 jobs):", best[:30], "...")
    print()

    records = []
    iteration = 0

    # main loop: 시간 제한 내 반복
    while time.time() - start_time < time_limit:
        iter_start = time.time()
        iteration += 1

        # --- Destruction ---
        if d_fixed is None:
            d = random.randint(1, len(current) - 1)
        else:
            d = max(1, min(d_fixed, len(current) - 1))
        # 뽑을 인덱스를 순서를 유지하며(뽑힌 순서도 랜덤) 선택하려면
        # 인덱스 후보에서 랜덤하게 d개 뽑고, 그 뽑힌 인덱스들을
        # '뽑힌 순서'를 랜덤하게 정해 제거 순서도 랜덤으로 처리
        chosen_indices = random.sample(range(len(current)), d)
        # 뽑힌 순서(random order)
        random.shuffle(chosen_indices)
        # 제거할 때는 뒤에서부터 제거해야 인덱스 흔들림이 없음 -> 제거 순서대로 리스트 만들기
        # 여기서는 '뽑힌 순서'대로 removed_jobs 리스트를 만들되 실제 제거는 index 내림차순으로 수행
        removed_jobs_in_order = [current[idx] for idx in chosen_indices]

        # 실제 제거 (인덱스 내림차순으로 pop)
        for idx in sorted(chosen_indices, reverse=True):
            current.pop(idx)

        # --- Construction ---
        # removed_jobs_in_order 순서대로 하나씩 삽입
        for job in removed_jobs_in_order:
            best_seq_for_job = None
            best_tard_for_job = float('inf')
            # 모든 위치에 삽입해 보고 가장 작은 tardiness 위치 선택
            for pos in range(len(current) + 1):
                # create new sequence with job inserted at pos
                # slicing is fine for n=100
                candidate = current[:pos] + [job] + current[pos:]
                tard = calc_total_tardiness(candidate)
                if tard < best_tard_for_job:
                    best_tard_for_job = tard
                    best_seq_for_job = candidate
            # adopt best insertion result
            current = best_seq_for_job

        iter_end = time.time()
        iter_elapsed = iter_end - iter_start

        # 평가
        current_tard = calc_total_tardiness(current)
        if current_tard < best_tard:
            best_tard = current_tard
            best = current[:]
            improved = True
        else:
            improved = False

        # 기록 (메모리에 보관 — 파일 저장 안 함)
        rec = {
            'iteration': iteration,
            'removed_jobs': removed_jobs_in_order,
            'd_used': d,
            'current_tardiness': current_tard,
            'best_tardiness': best_tard,
            'iteration_time': iter_elapsed,
            'total_elapsed': time.time() - start_time
        }
        records.append(rec)

        # 출력: 주기/개선 시
        if improved:
            print(f"[Iter {iteration}] Improvement! current_tard={current_tard} best_tard updated -> {best_tard} (iter_time={iter_elapsed:.3f}s total_elapsed={rec['total_elapsed']:.2f}s)")
        elif iteration % PRINT_PROGRESS_EVERY == 0:
            print(f"[Iter {iteration}] no improvement. current_tard={current_tard} best_tard={best_tard} (iter_time={iter_elapsed:.3f}s total_elapsed={rec['total_elapsed']:.2f}s)")

    total_time = time.time() - start_time
    print()
    print("=== Finished IG ===")
    print(f"Total iterations: {iteration}")
    print(f"Total wall-clock time: {total_time:.2f} seconds (limit was {time_limit}s)")
    print(f"Initial tardiness (init {INIT_RULE}): {initial_tard}")
    print(f"Best tardiness found: {best_tard}")
    print("Best sequence (first 50 jobs):", best[:50], "...")
    print()

    return best, best_tard, records

# ---------------------------
# 실행
# ---------------------------
if __name__ == "__main__":
    best_seq, best_val, logs = iterated_greedy(jobs, time_limit=TIME_LIMIT, d_fixed=D_FIXED, init_rule=INIT_RULE)
    # 요약 샘플 출력 (마지막 10회 기록만)
    print("=== Last 10 iteration records (sample) ===")
    for r in logs[-10:]:
        print(f"Iter {r['iteration']:>4}: d={r['d_used']}, removed={r['removed_jobs']}, current_tard={r['current_tardiness']}, best_tard={r['best_tardiness']}, iter_time={r['iteration_time']:.3f}s, total_elapsed={r['total_elapsed']:.2f}s")
    print()
    print("Run complete.")
