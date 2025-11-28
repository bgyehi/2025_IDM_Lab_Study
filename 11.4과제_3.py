import random
import time

# -------------------------------
# 🔹 설정
# -------------------------------
NUM_JOBS = 100
TIME_LIMIT = 1800  # 초 단위 (30분)
mode = "SPT"  # "SPT" or "EDD" 선택 가능

random.seed(42)  # 재현성

# -------------------------------
# 🔹 유틸리티 함수
# -------------------------------
def calc_total_tardiness(sequence):
    """총 Tardiness 계산"""
    t = 0
    total_tardiness = 0
    for p, d in sequence:
        t += p
        total_tardiness += max(t - d, 0)
    return total_tardiness

def local_search(sequence):
    """Swap 기반 Local Search"""
    best_seq = sequence[:]
    best_tard = calc_total_tardiness(best_seq)
    improved = True
    while improved:
        improved = False
        for i in range(len(sequence)):
            for j in range(i + 1, len(sequence)):
                seq = best_seq[:]
                seq[i], seq[j] = seq[j], seq[i]
                tard = calc_total_tardiness(seq)
                if tard < best_tard:
                    best_tard = tard
                    best_seq = seq[:]
                    improved = True
        if improved:
            sequence = best_seq[:]
    return best_seq, best_tard

# -------------------------------
# 🔹 Job 생성
# -------------------------------
jobs = [(random.randint(1, 100), random.randint(50, 500)) for _ in range(NUM_JOBS)]

# -------------------------------
# 🔹 초기해 (SPT or EDD)
# -------------------------------
if mode == "SPT":
    initial_sequence = sorted(jobs, key=lambda x: x[0])  # Processing time 기준
else:
    initial_sequence = sorted(jobs, key=lambda x: x[1])  # Due date 기준

best_sequence = initial_sequence[:]
best_tardiness = calc_total_tardiness(best_sequence)

print(f"초기해 ({mode}) tardiness:", best_tardiness)

# -------------------------------
# 🔹 IG 알고리즘 (Iterated Greedy)
# -------------------------------
start_time = time.time()
iteration = 0

while time.time() - start_time < TIME_LIMIT:
    iteration += 1
    current_sequence = best_sequence[:]

    # --- Destruction phase ---
    d = random.randint(1, 7)
    remove_indices = random.sample(range(len(current_sequence)), d)
    removed_jobs = [current_sequence[i] for i in remove_indices]
    remaining_jobs = [job for i, job in enumerate(current_sequence) if i not in remove_indices]

    print(f"\n[Iteration {iteration}]")
    print(f"랜덤 제거 개수 d = {d}")
    print(f"제거된 Job 리스트 = {removed_jobs}")

    # --- Construction phase ---
    for job in removed_jobs:
        best_pos = None
        best_tard = float('inf')

        for i in range(len(remaining_jobs) + 1):
            temp = remaining_jobs[:i] + [job] + remaining_jobs[i:]
            tard = calc_total_tardiness(temp)
            if tard < best_tard:
                best_tard = tard
                best_pos = i

        remaining_jobs.insert(best_pos, job)

    new_sequence = remaining_jobs[:]
    new_tardiness = calc_total_tardiness(new_sequence)

    # --- Local Search (swap)
    local_seq, local_tard = local_search(new_sequence)

    # --- 개선 확인 ---
    if local_tard < best_tardiness:
        best_tardiness = local_tard
        best_sequence = local_seq[:]
        print(f"✅ 개선됨 → 새 Tardiness = {best_tardiness}")
    else:
        print(f"❌ 개선 없음 (현재: {new_tardiness}, 최고: {best_tardiness})")

    # 시간 경과 표시
    elapsed = time.time() - start_time
    print(f"⏱ 경과 시간: {elapsed:.2f}초")

# -------------------------------
# 🔹 최종 결과
# -------------------------------
end_time = time.time()
print("\n======================")
print("✅ 최종 결과")
print(f"모드: {mode}")
print(f"총 Iteration: {iteration}")
print(f"최종 Tardiness: {best_tardiness}")
print(f"총 소요 시간: {end_time - start_time:.2f}초")
print("======================")
