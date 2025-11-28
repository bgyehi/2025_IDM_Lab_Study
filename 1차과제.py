#1차과제

import itertools
import random

# Total Tardiness 계산 함수
def calculate_tardiness(job_list):
    time = 0
    total_tardiness = 0
    print("작업순서:", [j["id"] for j in job_list])
    for j in job_list:
        time += j["p"]  # 누적 처리시간
        tardiness = max(0, time - j["d"])  # 지각시간
        total_tardiness += tardiness
        print(f"Job {j['id']} 완료시간={time}, 납기={j['d']}, 지각={tardiness}")
    print("총 Tardiness =", total_tardiness, "\n")
    return total_tardiness


# 1. 작은 예제 (A, B, C)
jobs_small = [
    {"id": "A", "p": 3, "d": 2},
    {"id": "B", "p": 4, "d": 4},
    {"id": "C", "p": 2, "d": 5}
]

print("[작은 예제] 생성된 작업들")
for j in jobs_small:
    print(j)

job_num = 100
# Enumeration (최적해)
best_order = None
best_tardiness = 999999
for perm in itertools.permutations(jobs_small):
    time = 0
    total_tardiness = 0
    for j in perm:
        time += j["p"]
        tardiness = max(0, time - j["d"])
        total_tardiness += tardiness
    if total_tardiness < best_tardiness:
        best_tardiness = total_tardiness
        best_order = perm

print("\n[Enumeration - 최적해]")
calculate_tardiness(best_order)


# EDD 정렬
edd_order = []
jobs_copy = jobs_small.copy()
while jobs_copy:
    min_job = jobs_copy[0]
    for job in jobs_copy:
        if job["d"] < min_job["d"]:
            min_job = job
    edd_order.append(min_job)
    jobs_copy.remove(min_job)

print("[EDD]")
calculate_tardiness(edd_order)

# SPT 정렬
spt_order = []
jobs_copy = jobs_small.copy()
while jobs_copy:
    min_job = jobs_copy[0]
    for job in jobs_copy:
        if job["p"] < min_job["p"]:
            min_job = job
    spt_order.append(min_job)
    jobs_copy.remove(min_job)

print("[SPT]")
calculate_tardiness(spt_order)


# 2. 큰 예제 (Job 100개)
print("\n[큰 예제] Job 100개 랜덤 생성")

random.seed(1)  # 랜덤 고정
jobs_large = []
for i in range(job_num):
    job_id = i + 1
    p = random.randint(1, 20)   # 처리시간 1~20
    d = random.randint(10, 500) # 납기 10~500
    jobs_large.append({"id": job_id, "p": p, "d": d})

# EDD 정렬
edd_order_large = []
jobs_copy = jobs_large.copy()
while jobs_copy:
    min_job = jobs_copy[0]
    for job in jobs_copy:
        if job["d"] < min_job["d"]:
            min_job = job
    edd_order_large.append(min_job)
    jobs_copy.remove(min_job)

edd_tardiness = calculate_tardiness(edd_order_large)
print("EDD_Total Tardiness =", edd_tardiness)

# SPT 정렬
spt_order_large = []
jobs_copy = jobs_large.copy()
while jobs_copy:
    min_job = jobs_copy[0]
    for job in jobs_copy:
        if job["p"] < min_job["p"]:
            min_job = job
    spt_order_large.append(min_job)
    jobs_copy.remove(min_job)

spt_tardiness = calculate_tardiness(spt_order_large)
print("SPT_Total Tardiness =", spt_tardiness)
