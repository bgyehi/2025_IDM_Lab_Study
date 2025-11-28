"""
sa_ans_compare_full_fixed_jobs.py

- 100개의 고정된 Job 데이터(jobs_data)를 사용
- 모든 알고리즘(SPT, EDD, LocalSearch, SimpleSA, SA-ANS, IG, RandRestart LS, Gurobi) 실행
- 실행 후 각 알고리즘 최종 Total Tardiness 콘솔 요약 출력 (표 형태)
"""

import random
import time
import math
import copy
import subprocess
import sys
import os
import matplotlib.pyplot as plt

# ------------------------------------------------
#  100개 고정 작업 데이터
# ------------------------------------------------
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
    (96, 45, 813), (97, 49, 531), (98, 40, 700), (99, 19, 219), (100, 48, 314),
]

processing_times = [p for (_, p, _) in jobs_data]
due_dates = [d for (_, _, d) in jobs_data]
n_jobs = len(processing_times)

# ------------------------------------------------
# 기본 함수
# ------------------------------------------------
def compute_tardiness(sequence, processing_times, due_dates):
    t = 0
    total = 0
    for j in sequence:
        t += processing_times[j]
        total += max(0, t - due_dates[j])
    return total

def neighbor_swap(seq):
    s = seq.copy()
    a, b = random.sample(range(len(seq)), 2)
    s[a], s[b] = s[b], s[a]
    return s

def neighbor_insert(seq):
    s = seq.copy()
    a, b = random.sample(range(len(seq)), 2)
    x = s.pop(a)
    s.insert(b, x)
    return s

def neighbor_block_relocate(seq, max_block_size=5):
    n = len(seq)
    block_size = random.randint(1, min(max_block_size, n - 1))
    i = random.randint(0, n - block_size)
    block = seq[i:i+block_size]
    rest = seq[:i] + seq[i+block_size:]
    pos = random.randint(0, len(rest))
    return rest[:pos] + block + rest[pos:]

def random_ans_neighbor(seq):
    op = random.choice(["swap", "insert", "block"])
    if op == "swap": return neighbor_swap(seq)
    if op == "insert": return neighbor_insert(seq)
    return neighbor_block_relocate(seq)

# ------------------------------------------------
# Local Search
# ------------------------------------------------
def local_search(processing_times, due_dates, init_seq=None,
                 max_no_improve=5000, time_limit=1800, seed=None,
                 progress_prefix="[LS]"):
    if seed: random.seed(seed)
    n=len(processing_times)
    S = init_seq.copy() if init_seq else list(range(n))
    if not init_seq: random.shuffle(S)
    best = S.copy()
    best_cost = compute_tardiness(best, processing_times, due_dates)
    no=0
    start=time.time()

    while no < max_no_improve and time.time()-start < time_limit:
        cand=random_ans_neighbor(S)
        cval=compute_tardiness(cand,processing_times,due_dates)
        sval=compute_tardiness(S,processing_times,due_dates)
        if cval < sval:
            S=cand
            no=0
            if cval < best_cost:
                best=cand.copy()
                best_cost=cval
        else:
            no+=1

    return {'best_sequence':best, 'best_cost':best_cost, 'timeline': ([0],[best_cost],[0])}

# ------------------------------------------------
# Simple SA
# ------------------------------------------------
def simple_sa(processing_times, due_dates, T0=500.0, Tmin=1e-3,
              K=0.9, Imax=1000, seed=None, time_limit=1800):
    if seed: random.seed(seed)
    n=len(processing_times)
    S=list(range(n))
    random.shuffle(S)
    best=S.copy()
    best_cost=compute_tardiness(best, processing_times, due_dates)
    T=T0
    start=time.time()
    while T>Tmin:
        for _ in range(Imax):
            if time.time()-start > time_limit: break
            cand=random.choice([neighbor_swap(S), neighbor_insert(S), neighbor_block_relocate(S)])
            cval=compute_tardiness(cand, processing_times, due_dates)
            sval=compute_tardiness(S, processing_times, due_dates)
            if cval<sval or random.random()<math.exp((sval-cval)/T):
                S=cand
            if cval<best_cost:
                best=cand.copy()
                best_cost=cval
        T*=K
    return {'best_sequence':best, 'best_cost':best_cost, 'timeline': ([0],[best_cost],[0])}

# ------------------------------------------------
# SA-ANS
# ------------------------------------------------
def sa_ans(processing_times, due_dates, T0=500.0, Tmin=1e-3, K=0.85,
           Imax=300, H_init=5, H_max=50, seed=None, time_limit=1800):
    if seed: random.seed(seed)
    n=len(processing_times)
    S=list(range(n))
    random.shuffle(S)
    SB=S.copy()
    T=T0
    V=0
    H=H_init
    TCB=compute_tardiness(SB, processing_times, due_dates)
    iter_count=0
    start=time.time()
    while T>=Tmin and V<=10000:
        I_compare=0
        for _ in range(Imax):
            if time.time()-start>time_limit: break
            candidates=[random_ans_neighbor(SB) for _ in range(H)]
            costs=[compute_tardiness(s, processing_times, due_dates) for s in candidates]
            idx=min(range(H), key=lambda x: costs[x])
            SN=candidates[idx]
            TCN=costs[idx]
            iter_count+=1
            if TCN<TCB:
                SB=SN.copy()
                TCB=TCN
                V=0
            else:
                if random.random() < math.exp((TCB-TCN)/T):
                    SB=SN.copy()
                    TCB=TCN
                    V=0
                else:
                    I_compare+=1
                V+=1
        T*=K
        if I_compare>=Imax//2 and H<H_max:
            H+=1
    return {'best_sequence':SB, 'best_cost':TCB, 'timeline': ([0],[TCB],[0])}

# ------------------------------------------------
# Iterated Greedy
# ------------------------------------------------
def iterated_greedy(processing_times, due_dates, max_restarts=200,
                    destruct_k=10, time_limit=1800, seed=None):
    if seed: random.seed(seed)
    n=len(processing_times)
    S=sorted(range(n), key=lambda x: processing_times[x])
    best=S.copy()
    best_cost=compute_tardiness(best, processing_times, due_dates)
    for _ in range(max_restarts):
        k=min(destruct_k,n-1)
        removed_idx=sorted(random.sample(range(n), k), reverse=True)
        removed=[S[i] for i in removed_idx]
        remain=[S[i] for i in range(len(S)) if i not in removed_idx]
        for job in removed:
            best_pos=0
            best_val=None
            for pos in range(len(remain)+1):
                cand=remain[:pos]+[job]+remain[pos:]
                val=compute_tardiness(cand, processing_times, due_dates)
                if best_val is None or val<best_val:
                    best_val=val
                    best_pos=pos
            remain.insert(best_pos, job)
        S=remain
        cost=compute_tardiness(S, processing_times, due_dates)
        if cost<best_cost:
            best=S.copy()
            best_cost=cost
    return {'best_sequence':best, 'best_cost':best_cost, 'timeline': ([0],[best_cost],[0])}

# ------------------------------------------------
# 결과 요약 출력 (표 형태)
# ------------------------------------------------
def print_results_summary(results):
    print("\n=== Final Total Tardiness Summary ===")
    print(f"{'Algorithm':15s} | {'Total Tardiness':>15s}")
    print("-"*33)
    for algo, res in results.items():
        cost = res['cost'] if 'cost' in res else res['best_cost']
        print(f"{algo:15s} | {cost:15d}")
    print("="*33 + "\n")

# ------------------------------------------------
# 실행
# ------------------------------------------------
def run_all_experiments():
    results={}
    timelines={}

    # SPT
    seq_spt=sorted(range(n_jobs), key=lambda i: processing_times[i])
    cost_spt=compute_tardiness(seq_spt, processing_times, due_dates)
    results['SPT']={'cost':cost_spt, 'seq':seq_spt}
    timelines['SPT']=([0],[cost_spt],[0])

    # EDD
    seq_edd=sorted(range(n_jobs), key=lambda i: due_dates[i])
    cost_edd=compute_tardiness(seq_edd, processing_times, due_dates)
    results['EDD']={'cost':cost_edd, 'seq':seq_edd}
    timelines['EDD']=([0],[cost_edd],[0])

    # Local Search
    results['LocalSearch']=local_search(processing_times, due_dates, init_seq=seq_spt)
    timelines['LocalSearch']=results['LocalSearch']['timeline']

    # Simple SA
    results['SimpleSA']=simple_sa(processing_times, due_dates)
    timelines['SimpleSA']=results['SimpleSA']['timeline']

    # SA-ANS
    results['SA-ANS']=sa_ans(processing_times, due_dates)
    timelines['SA-ANS']=results['SA-ANS']['timeline']

    # Iterated Greedy
    results['IG']=iterated_greedy(processing_times, due_dates)
    timelines['IG']=results['IG']['timeline']

    # Random Restart LS
    best_rr_cost=float("inf")
    best_rr_seq=None
    for _ in range(10):
        init=list(range(n_jobs))
        random.shuffle(init)
        rls=local_search(processing_times, due_dates, init_seq=init, max_no_improve=100, time_limit=1)
        if rls['best_cost']<best_rr_cost:
            best_rr_cost=rls['best_cost']
            best_rr_seq=rls['best_sequence']
    results['RandLS']={'cost':best_rr_cost, 'seq':best_rr_seq}
    timelines['RandLS']=([0],[best_rr_cost],[0])
    # --- 요약 출력 ---
    print("\n================ TARDINESS SUMMARY ================")
    for name, data in results.items():
        if isinstance(data, dict) and 'best_cost' in data:
            print(f"{name:15s} : {data['best_cost']}")
        elif isinstance(data, dict) and 'cost' in data:
            print(f"{name:15s} : {data['cost']}")
    print("====================================================")

    return processing_times, due_dates, results, timelines

# ------------------------------------------------
# Plotting
# ------------------------------------------------
def plot_all_timelines(timelines, title="Algorithm comparison"):
    plt.figure(figsize=(12,8))
    for name, tl in timelines.items():
        if tl:
            it,cost,_=tl
            plt.step(it,cost,where='post',label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Total Tardiness")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results_fixed_jobs.png")
    plt.show()
    print("[Plot] saved results_fixed_jobs.png")

# ------------------------------------------------
# Main
# ------------------------------------------------
if __name__=="__main__":
    processing_times,due_dates,results,timelines=run_all_experiments()
    print_results_summary(results)
    plot_all_timelines(timelines, title="Fixed 100 jobs – Algorithm Comparison")
