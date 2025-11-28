"""
sa_ans_compare_full_fixed_jobs.py

- 100개의 고정된 Job 데이터(jobs_data)를 사용
- 모든 알고리즘(SPT, EDD, LocalSearch, SimpleSA, SA-ANS, IG, RandRestart LS, Gurobi) 실행
"""

import random
import time
import math
import copy
import statistics
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

# 고정 데이터 → processing_times, due_dates 추출
processing_times = [p for (_, p, _) in jobs_data]
due_dates = [d for (_, _, d) in jobs_data]
n_jobs = len(processing_times)

# ------------------------------------------------
# 이하 기존 알고리즘 코드 그대로 유지
# ------------------------------------------------

def ensure_gurobi_installed():
    try:
        import gurobipy
        print("[INFO] gurobipy already installed.")
        return True
    except ImportError:
        print("[WARNING] gurobipy not installed. Trying install...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gurobipy"], stdout=sys.stdout, stderr=sys.stderr)
            import gurobipy
            print("[INFO] install success.")
            return True
        except Exception:
            print("[ERROR] install failed. Continue without Gurobi.")
            return False

def compute_tardiness(sequence, processing_times, due_dates):
    t = 0
    total = 0
    for j in sequence:
        t += processing_times[j]
        total += max(0, t - due_dates[j])
    return total

def schedule_completion_times(sequence, processing_times):
    t = 0
    C = {}
    for j in sequence:
        t += processing_times[j]
        C[j] = t
    return C

# --- Neighbors ---
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

# --- SA-ANS ---
def sa_ans(processing_times, due_dates, T0=500.0, Tmin=1e-3, K=0.85,
           Imax=300, Vmax=10000, H_init=5, H_max=50, seed=None,
           time_limit=1800, progress_prefix="[SA-ANS]"):

    if seed is not None:
        random.seed(seed)
    n = len(processing_times)

    S0 = list(range(n))
    random.shuffle(S0)
    SB = S0.copy()
    T = T0
    V = 0
    H = H_init
    TCB = compute_tardiness(SB, processing_times, due_dates)

    best_costs = [TCB]
    best_iters = [0]
    best_times = [0.0]
    iter_count = 0
    start = time.time()

    print(f"{progress_prefix} start cost={TCB}")

    while T >= Tmin and V <= Vmax:
        I_compare = 0
        for i in range(1, Imax+1):

            if time.time() - start > time_limit:
                return {'best_sequence': SB, 'best_cost': TCB,
                        'timeline': (best_iters, best_costs, best_times)}

            candidates = [random_ans_neighbor(SB) for _ in range(H)]
            costs = [compute_tardiness(s, processing_times, due_dates) for s in candidates]
            idx = min(range(H), key=lambda x: costs[x])
            SN = candidates[idx]
            TCN = costs[idx]

            iter_count += 1

            improved = False
            accepted = False

            if TCN < TCB:
                SB = SN.copy()
                TCB = TCN
                V = 0
                improved = True
                best_costs.append(TCB)
                best_iters.append(iter_count)
                best_times.append(time.time() - start)
                print(f"{progress_prefix} iter={iter_count} IMPROVED {TCB}")
            else:
                prob = math.exp((TCB - TCN)/T)
                if random.random() < prob:
                    SB = SN.copy()
                    TCB = TCN
                    accepted = True
                    V = 0
                    best_costs.append(TCB)
                    best_iters.append(iter_count)
                    best_times.append(time.time() - start)
                    print(f"{progress_prefix} iter={iter_count} accepted worse {TCB}")
                else:
                    I_compare += 1
                V += 1

        T *= K
        if I_compare >= Imax//2 and H < H_max:
            H += 1
            print(f"{progress_prefix} H increased -> {H}")

    return {'best_sequence': SB, 'best_cost': TCB,
            'timeline': (best_iters, best_costs, best_times)}

# --- Simple SA ---
def simple_sa(processing_times, due_dates, T0=500.0, Tmin=1e-3,
              K=0.9, Imax=1000, seed=None, time_limit=1800,
              progress_prefix="[SimpleSA]"):

    if seed: random.seed(seed)

    n = len(processing_times)
    S = list(range(n))
    random.shuffle(S)
    best = S.copy()
    best_cost = compute_tardiness(best, processing_times, due_dates)
    T = T0
    iter_count = 0
    start = time.time()

    print(f"{progress_prefix} start cost={best_cost}")

    while T > Tmin:
        for _ in range(Imax):
            if time.time() - start > time_limit:
                return {'best_sequence': best, 'best_cost': best_cost,
                        'timeline': ([0], [best_cost], [0])}

            cand = random.choice([neighbor_swap(S), neighbor_insert(S), neighbor_block_relocate(S)])
            cval = compute_tardiness(cand, processing_times, due_dates)

            sval = compute_tardiness(S, processing_times, due_dates)
            if cval < sval or random.random() < math.exp((sval - cval)/T):
                S = cand
            if cval < best_cost:
                best = cand.copy()
                best_cost = cval
                print(f"{progress_prefix} improved -> {best_cost}")
        T *= K

    return {'best_sequence': best, 'best_cost': best_cost,
            'timeline': ([0], [best_cost], [0])}

# --- Iterated Greedy ---
def iterated_greedy(processing_times, due_dates, max_restarts=200,
                    destruct_k=10, time_limit=1800, seed=None,
                    progress_prefix="[IG]"):

    if seed: random.seed(seed)
    n = len(processing_times)

    S = sorted(range(n), key=lambda x: processing_times[x])
    best = S.copy()
    best_cost = compute_tardiness(best, processing_times, due_dates)

    timeline_iters=[0]; timeline_costs=[best_cost]; timeline_times=[0]

    print(f"{progress_prefix} start (SPT cost={best_cost})")

    start = time.time()
    for r in range(1, max_restarts+1):
        if time.time() - start > time_limit: break

        k = min(destruct_k, n-1)
        removed_idx = sorted(random.sample(range(n), k), reverse=True)
        removed = [S[i] for i in removed_idx]
        remain = [S[i] for i in range(len(S)) if i not in removed_idx]

        for job in removed:
            best_pos = 0
            best_val = None
            for pos in range(len(remain)+1):
                cand = remain[:pos] + [job] + remain[pos:]
                val = compute_tardiness(cand, processing_times, due_dates)
                if best_val is None or val < best_val:
                    best_val = val
                    best_pos = pos
            remain.insert(best_pos, job)

        S = remain
        cost = compute_tardiness(S, processing_times, due_dates)
        if cost < best_cost:
            best = S.copy()
            best_cost = cost
            timeline_iters.append(r)
            timeline_costs.append(best_cost)
            timeline_times.append(time.time() - start)
            print(f"{progress_prefix} restart={r} improved -> {best_cost}")

    return {'best_sequence':best, 'best_cost':best_cost,
            'timeline': (timeline_iters, timeline_costs, timeline_times)}

# --- Local Search ---
def local_search(processing_times, due_dates, init_seq=None,
                 max_no_improve=5000, time_limit=1800, seed=None,
                 progress_prefix="[LS]"):

    if seed: random.seed(seed)
    n=len(processing_times)

    if init_seq: S = init_seq.copy()
    else:
        S=list(range(n)); random.shuffle(S)

    best=S.copy()
    best_cost=compute_tardiness(best, processing_times, due_dates)

    no=0
    iter_count=0
    start=time.time()

    print(f"{progress_prefix} start cost={best_cost}")

    while no < max_no_improve and time.time()-start < time_limit:
        iter_count+=1
        cand=random_ans_neighbor(S)
        cval=compute_tardiness(cand,processing_times,due_dates)
        sval=compute_tardiness(S,processing_times,due_dates)

        if cval < sval:
            S=cand
            no=0
            if cval < best_cost:
                best=cand.copy()
                best_cost=cval
                print(f"{progress_prefix} improved -> {best_cost}")
        else:
            no+=1

    return {'best_sequence':best, 'best_cost':best_cost,
            'timeline': ([0],[best_cost],[0])}

# ------------------------------------------------
# 알고리즘 실행
# ------------------------------------------------
def run_all_experiments():

    results = {}
    timelines = {}

    # --- SPT ---
    seq_spt = sorted(range(n_jobs), key=lambda i: processing_times[i])
    cost_spt = compute_tardiness(seq_spt, processing_times, due_dates)
    print(f"[SPT] cost={cost_spt}")
    results['SPT'] = {'cost':cost_spt, 'seq':seq_spt}
    timelines['SPT'] = ([0],[cost_spt],[0])

    # --- EDD ---
    seq_edd = sorted(range(n_jobs), key=lambda i: due_dates[i])
    cost_edd = compute_tardiness(seq_edd, processing_times, due_dates)
    print(f"[EDD] cost={cost_edd}")
    results['EDD'] = {'cost':cost_edd, 'seq':seq_edd}
    timelines['EDD'] = ([0],[cost_edd],[0])

    # --- Local Search from SPT ---
    print("\n[Main] Running LS from SPT...")
    ls_res = local_search(processing_times, due_dates, init_seq=seq_spt,
                          max_no_improve=2000, time_limit=300,
                          seed=42, progress_prefix="[LS-SPT]")
    results['LocalSearch'] = ls_res
    timelines['LocalSearch'] = ls_res['timeline']

    # --- Simple SA ---
    print("\n[Main] Running Simple SA...")
    sa_res = simple_sa(processing_times, due_dates, T0=500,
                       Tmin=1e-3, K=0.9, Imax=500,
                       seed=42, time_limit=300)
    results['SimpleSA'] = sa_res
    timelines['SimpleSA'] = sa_res['timeline']

    # --- SA-ANS ---
    print("\n[Main] Running SA-ANS...")
    sa_ans_res = sa_ans(processing_times, due_dates, T0=500,
                        Tmin=1e-3, K=0.85, Imax=300,
                        seed=42, time_limit=600)
    results['SA-ANS'] = sa_ans_res
    timelines['SA-ANS'] = sa_ans_res['timeline']

    # --- Iterated Greedy ---
    print("\n[Main] Running IG...")
    ig_res = iterated_greedy(processing_times, due_dates,
                             max_restarts=200,
                             destruct_k=max(5, n_jobs//20),
                             time_limit=300,
                             seed=42)
    results['IG'] = ig_res
    timelines['IG'] = ig_res['timeline']

    # --- Random Restart LS ---
    print("\n[Main] Running Random Restart LS...")
    best_rr_cost = float("inf")
    best_rr_seq = None
    rr_iters=[0]; rr_costs=[float("inf")]; rr_times=[0]

    start = time.time()
    rr_count=0
    while time.time() - start < 200:
        rr_count+=1
        init = list(range(n_jobs))
        random.shuffle(init)
        rls=local_search(processing_times, due_dates, init_seq=init,
                         max_no_improve=500, time_limit=5,
                         seed=random.randint(0, 999999),
                         progress_prefix=f"[RandLS-{rr_count}]")
        if rls['best_cost'] < best_rr_cost:
            best_rr_cost = rls['best_cost']
            best_rr_seq = rls['best_sequence']
            rr_iters.append(rr_count)
            rr_costs.append(best_rr_cost)
            rr_times.append(time.time() - start)
            print(f"[RandLS] new best {best_rr_cost}")

    results['RandLS'] = {'cost':best_rr_cost, 'seq':best_rr_seq}
    timelines['RandLS'] = (rr_iters, rr_costs, rr_times)
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
            it, cost, _ = tl
            plt.step(it, cost, where='post', label=name)
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
if __name__ == "__main__":
    print("[Main] Fixed 100 jobs experiment start...")
    processing_times, due_dates, results, timelines = run_all_experiments()
    print("[Main] Plotting...")
    plot_all_timelines(timelines, title="Fixed 100 jobs – Algorithm Comparison")
    print("[Main] Done.")
# --- 요약 출력 ---
    print("\n================ TARDINESS SUMMARY ================")
    for name, data in results.items():
        if isinstance(data, dict) and 'best_cost' in data:
            print(f"{name:15s} : {data['best_cost']}")
        elif isinstance(data, dict) and 'cost' in data:
            print(f"{name:15s} : {data['cost']}")
    print("====================================================")