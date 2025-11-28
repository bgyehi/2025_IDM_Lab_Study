#7차과제_1

import random
import time
import math
import copy
import statistics
import subprocess
import sys
import os
import matplotlib.pyplot as plt
import io  # Added for job data handling


# -------------------------
# Utility: ensure gurobi
# -------------------------
def ensure_gurobi_installed():
    try:
        import gurobipy  # noqa
        print("[INFO] gurobipy already installed.")
        return True
    except ImportError:
        print("[WARNING] gurobipy not installed. Attempting 'pip install gurobipy' ...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gurobipy"], stdout=sys.stdout,
                                  stderr=sys.stderr)
            import gurobipy  # noqa
            print("[INFO] gurobipy installed and import succeeded.")
            return True
        except Exception as e:
            print("[ERROR] Automatic gurobipy install failed:", e)
            print("[INFO] Proceeding without Gurobi.")
            return False


# -------------------------
# Problem utilities
# -------------------------
def compute_tardiness(sequence, processing_times, due_dates):
    t = 0
    total_tardy = 0.0
    for j in sequence:
        t += processing_times[j]
        total_tardy += max(0.0, t - due_dates[j])
    return total_tardy


def schedule_completion_times(sequence, processing_times):
    t = 0
    C = {}
    for j in sequence:
        t += processing_times[j]
        C[j] = t
    return C


# -------------------------
# Neighbourhood ops
# -------------------------
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
    if n <= 2:
        return seq.copy()
    block_size = random.randint(1, min(max_block_size, n - 1))
    i = random.randint(0, n - block_size)
    block = seq[i:i + block_size]
    remainder = seq[:i] + seq[i + block_size:]
    insert_pos = random.randint(0, len(remainder))
    return remainder[:insert_pos] + block + remainder[insert_pos:]


def random_ans_neighbor(seq):
    op = random.choice(['swap', 'insert', 'block'])
    if op == 'swap': return neighbor_swap(seq)
    if op == 'insert': return neighbor_insert(seq)
    return neighbor_block_relocate(seq)


# -------------------------
# SA-ANS (with console logging)
# -------------------------
def sa_ans(processing_times, due_dates, T0=500.0, Tmin=1e-3, K=0.85,
           Imax=300, Vmax=10000, H_init=5, H_max=50, seed=None, time_limit=1800,
           progress_prefix="[SA-ANS]"):
    if seed is not None:
        random.seed(seed)
    n = len(processing_times)

    # initial
    S0 = list(range(n))
    random.shuffle(S0)
    SB = S0.copy()
    T = T0
    V = 0
    H = H_init
    TCB = compute_tardiness(SB, processing_times, due_dates)

    # logging
    best_costs = [TCB]
    best_iters = [0]
    best_times = [0.0]
    iter_count = 0
    start_time = time.time()
    last_print = 0

    print(f"{progress_prefix} start: initial cost={TCB:.2f}, T0={T0}, H_init={H_init}")

    while T >= Tmin and V <= Vmax:
        I_compare = 0
        for i in range(1, Imax + 1):
            iter_count += 1
            if time.time() - start_time > time_limit:
                print(f"{progress_prefix} time limit reached, exiting loops.")
                return {'best_sequence': SB, 'best_cost': TCB, 'timeline': (best_iters, best_costs, best_times),
                        'iterations': iter_count, 'time': time.time() - start_time}

            # generate H candidates
            candidates = [random_ans_neighbor(SB) for _ in range(H)]
            costs = [compute_tardiness(s, processing_times, due_dates) for s in candidates]
            min_idx = min(range(len(costs)), key=lambda i: costs[i])
            SN = candidates[min_idx]
            TCN = costs[min_idx]

            improved = False
            accepted_worse = False
            if TCN < TCB:
                SB = SN.copy();
                TCB = TCN;
                V = 0
                improved = True
                best_costs.append(TCB);
                best_iters.append(iter_count);
                best_times.append(time.time() - start_time)
                print(f"{progress_prefix} iter={iter_count} I={i}/{Imax} T={T:.4f} IMPROVED -> best={TCB:.2f} (H={H})")
            else:
                g = random.random()
                prob = math.exp((TCB - TCN) / T) if T > 0 else 0.0
                if g < prob:
                    SB = SN.copy();
                    TCB = TCN;
                    V = 0
                    accepted_worse = True
                    best_costs.append(TCB);
                    best_iters.append(iter_count);
                    best_times.append(time.time() - start_time)
                    print(
                        f"{progress_prefix} iter={iter_count} I={i}/{Imax} T={T:.4f} accepted-worse -> cost={TCB:.2f} (H={H})")
                else:
                    I_compare += 1
                V += 1

            # periodic status print (every 100 inner iterations or on change)
            if iter_count - last_print >= 100 and not improved and not accepted_worse:
                last_print = iter_count
                print(
                    f"{progress_prefix} iter={iter_count} I={i}/{Imax} T={T:.4f} best={TCB:.2f} V={V} I_compare={I_compare} H={H}")

        T *= K
        if I_compare >= Imax // 2 and H < H_max:
            H = min(H + 1, H_max)
            print(f"{progress_prefix} Increased H -> {H} (I_compare={I_compare})")

    total_time = time.time() - start_time
    print(f"{progress_prefix} finished: iterations={iter_count}, best={TCB:.2f}, time={total_time:.2f}s")
    return {'best_sequence': SB, 'best_cost': TCB, 'timeline': (best_iters, best_costs, best_times),
            'iterations': iter_count, 'time': total_time}


# -------------------------
# Simple SA (with logging)
# -------------------------
def simple_sa(processing_times, due_dates, T0=500.0, Tmin=1e-3, K=0.9, Imax=1000, seed=None, time_limit=1800,
              progress_prefix="[SimpleSA]"):
    if seed is not None:
        random.seed(seed)
    n = len(processing_times)
    S = list(range(n));
    random.shuffle(S)
    best = S.copy();
    best_cost = compute_tardiness(best, processing_times, due_dates)
    T = T0
    iter_count = 0
    start_time = time.time()
    last_print = 0

    print(f"{progress_prefix} start: initial cost={best_cost:.2f}")
    while T > Tmin:
        for i in range(Imax):
            iter_count += 1
            if time.time() - start_time > time_limit:
                print(f"{progress_prefix} time limit reached.")
                return {'best_sequence': best, 'best_cost': best_cost, 'timeline': ([0], [best_cost], [0.0]),
                        'iterations': iter_count, 'time': time.time() - start_time}
            cand = random.choice([neighbor_swap(S), neighbor_insert(S), neighbor_block_relocate(S)])
            cost_c = compute_tardiness(cand, processing_times, due_dates)
            if cost_c < best_cost:
                best = cand.copy();
                best_cost = cost_c;
                S = cand
                print(f"{progress_prefix} iter={iter_count} IMPROVED -> best={best_cost:.2f}")
            else:
                sval = compute_tardiness(S, processing_times, due_dates)
                if random.random() < math.exp((sval - cost_c) / T):
                    S = cand
            if iter_count - last_print >= 200:
                last_print = iter_count
                print(f"{progress_prefix} iter={iter_count} T={T:.4f} best={best_cost:.2f}")
        T *= K
    total_time = time.time() - start_time
    print(f"{progress_prefix} finished: iterations={iter_count}, best={best_cost:.2f}, time={total_time:.2f}s")
    return {'best_sequence': best, 'best_cost': best_cost, 'timeline': ([0], [best_cost], [0.0]),
            'iterations': iter_count, 'time': total_time}


# -------------------------
# Iterated Greedy (with logging)
# -------------------------
def iterated_greedy(processing_times, due_dates, max_restarts=1000, destruct_k=10, time_limit=1800, seed=None,
                    progress_prefix="[IG]"):
    if seed is not None:
        random.seed(seed)
    n = len(processing_times)
    S = sorted(range(n), key=lambda i: processing_times[i])
    best = S.copy();
    best_cost = compute_tardiness(best, processing_times, due_dates)
    timeline_iters = [0];
    timeline_costs = [best_cost];
    timeline_times = [0.0]
    iter_count = 0
    start_time = time.time()

    print(f"{progress_prefix} start: initial (SPT) cost={best_cost:.2f}")

    while time.time() - start_time < time_limit and iter_count < max_restarts:
        iter_count += 1
        k = min(destruct_k, n - 1)
        removed_idx = sorted(random.sample(range(n), k), reverse=True)
        removed = [S[i] for i in removed_idx]
        S_remain = [S[i] for i in range(len(S)) if i not in removed_idx]

        # reconstruct greedily
        for job in removed:
            best_pos = 0;
            best_val = None
            for pos in range(len(S_remain) + 1):
                cand = S_remain[:pos] + [job] + S_remain[pos:]
                val = compute_tardiness(cand, processing_times, due_dates)
                if best_val is None or val < best_val:
                    best_val = val;
                    best_pos = pos
            S_remain.insert(best_pos, job)

        # simple local search (pairwise swap improvement)
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    cand = S_remain.copy()
                    cand[i], cand[j] = cand[j], cand[i]
                    cval = compute_tardiness(cand, processing_times, due_dates)
                    if cval < compute_tardiness(S_remain, processing_times, due_dates):
                        S_remain = cand;
                        improved = True
                        break
                if improved:
                    break

        S = S_remain
        cur_cost = compute_tardiness(S, processing_times, due_dates)
        if cur_cost < best_cost:
            best = S.copy();
            best_cost = cur_cost
            timeline_iters.append(iter_count);
            timeline_costs.append(best_cost);
            timeline_times.append(time.time() - start_time)
            print(f"{progress_prefix} restart={iter_count} IMPROVED -> best={best_cost:.2f}")
        if iter_count % 10 == 0:
            print(f"{progress_prefix} restart={iter_count} current_best={best_cost:.2f}")

    total_time = time.time() - start_time
    print(f"{progress_prefix} finished: restarts={iter_count}, best={best_cost:.2f}, time={total_time:.2f}s")
    return {'best_sequence': best, 'best_cost': best_cost, 'timeline': (timeline_iters, timeline_costs, timeline_times),
            'iterations': iter_count, 'time': total_time}


# -------------------------
# Local Search (hill climb) with logging
# -------------------------
def local_search(processing_times, due_dates, init_seq=None, max_no_improve=5000, time_limit=1800, seed=None,
                 progress_prefix="[LS]"):
    if seed is not None:
        random.seed(seed)
    n = len(processing_times)
    if init_seq is None:
        S = list(range(n));
        random.shuffle(S)
    else:
        S = init_seq.copy()
    best = S.copy();
    best_cost = compute_tardiness(best, processing_times, due_dates)
    no_improve = 0
    iter_count = 0
    start_time = time.time()
    last_print = 0

    print(f"{progress_prefix} start: initial cost={best_cost:.2f}")
    while no_improve < max_no_improve and time.time() - start_time < time_limit:
        iter_count += 1
        cand = random_ans_neighbor(S)
        cval = compute_tardiness(cand, processing_times, due_dates)
        sval = compute_tardiness(S, processing_times, due_dates)
        if cval < sval:
            S = cand
            no_improve = 0
            if cval < best_cost:
                best = cand.copy();
                best_cost = cval
                print(f"{progress_prefix} iter={iter_count} IMPROVED -> best={best_cost:.2f}")
        else:
            no_improve += 1
        if iter_count - last_print >= 500:
            last_print = iter_count
            print(f"{progress_prefix} iter={iter_count} best={best_cost:.2f} no_improve={no_improve}")
    total_time = time.time() - start_time
    print(f"{progress_prefix} finished: iter={iter_count}, best={best_cost:.2f}, time={total_time:.2f}s")
    return {'best_sequence': best, 'best_cost': best_cost, 'timeline': ([0], [best_cost], [0.0]),
            'iterations': iter_count, 'time': total_time}


# -------------------------
# Heuristics
# -------------------------
def heuristic_spt(processing_times):
    return sorted(range(len(processing_times)), key=lambda i: processing_times[i])


def heuristic_edd(due_dates):
    return sorted(range(len(due_dates)), key=lambda i: due_dates[i])


# -------------------------
# Gurobi MIP for total tardiness (with basic logging)
# -------------------------
def gurobi_total_tardiness(processing_times, due_dates, time_limit=1800, progress_prefix="[Gurobi]"):
    try:
        from gurobipy import Model, GRB
    except Exception as e:
        raise ImportError("gurobipy import failed: " + str(e))

    n = len(processing_times)
    Psum = sum(processing_times)
    M = Psum + max(abs(min(due_dates)), 0) + 1000

    m = Model("total_tardiness")
    m.setParam('TimeLimit', time_limit)
    m.setParam('OutputFlag', 0)  # suppress gurobi stdout; we will print summary logs
    # optionally set threads/time-per-node etc.
    # m.setParam('Threads', 4)

    # decision vars x[i,j]
    x = {}
    for i in range(n):
        for j in range(n):
            x[i, j] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    Cpos = {}
    for j in range(n):
        Cpos[j] = m.addVar(lb=0.0, name=f"Cpos_{j}")
    Tjob = {}
    for i in range(n):
        Tjob[i] = m.addVar(lb=0.0, name=f"T_{i}")

    m.update()

    # constraints
    for i in range(n):
        m.addConstr(sum(x[i, j] for j in range(n)) == 1)
    for j in range(n):
        m.addConstr(sum(x[i, j] for i in range(n)) == 1)
    for j in range(n):
        if j == 0:
            m.addConstr(Cpos[0] >= sum(processing_times[i] * x[i, 0] for i in range(n)))
        else:
            m.addConstr(Cpos[j] >= Cpos[j - 1] + sum(processing_times[i] * x[i, j] for i in range(n)))
    for i in range(n):
        for j in range(n):
            m.addConstr(Tjob[i] >= Cpos[j] - due_dates[i] - M * (1 - x[i, j]))

    m.setObjective(sum(Tjob[i] for i in range(n)), GRB.MINIMIZE)

    print(f"{progress_prefix} Starting optimization (time_limit={time_limit}s). This may take long for n={n}...")
    t0 = time.time()
    m.optimize()
    t1 = time.time()
    runtime = t1 - t0

    status = m.Status
    print(f"{progress_prefix} Finished solver call. Status={status}, runtime={runtime:.2f}s")

    if status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED):
        assign = [-1] * n
        for i in range(n):
            for j in range(n):
                try:
                    if x[i, j].X > 0.5:
                        assign[j] = i
                except Exception:
                    pass
        seq = assign.copy()
        obj = sum(Tjob[i].X for i in range(n))
        bestbound = m.ObjBound
        mipgap = m.MIPGap if hasattr(m, 'MIPGap') else None
        print(f"{progress_prefix} Objective (incumbent) = {obj}, BestBound = {bestbound}, MIPGap = {mipgap}")
        return {'sequence': seq, 'objective': obj, 'bestbound': bestbound, 'mipgap': mipgap, 'status': status,
                'runtime': runtime}
    else:
        raise RuntimeError("Gurobi failed with status " + str(status))


# -------------------------
# Runner that executes all algorithms and logs
# -------------------------
def run_all_experiments(n_jobs=100, seed=42, time_limit=1800):
    random.seed(seed)

    # -----------------------------------------------------
    # START: Modified section to use specified job list
    # -----------------------------------------------------
    job_data_str = """
    Job 1: p=45, d=532
    Job 2: p=12, d=986
    Job 3: p=6, d=994
    Job 4: p=22, d=257
    Job 5: p=20, d=434
    Job 6: p=19, d=232
    Job 7: p=13, d=523
    Job 8: p=11, d=610
    Job 9: p=48, d=474
    Job 10: p=39, d=267
    Job 11: p=10, d=416
    Job 12: p=42, d=780
    Job 13: p=32, d=935
    Job 14: p=7, d=522
    Job 15: p=6, d=417
    Job 16: p=10, d=871
    Job 17: p=18, d=711
    Job 18: p=19, d=605
    Job 19: p=37, d=858
    Job 20: p=43, d=669
    Job 21: p=6, d=346
    Job 22: p=40, d=471
    Job 23: p=17, d=342
    Job 24: p=50, d=452
    Job 25: p=46, d=962
    Job 26: p=49, d=774
    Job 27: p=39, d=751
    Job 28: p=31, d=469
    Job 29: p=19, d=964
    Job 30: p=33, d=798
    Job 31: p=42, d=638
    Job 32: p=22, d=797
    Job 33: p=5, d=608
    Job 34: p=15, d=570
    Job 35: p=49, d=424
    Job 36: p=32, d=341
    Job 37: p=26, d=721
    Job 38: p=22, d=705
    Job 39: p=14, d=293
    Job 40: p=18, d=973
    Job 41: p=26, d=248
    Job 42: p=11, d=312
    Job 43: p=10, d=356
    Job 44: p=29, d=842
    Job 45: p=11, d=363
    Job 46: p=27, d=896
    Job 47: p=27, d=632
    Job 48: p=43, d=810
    Job 49: p=21, d=265
    Job 50: p=7, d=594
    Job 51: p=34, d=590
    Job 52: p=39, d=810
    Job 53: p=12, d=679
    Job 54: p=29, d=741
    Job 55: p=10, d=457
    Job 56: p=40, d=766
    Job 57: p=23, d=211
    Job 58: p=45, d=896
    Job 59: p=44, d=938
    Job 60: p=28, d=317
    Job 61: p=41, d=898
    Job 62: p=17, d=749
    Job 63: p=50, d=968
    Job 64: p=9, d=473
    Job 65: p=7, d=987
    Job 66: p=47, d=856
    Job 67: p=19, d=548
    Job 68: p=23, d=314
    Job 69: p=10, d=500
    Job 70: p=19, d=645
    Job 71: p=11, d=361
    Job 72: p=29, d=664
    Job 73: p=22, d=203
    Job 74: p=34, d=939
    Job 75: p=45, d=936
    Job 76: p=28, d=469
    Job 77: p=15, d=712
    Job 78: p=28, d=980
    Job 79: p=27, d=382
    Job 80: p=18, d=719
    Job 81: p=47, d=308
    Job 82: p=22, d=840
    Job 83: p=49, d=505
    Job 84: p=48, d=854
    Job 85: p=46, d=719
    Job 86: p=9, d=823
    Job 87: p=43, d=403
    Job 88: p=45, d=356
    Job 89: p=15, d=582
    Job 90: p=39, d=980
    Job 91: p=20, d=365
    Job 92: p=15, d=752
    Job 93: p=34, d=997
    Job 94: p=29, d=743
    Job 95: p=22, d=200
    Job 96: p=45, d=813
    Job 97: p=49, d=531
    Job 98: p=40, d=700
    Job 99: p=19, d=219
    Job 100: p=48, d=314
    """

    processing_times = []
    due_dates = []

    # Parse the job data string
    for line in io.StringIO(job_data_str):
        if 'Job' in line:
            # Expected format: "Job X: p=P, d=D"
            parts = line.split(':')[-1].strip().split(',')
            p_val = int(parts[0].split('=')[-1].strip())
            d_val = int(parts[1].split('=')[-1].strip())
            processing_times.append(p_val)
            due_dates.append(d_val)

    n_jobs = len(processing_times)
    if n_jobs != 100:
        print(f"[ERROR] Expected 100 jobs, but parsed {n_jobs} jobs. Check input data.")
        # Proceed with what we have, but log the error

    # -----------------------------------------------------
    # END: Modified section
    # -----------------------------------------------------

    # save job list (as before)
    filename = "generated_jobs.txt"
    with open(filename, "w") as f:
        f.write("job_id,processing_time,due_date\n")
        for i in range(n_jobs):
            f.write(f"{i + 1},{processing_times[i]},{due_dates[i]}\n")  # Job ID starting from 1
    print(f"[Main] Saved generated jobs to '{filename}'")

    # print full list to console
    print("[Main] Full job list (job_id: p, d):")
    for i in range(n_jobs):
        print(f"  Job {i + 1}: p={processing_times[i]}, d={due_dates[i]}")
    print("[Main] ---------- end job list ----------\n")

    results = {}
    timelines = {}

    # 1) SPT
    seq_spt = heuristic_spt(processing_times)
    cost_spt = compute_tardiness(seq_spt, processing_times, due_dates)
    print(f"[SPT] Completed. Total tardiness = {cost_spt:.2f}")
    results['SPT'] = {'cost': cost_spt, 'seq': seq_spt}
    timelines['SPT'] = ([0], [cost_spt], [0.0])

    # 2) EDD
    seq_edd = heuristic_edd(due_dates)
    cost_edd = compute_tardiness(seq_edd, processing_times, due_dates)
    print(f"[EDD] Completed. Total tardiness = {cost_edd:.2f}")
    results['EDD'] = {'cost': cost_edd, 'seq': seq_edd}
    timelines['EDD'] = ([0], [cost_edd], [0.0])

    # 3) Local Search starting from SPT
    print("\n[Main] Running Local Search (from SPT)...")
    ls_res = local_search(processing_times, due_dates, init_seq=seq_spt, max_no_improve=2000, time_limit=300, seed=seed,
                          progress_prefix="[LS-SPT]")
    results['LocalSearch'] = {'cost': ls_res['best_cost'], 'seq': ls_res['best_sequence']}
    timelines['LocalSearch'] = ls_res['timeline']

    # 4) Simple SA
    print("\n[Main] Running Simple SA...")
    sa_res = simple_sa(processing_times, due_dates, T0=500, Tmin=1e-3, K=0.9, Imax=500, seed=seed, time_limit=300,
                       progress_prefix="[SimpleSA]")
    results['SimpleSA'] = {'cost': sa_res['best_cost'], 'seq': sa_res['best_sequence']}
    timelines['SimpleSA'] = sa_res['timeline']

    # 5) SA-ANS
    print("\n[Main] Running SA-ANS...")
    sa_ans_res = sa_ans(processing_times, due_dates, T0=500, Tmin=1e-3, K=0.85, Imax=300, Vmax=10000, H_init=5,
                        H_max=50, seed=seed, time_limit=600, progress_prefix="[SA-ANS]")
    results['SA-ANS'] = {'cost': sa_ans_res['best_cost'], 'seq': sa_ans_res['best_sequence']}
    timelines['SA-ANS'] = sa_ans_res['timeline']

    # 6) Iterated Greedy
    print("\n[Main] Running Iterated Greedy (IG)...")
    ig_res = iterated_greedy(processing_times, due_dates, max_restarts=200, destruct_k=max(5, n_jobs // 20),
                             time_limit=300, seed=seed, progress_prefix="[IG]")
    results['IG'] = {'cost': ig_res['best_cost'], 'seq': ig_res['best_sequence']}
    timelines['IG'] = ig_res['timeline']

    # 7) Random-restart Local Search
    print("\n[Main] Running Random-restart Local Search...")
    best_rand_ls_cost = float('inf');
    best_rand_ls_seq = None
    rr_iters = [0];
    rr_costs = [float('inf')];
    rr_times = [0.0]
    rstart_time = time.time();
    rr_iter = 0
    while time.time() - rstart_time < 200:
        rr_iter += 1
        init = list(range(n_jobs));
        random.shuffle(init)
        rls = local_search(processing_times, due_dates, init_seq=init, max_no_improve=500, time_limit=5,
                           seed=random.randint(0, 10 ** 6), progress_prefix=f"[RandLS-{rr_iter}]")
        if rls['best_cost'] < best_rand_ls_cost:
            best_rand_ls_cost = rls['best_cost'];
            best_rand_ls_seq = rls['best_sequence']
            rr_iters.append(rr_iter);
            rr_costs.append(best_rand_ls_cost);
            rr_times.append(time.time() - rstart_time)
            print(f"[RandLS] new best {best_rand_ls_cost:.2f} at restart {rr_iter}")
    results['RandLS'] = {'cost': best_rand_ls_cost, 'seq': best_rand_ls_seq}
    timelines['RandLS'] = (rr_iters, rr_costs, rr_times)

    # 8) Gurobi attempt (with install attempt)
    print("\n[Main] Preparing Gurobi...")
    gurobi_installed = ensure_gurobi_installed()
    if gurobi_installed:
        try:
            gurobi_info = gurobi_total_tardiness(processing_times, due_dates, time_limit=time_limit,
                                                 progress_prefix="[Gurobi]")
            seq_g = gurobi_info['sequence'];
            obj_g = gurobi_info['objective'];
            bound = gurobi_info.get('bestbound', None);
            gap = gurobi_info.get('mipgap', None)
            results['Gurobi'] = {'cost': obj_g, 'seq': seq_g, 'bound': bound, 'mipgap': gap}
            timelines['Gurobi'] = ([0], [obj_g], [0.0])
        except Exception as e:
            print("[Gurobi] Failed to run:", e)
            results['Gurobi'] = None
            timelines['Gurobi'] = None
    else:
        results['Gurobi'] = None
        timelines['Gurobi'] = None

    # Final summary
    print("\n[Main] ===== FINAL SUMMARY (Total Tardiness) =====")
    for name, r in results.items():
        if r is None:
            print(f"  {name}: skipped/failed")
        else:
            if name == 'Gurobi' and r.get('mipgap') is not None:
                print(f"  {name}: cost={r['cost']:.2f}, mipgap={r['mipgap']}")
            else:
                print(f"  {name}: cost={r['cost']:.2f}")

    return processing_times, due_dates, results, timelines


# -------------------------
# Plotting timelines
# -------------------------
def plot_all_timelines(timelines, title="Algorithm comparison (total tardiness)"):
    plt.figure(figsize=(12, 8))
    for name, tl in timelines.items():
        if tl is None:
            continue
        iters, costs, _ = tl
        plt.step(iters, costs, where='post', label=name)
    plt.xlabel("Iteration / improvement step")
    plt.ylabel("Total tardiness")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results_comparison.png")
    plt.show()
    print("[Plot] saved to results_comparison.png and displayed.")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # parameters
    n_jobs = 100
    seed = 42
    time_limit = 1800  # Gurobi time limit; other algs have smaller caps as coded

    print("[Main] Starting full experiment with SPECIFIED 100 JOBS...")
    processing_times, due_dates, results, timelines = run_all_experiments(n_jobs=n_jobs, seed=seed,
                                                                          time_limit=time_limit)
    print("[Main] Plotting results...")
    plot_all_timelines(timelines, title=f"Comparison of algorithms on {n_jobs} specified jobs (tardiness)")

    print("[Main] Done.")