"""
sa_ans_compare.py (FULL VERSION)

- Generate 100 random jobs (processing time, due date)
- Save full job list to a file (generated_jobs.txt)
- If gurobi is not installed, attempt automatic installation
- Run 8 algorithms on the same job list and compare total tardiness
- Algorithms: SA-ANS, IG, Gurobi MIP, SPT, EDD, Local Search, Simple SA, RandomSearch+LS
- Record improvements over time/iterations and plot comparison using matplotlib.
- Time limit for heavy solvers set to 1800 seconds.
"""

import random
import time
import math
import copy
import statistics
import subprocess
import sys
import matplotlib.pyplot as plt

# ===================================================
# Optional: Try to install gurobi if missing
# ===================================================
def ensure_gurobi_installed():
    try:
        import gurobipy   # noqa
        print("[INFO] gurobipy already installed.")
        return True
    except ImportError:
        print("[WARNING] gurobipy not installed. Attempting installation...")

        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gurobipy"])
            print("[INFO] gurobipy installation completed. Re-trying import...")
            import gurobipy   # noqa
            print("[INFO] gurobipy import successful.")
            return True
        except Exception as e:
            print("[ERROR] Automatic installation of gurobipy failed:", e)
            print("[INFO] Continue without Gurobi.")
            return False


# ===================================================
# Utilities
# ===================================================
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


# ===================================================
# Neighbourhood operations
# ===================================================
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
    if op == 'swap':
        return neighbor_swap(seq)
    if op == 'insert':
        return neighbor_insert(seq)
    return neighbor_block_relocate(seq)


# ===================================================
# SA-ANS implementation
# ===================================================
def sa_ans(processing_times, due_dates, T0=500.0, Tmin=1e-3, K=0.85,
           Imax=300, Vmax=1000, H_init=5, H_max=50, seed=None, time_limit=1800):

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
    start_time = time.time()

    while T >= Tmin and V <= Vmax:
        I_compare = 0
        for _ in range(Imax):
            iter_count += 1
            if time.time() - start_time > time_limit:
                return dict(best_sequence=SB, best_cost=TCB,
                            timeline=(best_iters, best_costs, best_times),
                            iterations=iter_count, time=time.time() - start_time)

            candidates = [random_ans_neighbor(SB) for _ in range(H)]
            costs = [compute_tardiness(s, processing_times, due_dates) for s in candidates]

            min_idx = min(range(len(costs)), key=lambda i: costs[i])
            SN = candidates[min_idx]
            TCN = costs[min_idx]

            if TCN < TCB:
                SB = SN.copy()
                TCB = TCN
                V = 0
                best_costs.append(TCB)
                best_iters.append(iter_count)
                best_times.append(time.time() - start_time)
            else:
                g = random.random()
                prob = math.exp((TCB - TCN) / T) if T > 0 else 0.0
                if g < prob:
                    SB = SN.copy()
                    TCB = TCN
                    V = 0
                    best_costs.append(TCB)
                    best_iters.append(iter_count)
                    best_times.append(time.time() - start_time)
                else:
                    I_compare += 1
                V += 1

        T *= K
        if I_compare >= Imax // 2 and H < H_max:
            H = min(H + 1, H_max)

    return dict(best_sequence=SB, best_cost=TCB,
                timeline=(best_iters, best_costs, best_times),
                iterations=iter_count, time=time.time() - start_time)


# ===================================================
# Simple SA
# ===================================================
def simple_sa(processing_times, due_dates, T0=500.0, Tmin=1e-3, K=0.9,
              Imax=1000, seed=None, time_limit=1800):

    if seed is not None:
        random.seed(seed)
    n = len(processing_times)

    S = list(range(n))
    random.shuffle(S)
    best = S.copy()
    best_cost = compute_tardiness(best, processing_times, due_dates)

    T = T0
    iter_count = 0
    start_time = time.time()

    timeline_iters = [0]
    timeline_costs = [best_cost]
    timeline_times = [0.0]

    while T > Tmin:
        for _ in range(Imax):
            iter_count += 1
            if time.time() - start_time > time_limit:
                return dict(best_sequence=best, best_cost=best_cost,
                            timeline=(timeline_iters, timeline_costs, timeline_times),
                            iterations=iter_count, time=time.time() - start_time)

            cand = random.choice([neighbor_swap(S), neighbor_insert(S), neighbor_block_relocate(S)])
            cost_c = compute_tardiness(cand, processing_times, due_dates)
            if cost_c < best_cost:
                best = cand.copy()
                best_cost = cost_c
                timeline_iters.append(iter_count)
                timeline_costs.append(best_cost)
                timeline_times.append(time.time() - start_time)
                S = cand
            else:
                if random.random() < math.exp((compute_tardiness(S, processing_times, due_dates) - cost_c) / T):
                    S = cand
        T *= K

    return dict(best_sequence=best, best_cost=best_cost,
                timeline=(timeline_iters, timeline_costs, timeline_times),
                iterations=iter_count, time=time.time() - start_time)


# ===================================================
# IG
# ===================================================
def iterated_greedy(processing_times, due_dates,
                    max_restarts=1000, destruct_k=10,
                    time_limit=1800, seed=None):
    if seed is not None:
        random.seed(seed)
    n = len(processing_times)

    S = sorted(range(n), key=lambda i: processing_times[i])
    best = S.copy()
    best_cost = compute_tardiness(best, processing_times, due_dates)

    timeline_iters = [0]
    timeline_costs = [best_cost]
    timeline_times = [0.0]

    iter_count = 0
    start_time = time.time()

    while time.time() - start_time < time_limit and iter_count < max_restarts:
        iter_count += 1

        k = min(destruct_k, n - 1)
        removed_idx = sorted(random.sample(range(n), k), reverse=True)
        removed = [S[i] for i in removed_idx]
        S_remain = [S[i] for i in range(len(S)) if i not in removed_idx]

        for job in removed:
            best_pos = 0
            best_val = None
            for pos in range(len(S_remain) + 1):
                cand = S_remain[:pos] + [job] + S_remain[pos:]
                val = compute_tardiness(cand, processing_times, due_dates)
                if best_val is None or val < best_val:
                    best_val = val
                    best_pos = pos
            S_remain.insert(best_pos, job)

        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    cand = S_remain.copy()
                    cand[i], cand[j] = cand[j], cand[i]
                    cval = compute_tardiness(cand, processing_times, due_dates)
                    if cval < compute_tardiness(S_remain, processing_times, due_dates):
                        S_remain = cand
                        improved = True
                        break
                if improved:
                    break

        S = S_remain
        cur_cost = compute_tardiness(S, processing_times, due_dates)
        if cur_cost < best_cost:
            best = S.copy()
            best_cost = cur_cost
            timeline_iters.append(iter_count)
            timeline_costs.append(best_cost)
            timeline_times.append(time.time() - start_time)

    return dict(best_sequence=best, best_cost=best_cost,
                timeline=(timeline_iters, timeline_costs, timeline_times),
                iterations=iter_count, time=time.time() - start_time)


# ===================================================
# Local Search (hill climb)
# ===================================================
def local_search(processing_times, due_dates, init_seq=None,
                 max_no_improve=5000, time_limit=1800, seed=None):

    if seed is not None:
        random.seed(seed)
    n = len(processing_times)

    if init_seq is None:
        S = list(range(n))
        random.shuffle(S)
    else:
        S = init_seq.copy()

    best = S.copy()
    best_cost = compute_tardiness(best, processing_times, due_dates)

    no_improve = 0
    iter_count = 0
    start_time = time.time()

    timeline_iters = [0]
    timeline_costs = [best_cost]
    timeline_times = [0.0]

    while no_improve < max_no_improve and time.time() - start_time < time_limit:
        iter_count += 1

        cand = random_ans_neighbor(S)
        cval = compute_tardiness(cand, processing_times, due_dates)
        sval = compute_tardiness(S, processing_times, due_dates)

        if cval < sval:
            S = cand
            no_improve = 0
            if cval < best_cost:
                best = cand.copy()
                best_cost = cval
                timeline_iters.append(iter_count)
                timeline_costs.append(best_cost)
                timeline_times.append(time.time() - start_time)
        else:
            no_improve += 1

    return dict(best_sequence=best, best_cost=best_cost,
                timeline=(timeline_iters, timeline_costs, timeline_times),
                iterations=iter_count, time=time.time() - start_time)


# ===================================================
# Heuristics SPT / EDD
# ===================================================
def heuristic_spt(processing_times):
    return sorted(range(len(processing_times)), key=lambda i: processing_times[i])


def heuristic_edd(due_dates):
    return sorted(range(len(due_dates)), key=lambda i: due_dates[i])


# ===================================================
# Gurobi MIP
# ===================================================
def gurobi_total_tardiness(processing_times, due_dates, time_limit=1800):
    try:
        from gurobipy import Model, GRB
    except Exception as e:
        raise ImportError("gurobipy import failed: " + str(e))

    n = len(processing_times)
    Psum = sum(processing_times)
    M = Psum + max(abs(min(due_dates)), 0) + 1000

    m = Model("total_tardiness")
    m.setParam('TimeLimit', time_limit)
    m.setParam('OutputFlag', 0)

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

    for i in range(n):
        m.addConstr(sum(x[i, j] for j in range(n)) == 1)

    for j in range(n):
        m.addConstr(sum(x[i, j] for i in range(n)) == 1)

    for j in range(n):
        if j == 0:
            m.addConstr(Cpos[0] >= sum(processing_times[i] * x[i, 0] for i in range(n)))
        else:
            m.addConstr(Cpos[j] >= Cpos[j - 1] +
                        sum(processing_times[i] * x[i, j] for i in range(n)))

    for i in range(n):
        for j in range(n):
            m.addConstr(Tjob[i] >= Cpos[j] - due_dates[i] - M * (1 - x[i, j]))

    m.setObjective(sum(Tjob[i] for i in range(n)), GRB.MINIMIZE)
    m.optimize()

    status = m.Status
    if status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED):
        assign = [-1] * n
        for i in range(n):
            for j in range(n):
                if x[i, j].X > 0.5:
                    assign[j] = i

        seq = assign.copy()
        obj = sum(Tjob[i].X for i in range(n))
        bestbound = m.ObjBound
        mipgap = m.MIPGap if hasattr(m, 'MIPGap') else None

        return dict(sequence=seq, objective=obj, bestbound=bestbound,
                    mipgap=mipgap, status=status)
    else:
        raise RuntimeError("Gurobi failed with status " + str(status))


# ===================================================
# Runner: run all algs
# ===================================================
def run_all_experiments(n_jobs=100, seed=42, time_limit=1800):
    random.seed(seed)

    processing_times = [random.randint(1, 100) for _ in range(n_jobs)]
    avg_p = statistics.mean(processing_times)
    due_dates = [int(random.gauss(avg_p * n_jobs / 2, avg_p * n_jobs / 6)) for _ in range(n_jobs)]
    due_dates = [max(0, d) for d in due_dates]

    # -------------------------
    # Save full job list
    # -------------------------
    with open("generated_jobs.txt", "w") as f:
        f.write("job_id, processing_time, due_date\n")
        for i in range(n_jobs):
            f.write(f"{i}, {processing_times[i]}, {due_dates[i]}\n")

    print("\nSaved full job list to generated_jobs.txt")

    print("\nSample jobs (first 20 shown):")
    for i in range(min(20, n_jobs)):
        print(f"Job {i}: p={processing_times[i]}, d={due_dates[i]}")
    print("... total jobs:", n_jobs)

    results = {}
    timelines = {}

    # -------------------------
    # SPT
    # -------------------------
    seq_spt = heuristic_spt(processing_times)
    cost_spt = compute_tardiness(seq_spt, processing_times, due_dates)
    results['SPT'] = {'cost': cost_spt, 'seq': seq_spt}
    timelines['SPT'] = ([0], [cost_spt], [0.0])

    # -------------------------
    # EDD
    # -------------------------
    seq_edd = heuristic_edd(due_dates)
    cost_edd = compute_tardiness(seq_edd, processing_times, due_dates)
    results['EDD'] = {'cost': cost_edd, 'seq': seq_edd}
    timelines['EDD'] = ([0], [cost_edd], [0.0])

    # -------------------------
    # Local Search from SPT
    # -------------------------
    print("\nRunning Local Search (from SPT)...")
    ls_res = local_search(processing_times, due_dates, init_seq=seq_spt,
                          max_no_improve=2000, time_limit=300, seed=seed)
    results['LocalSearch'] = {'cost': ls_res['best_cost'], 'seq': ls_res['best_sequence']}
    timelines['LocalSearch'] = ls_res['timeline']

    # -------------------------
    # Simple SA
    # -------------------------
    print("\nRunning Simple SA...")
    sa_res = simple_sa(processing_times, due_dates, T0=500, Tmin=1e-3, K=0.9,
                       Imax=500, seed=seed, time_limit=300)
    results['SimpleSA'] = {'cost': sa_res['best_cost'], 'seq': sa_res['best_sequence']}
    timelines['SimpleSA'] = sa_res['timeline']

    # -------------------------
    # SA-ANS
    # -------------------------
    print("\nRunning SA-ANS...")
    sa_ans_res = sa_ans(processing_times, due_dates, T0=500, Tmin=1e-3,
                        K=0.85, Imax=300, Vmax=10000,
                        H_init=5, H_max=50, seed=seed, time_limit=600)
    results['SA-ANS'] = {'cost': sa_ans_res['best_cost'], 'seq': sa_ans_res['best_sequence']}
    timelines['SA-ANS'] = sa_ans_res['timeline']

    # -------------------------
    # IG
    # -------------------------
    print("\nRunning Iterated Greedy (IG)...")
    ig_res = iterated_greedy(processing_times, due_dates, max_restarts=200,
                             destruct_k=max(5, n_jobs // 20), time_limit=300, seed=seed)
    results['IG'] = {'cost': ig_res['best_cost'], 'seq': ig_res['best_sequence']}
    timelines['IG'] = ig_res['timeline']

    # -------------------------
    # Random-restart LS
    # -------------------------
    print("\nRunning Random-restart Local Search...")
    best_rand_ls_cost = float('inf')
    best_rand_ls_seq = None
    rr_iters = [0]
    rr_costs = [float('inf')]
    rr_times = [0.0]

    rstart_time = time.time()
    iterations = 0

    while time.time() - rstart_time < 200:
        iterations += 1
        init = list(range(n_jobs))
        random.shuffle(init)

        rls = local_search(processing_times, due_dates, init_seq=init,
                           max_no_improve=500, time_limit=5,
                           seed=random.randint(0, 10**6))

        if rls['best_cost'] < best_rand_ls_cost:
            best_rand_ls_cost = rls['best_cost']
            best_rand_ls_seq = rls['best_sequence']
            rr_iters.append(iterations)
            rr_costs.append(best_rand_ls_cost)
            rr_times.append(time.time() - rstart_time)

    results['RandLS'] = {'cost': best_rand_ls_cost, 'seq': best_rand_ls_seq}
    timelines['RandLS'] = (rr_iters, rr_costs, rr_times)

    # -------------------------
    # Gurobi
    # -------------------------
    gurobi_available = ensure_gurobi_installed()

    if gurobi_available:
        try:
            print("\nAttempting Gurobi MIP...")
            gurobi_info = gurobi_total_tardiness(processing_times, due_dates, time_limit=time_limit)
            seq_g = gurobi_info['sequence']
            obj_g = gurobi_info['objective']
            bound = gurobi_info.get('bestbound', None)
            gap = gurobi_info.get('mipgap', None)
            results['Gurobi'] = {'cost': obj_g, 'seq': seq_g, 'bound': bound, 'mipgap': gap}
            timelines['Gurobi'] = ([0], [obj_g], [0.0])
            print("Gurobi result → cost:", obj_g, "mipgap:", gap)
        except Exception as e:
            print("[ERROR] Gurobi failed:", e)
            results['Gurobi'] = None
            timelines['Gurobi'] = None
    else:
        results['Gurobi'] = None
        timelines['Gurobi'] = None

    # -------------------------
    # Summary
    # -------------------------
    print("\n==== FINAL SUMMARY (Total Tardiness) ====")
    for name, r in results.items():
        if r is None:
            print(f"{name}: skipped (no Gurobi or failure)")
        else:
            if name == "Gurobi" and r.get("mipgap") is not None:
                print(f"{name}: cost={r['cost']:.2f}, mipgap={r['mipgap']}")
            else:
                print(f"{name}: cost={r['cost']:.2f}")

    return processing_times, due_dates, results, timelines


# ===================================================
# Plotting
# ===================================================
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


# ===================================================
# Main
# ===================================================
if __name__ == "__main__":
    n_jobs = 100
    seed = 42
    time_limit = 1800

    processing_times, due_dates, results, timelines = run_all_experiments(
        n_jobs=n_jobs, seed=seed, time_limit=time_limit
    )

    plot_all_timelines(timelines, title=f"Comparison of 8 algorithms on {n_jobs} jobs (tardiness)")
