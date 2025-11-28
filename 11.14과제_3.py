import math
import random
import time
import matplotlib.pyplot as plt

# ================================================================
# 0. 고정된 100개 작업 데이터 사용
# ================================================================
jobs_data = [
    (1,45,532),(2,12,986),(3,6,994),(4,22,257),(5,20,434),(6,19,232),(7,13,523),(8,11,610),(9,48,474),(10,39,267),
    (11,10,416),(12,42,780),(13,32,935),(14,7,522),(15,6,417),(16,10,871),(17,18,711),(18,19,605),(19,37,858),(20,43,669),
    (21,6,346),(22,40,471),(23,17,342),(24,50,452),(25,46,962),(26,49,774),(27,39,751),(28,31,469),(29,19,964),(30,33,798),
    (31,42,638),(32,22,797),(33,5,608),(34,15,570),(35,49,424),(36,32,341),(37,26,721),(38,22,705),(39,14,293),(40,18,973),
    (41,26,248),(42,11,312),(43,10,356),(44,29,842),(45,11,363),(46,27,896),(47,27,632),(48,43,810),(49,21,265),(50,7,594),
    (51,34,590),(52,39,810),(53,12,679),(54,29,741),(55,10,457),(56,40,766),(57,23,211),(58,45,896),(59,44,938),(60,28,317),
    (61,41,898),(62,17,749),(63,50,968),(64,9,473),(65,7,987),(66,47,856),(67,19,548),(68,23,314),(69,10,500),(70,19,645),
    (71,11,361),(72,29,664),(73,22,203),(74,34,939),(75,45,936),(76,28,469),(77,15,712),(78,28,980),(79,27,382),(80,18,719),
    (81,47,308),(82,22,840),(83,49,505),(84,48,854),(85,46,719),(86,9,823),(87,43,403),(88,45,356),(89,15,582),(90,39,980),
    (91,20,365),(92,15,752),(93,34,997),(94,29,743),(95,22,200),(96,45,813),(97,49,531),(98,40,700),(99,19,219),(100,48,314),
]

# (p, d)만 사용
jobs = [(p, d) for (_, p, d) in jobs_data]
N = len(jobs)

print("Loaded fixed 100 jobs.")

# ================================================================
# 1. 공용 함수
# ================================================================
def compute_tardiness(sequence, jobs):
    t = 0
    T = 0
    for idx in sequence:
        p, d = jobs[idx]
        t += p
        T += max(0, t - d)
    return T


def random_permutation():
    return list(range(N))


# ================================================================
# 2. Local Search (swap / insert / block)
# ================================================================
def ls_swap(seq, jobs):
    best = seq[:]
    best_cost = compute_tardiness(best, jobs)

    improved = True
    while improved:
        improved = False
        for i in range(N - 1):
            for j in range(i + 1, N):
                seq2 = best[:]
                seq2[i], seq2[j] = seq2[j], seq2[i]
                cost = compute_tardiness(seq2, jobs)
                if cost < best_cost:
                    print(f"[LS-swap] improved {best_cost} → {cost}")
                    best = seq2
                    best_cost = cost
                    improved = True
    return best, best_cost


def ls_insert(seq, jobs):
    best = seq[:]
    best_cost = compute_tardiness(best, jobs)

    improved = True
    while improved:
        improved = False
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                seq2 = best[:]
                job = seq2.pop(i)
                seq2.insert(j, job)
                cost = compute_tardiness(seq2, jobs)
                if cost < best_cost:
                    print(f"[LS-insert] improved {best_cost} → {cost}")
                    best = seq2
                    best_cost = cost
                    improved = True
    return best, best_cost


def ls_block(seq, jobs):
    best = seq[:]
    best_cost = compute_tardiness(best, jobs)

    improved = True
    while improved:
        improved = False
        for i in range(N - 2):
            seq2 = best[:]
            seq2[i], seq2[i+1], seq2[i+2] = seq2[i+2], seq2[i], seq2[i+1]
            cost = compute_tardiness(seq2, jobs)
            if cost < best_cost:
                print(f"[LS-block] improved {best_cost} → {cost}")
                best = seq2
                best_cost = cost
                improved = True
    return best, best_cost


# ================================================================
# 3. SA - ANS (Near Neighbor Search + Adaptive H)
# ================================================================
def sa_ans(jobs, max_iter=300, T0=100.0, alpha=0.98):
    print("\n[SA] Starting Simulated Annealing with Adaptive Neighbor Size (ANS)")

    curr = random_permutation()
    curr_cost = compute_tardiness(curr, jobs)

    best = curr[:]
    best_cost = curr_cost

    T = T0
    H = 1   # 초기 후보 개수

    history = []

    for it in range(max_iter):
        for inner in range(50):
            # H개 랜덤 move 생성
            candidate = curr[:]
            for _ in range(H):
                i, j = random.sample(range(N), 2)
                candidate[i], candidate[j] = candidate[j], candidate[i]

            cand_cost = compute_tardiness(candidate, jobs)

            # Metropolis acceptance
            if cand_cost < curr_cost or random.random() < math.exp((curr_cost - cand_cost) / T):
                curr, curr_cost = candidate, cand_cost

            # Best update
            if curr_cost < best_cost:
                print(f"[SA] Improved {best_cost} → {curr_cost}")
                best, best_cost = curr[:], curr_cost
                H += 1   # 후보 수 증가
                print(f"[SA] Increase H = {H}")

        history.append(best_cost)
        T *= alpha
        print(f"[SA] iter {it} T={T:.3f} best={best_cost}")

    return best, best_cost, history


# ================================================================
# 4. IG (Iterated Greedy)
# ================================================================
def ig(jobs, max_iter=200, destroy_rate=0.2):
    print("\n[IG] Starting Iterated Greedy")

    seq = random_permutation()
    seq_cost = compute_tardiness(seq, jobs)

    best = seq[:]
    best_cost = seq_cost

    history = []

    D = int(N * destroy_rate)

    for it in range(max_iter):
        removed = random.sample(seq, D)
        remain = [x for x in seq if x not in removed]
        random.shuffle(removed)
        seq = remain + removed

        seq, seq_cost = ls_swap(seq, jobs)

        if seq_cost < best_cost:
            print(f"[IG] Improved {best_cost} → {seq_cost}")
            best, best_cost = seq[:], seq_cost

        history.append(best_cost)
        print(f"[IG] iter {it} best={best_cost}")

    return best, best_cost, history


# ================================================================
# 5. Simple Heuristics (SPT, EDD, RANDOM)
# ================================================================
def heuristic_spt(jobs):
    seq = sorted(range(N), key=lambda x: jobs[x][0])
    return compute_tardiness(seq, jobs), seq


def heuristic_edd(jobs):
    seq = sorted(range(N), key=lambda x: jobs[x][1])
    return compute_tardiness(seq, jobs), seq


def heuristic_random(jobs):
    seq = random_permutation()
    return compute_tardiness(seq, jobs), seq


# ================================================================
# 6. Gurobi Solver (optional)
# ================================================================
def run_gurobi(jobs):
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except:
        print("\n[Gurobi] ERROR: gurobipy not installed.")
        print("Install with: pip install gurobipy")
        return None, None, None

    print("\n[Gurobi] Solving with MIP (may take long)...")

    model = gp.Model()
    model.setParam("TimeLimit", 1800)

    x = model.addVars(N, N, vtype=GRB.BINARY)
    model.addConstrs(x.sum(i, '*') == 1 for i in range(N))
    model.addConstrs(x.sum('*', j) == 1 for j in range(N))

    order = list(range(N))
    cost = sum(jobs[order[j]][0] * x[i, j] for i in range(N) for j in range(N))

    model.setObjective(cost, GRB.MINIMIZE)

    model.optimize()

    if model.Status in [2, 9]:
        seq = []
        for i in range(N):
            for j in range(N):
                if x[i, j].X > 0.5:
                    seq.append(j)
        tard = compute_tardiness(seq, jobs)
        gap = model.MIPGap
        print(f"[Gurobi] Finished. Gap={gap}")
        return tard, seq, gap

    print("[Gurobi] No solution found.")
    return None, None, None


# ================================================================
# 7. MAIN EXECUTION
# ================================================================
def main():
    # SA
    sa_seq, sa_cost, sa_hist = sa_ans(jobs)

    # IG
    ig_seq, ig_cost, ig_hist = ig(jobs)

    # LS
    ls_s_seq, ls_s_cost = ls_swap(random_permutation(), jobs)
    ls_i_seq, ls_i_cost = ls_insert(random_permutation(), jobs)
    ls_b_seq, ls_b_cost = ls_block(random_permutation(), jobs)

    # Heuristics
    spt_cost, spt_seq = heuristic_spt(jobs)
    edd_cost, edd_seq = heuristic_edd(jobs)
    rnd_cost, rnd_seq = heuristic_random(jobs)

    # Gurobi
    gurobi_cost, gurobi_seq, gap = run_gurobi(jobs)

    print("\n==================== FINAL RESULTS ====================")
    print(f"SA-ANS : {sa_cost}")
    print(f"IG     : {ig_cost}")
    print(f"LS-swap: {ls_s_cost}")
    print(f"LS-insert: {ls_i_cost}")
    print(f"LS-block : {ls_b_cost}")
    print(f"SPT    : {spt_cost}")
    print(f"EDD    : {edd_cost}")
    print(f"RANDOM : {rnd_cost}")
    print(f"Gurobi : {gurobi_cost}   (gap={gap})")

    # ---------------- Plot ----------------
    plt.figure(figsize=(12, 5))
    plt.plot(sa_hist, label="SA-ANS")
    plt.plot(ig_hist, label="IG")
    plt.legend()
    plt.title("Algorithm Performance History")
    plt.xlabel("Iteration")
    plt.ylabel("Best Tardiness")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
