# Placeholder: 작성 요청하신 "기존 결과와 동일하게 맞추는 버전"의 코드를 위한 기본 구조입니다.
# 이전 대화에서 제공된 데이터/함수/입력 형식이 누락되어 있어, 정확한 재현을 위해 아래 틀을 먼저 구성합니다.

import random
import math
from typing import List, Tuple

# ----------------------
# 기본 Job 구조
# ----------------------
class Job:
    def __init__(self, jid: int, p: int, d: int):
        self.jid = jid
        self.p = p
        self.d = d

    def __repr__(self):
        return f"Job({self.jid}, p={self.p}, d={self.d})"

# ----------------------
# Tardiness 계산
# ----------------------
def compute_tardiness(seq: List[Job]) -> int:
    t = 0
    time = 0
    for job in seq:
        time += job.p
        t += max(time - job.d, 0)
    return t

# ----------------------
# SPT / EDD
# ----------------------
def spt(jobs: List[Job]) -> List[Job]:
    return sorted(jobs, key=lambda x: x.p)

def edd(jobs: List[Job]) -> List[Job]:
    return sorted(jobs, key=lambda x: x.d)

# ----------------------
# Local Search (Insertion)
# ----------------------
def local_search(seq: List[Job]) -> List[Job]:
    best = seq[:]
    best_t = compute_tardiness(best)

    improved = True
    while improved:
        improved = False
        for i in range(len(best)):
            job = best[i]
            removed = best[:i] + best[i+1:]
            for j in range(len(removed)+1):
                new_seq = removed[:j] + [job] + removed[j:]
                new_t = compute_tardiness(new_seq)
                if new_t < best_t:
                    best = new_seq
                    best_t = new_t
                    improved = True
                    break
            if improved:
                break
    return best

# ----------------------
# IG (기존 결과와 동일한 랜덤 d개 제거 버전)
# ----------------------
def ig(seq: List[Job], d: int = 3, iter_n: int = 100) -> List[Job]:
    best = seq[:]
    best_t = compute_tardiness(best)

    for _ in range(iter_n):
        # --- 랜덤하게 d개 제거 (기존 코드 동일 방식) ---
        removed_idx = sorted(random.sample(range(len(best)), d))
        removed = [best[i] for i in removed_idx]
        base = [job for i, job in enumerate(best) if i not in removed_idx]

        # --- greedy insertion ---
        for job in removed:
            best_pos = None
            best_cost = math.inf
            for i in range(len(base)+1):
                trial = base[:i] + [job] + base[i:]
                c = compute_tardiness(trial)
                if c < best_cost:
                    best_cost = c
                    best_pos = i
            base.insert(best_pos, job)

        # --- local search for stability ---
        improved = local_search(base)
        new_t = compute_tardiness(improved)

        if new_t < best_t:
            best = improved
            best_t = new_t

    return best

# ----------------------
# (추후 추가) Gurobi 최적화 버전 자리
# ----------------------
# 주의: Gurobi 모델은 실제 License 환경에서만 작동합니다.
# 요청 시 여기 아래에 full working version 삽입 가능.


def solve_all(jobs: List[Job]):
    result = {}

    seq_spt = spt(jobs)
    result['SPT'] = compute_tardiness(seq_spt)

    seq_edd = edd(jobs)
    result['EDD'] = compute_tardiness(seq_edd)

    ls_seq = local_search(jobs)
    result['LocalSearch'] = compute_tardiness(ls_seq)

    ig_seq = ig(jobs, d=3, iter_n=100)
    result['IG'] = compute_tardiness(ig_seq)

    # grb_result = solve_gurobi(...)  # 자리만 구성 (필요 시 제공)
    # result['Gurobi'] = grb_result

    return result

# ----------------------
# 테스트 (예시)
# ----------------------
if __name__ == "__main__":
    sample = [Job(i, p=random.randint(1,20), d=random.randint(10,100)) for i in range(10)]
    print(solve_all(sample))