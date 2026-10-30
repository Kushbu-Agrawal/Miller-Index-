import sys, pickle, numpy as np
sys.path.insert(0, '.')
import mi_core as m
from joblib import Parallel, delayed

def job(combo, seed, mode, noise, cal):
    rng = np.random.default_rng(seed)
    return m.make_sample(combo, rng, mode, noise, cal)

def build(name, per_combo, seed0, spec):
    tasks = []
    sd = seed0
    for combo in m.COMBOS:
        for i in range(per_combo):
            mode, noise, cal = spec(i)
            tasks.append((combo, sd, mode, noise, cal)); sd += 1
    out = Parallel(n_jobs=2, batch_size=16)(delayed(job)(*t) for t in tasks)
    pickle.dump(out, open(f'{name}.pkl', 'wb'))
    print(name, len(out), flush=True)

if __name__ == '__main__':
    rs = np.random.default_rng(0)
    def train_spec(i):
        r = np.random.default_rng(1000 + i)
        return ('saed' if r.random() < 0.65 else 'hrtem', float(r.uniform(0.02, 0.10)), 0.03)
    build('train', 100, 100000, train_spec)
    build('val', 10, 200000, train_spec)
    for mode in ('saed', 'hrtem'):
        for nz in (0.03, 0.06, 0.10, 0.15):
            build(f'test_{mode}_{nz}', 8, 300000 + int(nz * 1000) + (0 if mode == 'saed' else 50000), lambda i, mode=mode, nz=nz: (mode, nz, 0.03))
    build('test_saed_cal6', 8, 400000, lambda i: ('saed', 0.06, 0.06))
