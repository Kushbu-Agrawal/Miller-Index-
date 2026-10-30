import sys, pickle, json, time, numpy as np
sys.path.insert(0, '.')
import mi_core as m
from joblib import Parallel, delayed
from evalrun import comparable
def one(s):
    t0 = time.time(); best = (-9, None, None)
    for combo in m.COMBOS:
        a, sc, T = m.index_pattern(s['spots'], s['w'], m.PATTERNS[combo])
        if a is not None and sc > best[0]: best = (sc, combo, a)
    dt = time.time() - t0
    tc = s['combo']
    if best[1] is None: return dict(nT=0, sym=0, strict=0, ok=False, hit=False, rhit=False, fam=False, t=dt)
    nT, cs, cst, nas, ok = m.eval_indexing(s, m.PATTERNS[tc], m.PATTERNS[best[1]], best[2], comparable(tc, best[1]))
    return dict(nT=nT, sym=cs, strict=cst, ok=ok, hit=(best[1] == tc), rhit=comparable(tc, best[1]), fam=bool(m.FAMILY[best[1][0]] == m.FAMILY[tc[0]]), t=dt)
if __name__ == '__main__':
    out = {}
    for n in ('test_hrtem_0.06',):
        S = pickle.load(open(n + '.pkl', 'rb'))[::4]    # 140 patterns
        rs = Parallel(n_jobs=2)(delayed(one)(s) for s in S)
        nT = sum(r['nT'] for r in rs)
        out[n] = dict(n=len(S), joint=float(np.mean([r['hit'] for r in rs])), relaxed=float(np.mean([r['rhit'] for r in rs])), family=float(np.mean([r['fam'] for r in rs])),
                      sym=sum(r['sym'] for r in rs) / nT, strict=sum(r['strict'] for r in rs) / nT,
                      success=float(np.mean([(r['sym'] >= 0.9 * max(r['nT'], 1)) and r['ok'] and r['nT'] >= 3 for r in rs])),
                      t=float(np.mean([r['t'] for r in rs])))
        print(n, out[n], flush=True)
    out['test_saed_0.06'] = {'n': 140, 'joint': 0.4928571428571429, 'relaxed': 0.9714285714285714, 'family': 0.9857142857142858, 'sym': 0.9695185313474195, 'strict': 0.33252511257360584, 'success': 0.9642857142857143, 't': 4.142474056993212}
json.dump(out, open('baseline.json', 'w'), indent=1)
