import sys, pickle, json, time, numpy as np, torch
sys.path.insert(0, '.')
import mi_core as m
from model import Net
from joblib import Parallel, delayed
torch.set_num_threads(1)
NV = [len(z) for z in m.ZA_LIST]
def load_model(tag):
    ck = torch.load(tag + '.pt', weights_only=False)
    net = Net(11, 13, bool(ck['use_img']), bool(ck['use_phys'])); net.load_state_dict(ck['state']); net.eval()
    return net, ck['mu'], ck['sd']
def predict(net, mu, sd, samples, topk=3):
    X = torch.from_numpy(np.stack([s['img'] for s in samples]).astype(np.float32))[:, None]
    P = torch.from_numpy(np.stack([s['phys'] for s in samples]))
    with torch.no_grad():
        lp, lz = net(X, (P[:, :mu.shape[0]] - mu) / sd)
    pp = torch.softmax(lp, 1).numpy(); pz = torch.softmax(lz, 1).numpy()
    res = []
    for i in range(len(samples)):
        cands = []
        for p in range(11):
            z = pz[i, :NV[p]]; z = z / z.sum()
            for j in range(NV[p]): cands.append((pp[i, p] * z[j], p, j))
        cands.sort(reverse=True)
        res.append([(p, j, float(sc)) for sc, p, j in cands[:topk]])
    return res
def comparable(tc, pc):
    (tp, tz), (pp, pz) = tc, pc
    if tz != pz and not (m.FAMILY[tp] == 0 and m.FAMILY[pp] == 0 and tz == pz): return False
    return tp == pp or (m.FAMILY[tp] == 0 and m.FAMILY[pp] == 0)
def index_eval(s, cand_list, topk):
    tc = s['combo']; t0 = time.time()
    best = (-9, None, None, None)
    for (p, j, _) in cand_list[:topk]:
        pat = m.PATTERNS[(p, j)]
        a, sc, T = m.index_pattern(s['spots'], s['w'], pat)
        if a is not None and sc > best[0]: best = (sc, (p, j), a, pat)
    dt = time.time() - t0
    if best[1] is None: return dict(nT=0, sym=0, strict=0, nas=0, ok=False, hit=False, rhit=False, t=dt)
    nT, cs, cst, nas, ok = m.eval_indexing(s, m.PATTERNS[tc], best[3], best[2], comparable(tc, best[1]))
    return dict(nT=nT, sym=cs, strict=cst, nas=nas, ok=ok, hit=(best[1] == tc), rhit=comparable(tc, best[1]), t=dt)
def oracle(s):
    tc = s['combo']; pat = m.PATTERNS[tc]
    a, sc, T = m.index_pattern(s['spots'], s['w'], pat)
    nT, cs, cst, nas, ok = m.eval_indexing(s, pat, pat, a, True)
    return dict(nT=nT, sym=cs, strict=cst, nas=nas, ok=ok, hit=True, rhit=True)
def chunk_eval(samples, preds):
    return [dict(top1=index_eval(s, p, 1), top3=index_eval(s, p, 3), orc=oracle(s)) for s, p in zip(samples, preds)]
def agg(rs, key):
    nT = sum(r[key]['nT'] for r in rs)
    return dict(sym=sum(r[key]['sym'] for r in rs) / max(nT, 1), strict=sum(r[key]['strict'] for r in rs) / max(nT, 1),
                assigned=sum(r[key]['nas'] for r in rs) / max(nT, 1),
                success=float(np.mean([(r[key]['sym'] >= 0.9 * max(r[key]['nT'], 1)) and r[key]['ok'] and r[key]['nT'] >= 3 for r in rs])),
                hit=float(np.mean([r[key]['hit'] for r in rs])), rhit=float(np.mean([r[key]['rhit'] for r in rs])))
if __name__ == '__main__':
    models = {t: load_model(t) for t in ('fused', 'imgonly', 'physonly')}
    names = ['test_saed_0.03', 'test_saed_0.06', 'test_saed_0.1', 'test_saed_0.15',
             'test_hrtem_0.03', 'test_hrtem_0.06', 'test_hrtem_0.1', 'test_hrtem_0.15', 'test_saed_cal6']
    R = {}
    for n in names:
        S = m.upgrade(pickle.load(open(n + '.pkl', 'rb')))
        y = np.array([s['combo'] for s in S])
        R[n] = {'n': len(S)}
        # detection stats
        det = tot = 0
        for s in S:
            D = np.linalg.norm((s['spots'] * m.S_NOM + s['centre'])[:, None, :] - s['true_px'][None, :, :][:, s['vis']], axis=2) if len(s['spots']) and s['vis'].any() else np.zeros((0, 0))
            det += int((D.min(1) < 3.5).sum()) if D.size else 0; tot += int(s['vis'].sum())
        R[n]['det_recall'] = det / tot
        R[n]['det_precision'] = det / max(sum(len(s['spots']) for s in S), 1)
        for t, (net, mu, sd) in models.items():
            pr = predict(net, mu, sd, S)
            p1 = [p[0] for p in pr]
            ph = np.mean([a[0] == b[0] for a, b in zip(p1, y)])
            fam = np.mean([m.FAMILY[a[0]] == m.FAMILY[b[0]] for a, b in zip(p1, y)])
            joint = np.mean([(a[0], a[1]) == tuple(b) for a, b in zip(p1, y)])
            rel = np.mean([comparable(tuple(b), (a[0], a[1])) for a, b in zip(p1, y)])
            top3 = np.mean([tuple(b) in [(c[0], c[1]) for c in p] for p, b in zip(pr, y)])
            R[n][t] = dict(phase=float(ph), family=float(fam), joint=float(joint), relaxed=float(rel), top3=float(top3))
            if t == 'fused':
                preds = pr
        half = len(S) // 2 + 1
        parts = Parallel(n_jobs=2)(delayed(chunk_eval)(S[i::2], preds[i::2]) for i in range(2))
        rs = parts[0] + parts[1]
        R[n]['idx'] = {k: agg(rs, k) for k in ('top1', 'top3', 'orc')}
        R[n]['t_top1'] = float(np.mean([r['top1']['t'] for r in rs])); R[n]['t_top3'] = float(np.mean([r['top3']['t'] for r in rs]))
        print(n, json.dumps(R[n]), flush=True)
        json.dump(R, open('results.json', 'w'), indent=1)
