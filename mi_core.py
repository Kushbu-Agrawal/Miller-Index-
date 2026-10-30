"""Miller index assignment pipeline: crystal models, SAED / HRTEM-FFT simulation,
spot detection, physics features and geometric indexing."""
import itertools, math
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

IMG = 256
HALF = 128.0
S_NOM = 160.0      # nominal calibration, pixels per 1/Angstrom
GMAX = 0.75        # largest |g| simulated, 1/Angstrom
PIX_HR = 1.0 / (1024 * (1.0 / S_NOM))   # real-space pixel (A) so that FFT crop gives S_NOM

# ---------------------------------------------------------------- reflection rules
def rule_pbnm(h, k, l):
    ok = np.ones(h.shape, bool)
    ok &= ~((h == 0) & (k % 2 != 0))
    ok &= ~((k == 0) & ((h + l) % 2 != 0))
    return ok

def rule_hematite(h, k, l):
    ok = ((-h + k + l) % 3 == 0)
    ok &= ~((k == -h) & (l % 2 != 0))
    ok &= ~((h == 0) & (k == 0) & (l % 6 != 0))
    return ok

def rule_magnetite(h, k, l):
    par = (h % 2 == k % 2) & (k % 2 == l % 2)
    ok = par.copy()
    idx = np.stack([h, k, l], 1)
    for a, b, c in ((0, 1, 2), (1, 0, 2), (2, 0, 1)):
        z = idx[:, a] == 0
        ok &= ~(z & (((idx[:, b] + idx[:, c]) % 4) != 0))
    for a, b, c in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
        e = idx[:, a] == idx[:, b]
        ok &= ~(e & (((idx[:, a] + idx[:, c]) % 2) != 0))
    return ok

def rule_bixbyite(h, k, l):
    ok = ((h + k + l) % 2 == 0)
    idx = np.stack([h, k, l], 1)
    for a, b, c in ((0, 1, 2), (1, 0, 2), (2, 0, 1)):
        z = idx[:, a] == 0
        ok &= ~(z & ((idx[:, b] % 2 != 0) | (idx[:, c] % 2 != 0)))
    for a, b, c in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
        e = idx[:, a] == idx[:, b]
        ok &= ~(e & (idx[:, c] % 2 != 0))
    return ok

# typical literature cell values (Pbnm setting, Angstrom); rounded, to be replaced by ICSD/COD entries
OF = {
 'LaFeO3': (5.556, 5.565, 7.862), 'PrFeO3': (5.495, 5.578, 7.786),
 'NdFeO3': (5.450, 5.588, 7.761), 'SmFeO3': (5.400, 5.597, 7.711),
 'GdFeO3': (5.349, 5.611, 7.669), 'DyFeO3': (5.302, 5.598, 7.623),
 'ErFeO3': (5.263, 5.582, 7.582), 'YFeO3': (5.280, 5.592, 7.603),
}
PHASES = []
for n, (a, b, c) in OF.items():
    PHASES.append(dict(name=n, family='orthoferrite', cell=(a, b, c, 90, 90, 90), rule=rule_pbnm))
PHASES.append(dict(name='a-Fe2O3', family='secondary', cell=(5.035, 5.035, 13.747, 90, 90, 120), rule=rule_hematite))
PHASES.append(dict(name='Fe3O4', family='secondary', cell=(8.396, 8.396, 8.396, 90, 90, 90), rule=rule_magnetite))
PHASES.append(dict(name='Y2O3', family='secondary', cell=(10.604, 10.604, 10.604, 90, 90, 90), rule=rule_bixbyite))
PHASE_NAMES = [p['name'] for p in PHASES]
N_PHASE = len(PHASES)
FAMILY = np.array([0 if p['family'] == 'orthoferrite' else 1 for p in PHASES])

def cell_vectors(a, b, c, al, be, ga):
    al, be, ga = map(math.radians, (al, be, ga))
    av = np.array([a, 0, 0.])
    bv = np.array([b * math.cos(ga), b * math.sin(ga), 0.])
    cx = c * math.cos(be)
    cy = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
    cz = math.sqrt(max(c * c - cx * cx - cy * cy, 1e-12))
    return np.array([av, bv, np.array([cx, cy, cz])])

def _za_candidates():
    out = []
    for t in itertools.product((-1, 0, 1), repeat=3):
        if t == (0, 0, 0): continue
        # keep one of +-t
        for x in t:
            if x != 0:
                if x < 0: break
                out.append(t); break
    return out
ZA_ALL = _za_candidates()     # 13 low-index directions

def build_phase_patterns():
    """returns PATTERNS[(phase, za)] = dict(hkl, xy, g, zone) and ZA_LIST[phase]"""
    patterns, za_list = {}, []
    for pi, ph in enumerate(PHASES):
        A = cell_vectors(*ph['cell'])
        B = np.linalg.inv(A).T            # rows: a*, b*, c*
        R = int(math.ceil(GMAX * max(ph['cell'][:3]))) + 1
        rng_ = np.arange(-R, R + 1)
        H = np.array(list(itertools.product(rng_, rng_, rng_)))
        H = H[np.any(H != 0, 1)]
        H = H[ph['rule'](H[:, 0], H[:, 1], H[:, 2])]
        Gc = H @ B
        gl = np.linalg.norm(Gc, axis=1)
        m = gl <= GMAX
        H, Gc, gl = H[m], Gc[m], gl[m]
        seen, reps = {}, []
        for zi, z in enumerate(ZA_ALL):
            inz = (H @ np.array(z)) == 0
            if inz.sum() < 8: continue
            g = Gc[inz]; hk = H[inz]; gn = gl[inz]
            zc = np.array(z) @ A
            n = zc / np.linalg.norm(zc)
            e1 = g[np.argmin(gn)]; e1 = e1 / np.linalg.norm(e1)
            e2 = np.cross(n, e1)
            xy = np.stack([g @ e1, g @ e2], 1)
            fp = tuple(np.round(np.sort(gn[gn <= 0.6]), 2))
            if fp in seen: continue
            seen[fp] = len(reps)
            order = np.argsort(gn)
            patterns[(pi, len(reps))] = dict(hkl=hk[order], xy=xy[order], g=gn[order], zone=z)
            reps.append(z)
        za_list.append(reps)
    return patterns, za_list
PATTERNS, ZA_LIST = build_phase_patterns()
N_ZA = 13
COMBOS = sorted(PATTERNS.keys())

# ---------------------------------------------------------------- rendering
def _rot(th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]])

def _transform(pat, rng):
    th = rng.uniform(0, 2 * math.pi); flip = rng.random() < 0.5
    xy = pat['xy'].copy()
    if flip: xy[:, 1] *= -1
    return xy @ _rot(th).T, th, flip

def _splat(img, px, amp):
    x, y = px[:, 0], px[:, 1]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                      (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = x0 + dx, y0 + dy
        ok = (xi >= 0) & (xi < IMG) & (yi >= 0) & (yi < IMG)
        np.add.at(img, (yi[ok], xi[ok]), (amp * w)[ok])

def render_saed(pat, rng, noise=0.04, cal=0.03):
    xy, th, flip = _transform(pat, rng)
    s = S_NOM * (1 + rng.uniform(-cal, cal))
    c = HALF + rng.uniform(-2, 2, 2)
    px = xy * s + c
    amp = np.exp(-(pat['g'] / 0.55) ** 2) * rng.uniform(0.4, 1.0, len(px))
    keep = rng.random(len(px)) > 0.08
    img = np.zeros((IMG, IMG), np.float32)
    _splat(img, px[keep], amp[keep])
    sp = rng.uniform(20, 110, (rng.integers(2, 5), 2)) * 1.0 + np.array([10, 10])
    _splat(img, sp, rng.uniform(0.15, 0.4, len(sp)))
    img = gaussian_filter(img, 2.0 + rng.uniform(-0.3, 0.3))
    img = img / (img.max() + 1e-9)
    yy, xx = np.mgrid[:IMG, :IMG]
    r = np.hypot(xx - c[0], yy - c[1])
    img += 3.0 * np.exp(-(r / 4.0) ** 2) + 0.08 * np.exp(-r / 50.0)
    img = img / img.max()
    img += rng.normal(0, noise, img.shape).astype(np.float32)
    img = np.clip(img, 0, None); img /= img.max()
    vis = keep & (px[:, 0] >= 0) & (px[:, 0] < IMG) & (px[:, 1] >= 0) & (px[:, 1] < IMG)
    return img.astype(np.float32), dict(px=px, vis=vis, centre=c, mode='saed')

def render_hrtem_fft(pat, rng, noise=0.04, cal=0.03):
    xy, th, flip = _transform(pat, rng)
    err = rng.uniform(-cal, cal)
    p = PIX_HR * (1 + err)                # true pixel size
    g = xy
    half = (g[:, 0] > 1e-9) | ((np.abs(g[:, 0]) <= 1e-9) & (g[:, 1] > 0))
    gh = g[half]; gg = pat['g'][half]
    amp = 0.2 * np.exp(-(gg / 0.45) ** 2) * rng.uniform(0.4, 1.0, len(gh))
    phi = rng.uniform(0, 2 * math.pi, len(gh))
    x = (np.arange(IMG) - HALF) * p
    X, Y = np.meshgrid(x, x)
    im = np.zeros((IMG, IMG), np.float64)
    for (gx, gy), a, ph in zip(gh, amp, phi):
        im += a * np.cos(2 * math.pi * (gx * X + gy * Y) + ph)
    im += rng.normal(0, noise * 12.0, im.shape)
    win = np.outer(np.hanning(IMG), np.hanning(IMG))
    im = (im - im.mean()) * win
    F = np.abs(np.fft.fftshift(np.fft.fft2(im, s=(1024, 1024))))
    crop = F[512 - 128:512 + 128, 512 - 128:512 + 128]
    out = np.sqrt(crop / crop.max())
    c = np.array([HALF, HALF])
    px = xy * (S_NOM / (1 + err)) + c
    vis = (np.hypot(*(px - c).T) < 120)
    return out.astype(np.float32), dict(px=px, vis=vis, centre=c, mode='hrtem', real=im)

# ---------------------------------------------------------------- detection + features
def detect_spots(img, centre_search=True, nmax=50):
    sm = gaussian_filter(img, 1.2)
    d = sm - gaussian_filter(img, 10)
    med = np.median(d); sig = 1.4826 * np.median(np.abs(d - med)) + 1e-9
    mx = maximum_filter(d, size=7)
    yy, xx = np.mgrid[:IMG, :IMG]
    c = np.array([HALF, HALF])
    if centre_search:
        m = np.hypot(xx - HALF, yy - HALF) < 12
        w = np.where(m, sm, 0)
        w = np.where(w > 0.5 * w.max(), w, 0)
        c = np.array([(w * xx).sum() / w.sum(), (w * yy).sum() / w.sum()])
    r = np.hypot(xx - c[0], yy - c[1])
    thr = max(6 * sig, 0.03 * d.max())
    pk = (d == mx) & (d > thr) & (r > 11) & (r < 125)
    ys, xs = np.nonzero(pk)
    order = np.argsort(d[ys, xs])[::-1][:nmax]
    ys, xs = ys[order], xs[order]
    out = []
    for y, x in zip(ys, xs):
        if y < 3 or x < 3 or y > IMG - 4 or x > IMG - 4: continue
        w = np.clip(d[y - 2:y + 3, x - 2:x + 3], 0, None)
        if w.sum() <= 0: continue
        gy, gx = np.mgrid[y - 2:y + 3, x - 2:x + 3]
        out.append(((w * gx).sum() / w.sum(), (w * gy).sum() / w.sum(), d[y, x]))
    if not out:
        return np.zeros((0, 2)), np.zeros(0), c
    o = np.array(out)
    return (o[:, :2] - c) / S_NOM, o[:, 2] / o[:, 2].max(), c

NF = 12
def phys_features(xy, w):
    f = np.zeros(2 * NF + 1, np.float32)
    if len(xy) == 0: return f
    r = np.hypot(xy[:, 0], xy[:, 1])
    o = np.argsort(r)[:NF]
    f[:len(o)] = r[o]
    v0 = xy[o[0]]
    for i, j in enumerate(o):
        cs = np.dot(v0, xy[j]) / (np.linalg.norm(v0) * np.linalg.norm(xy[j]) + 1e-9)
        f[NF + i] = math.degrees(math.acos(np.clip(cs, -1, 1))) / 180.0
    f[-1] = len(xy) / 50.0
    return f

def phys_features2(xy, w):
    """v2: v1 (25 values) plus 11 calibration-free radius ratios r_i / r_0."""
    f = np.zeros(2 * NF + 1 + (NF - 1), np.float32)
    f[:2 * NF + 1] = phys_features(xy, w)
    if len(xy):
        r = np.hypot(xy[:, 0], xy[:, 1]); o = np.argsort(r)[:NF]
        f[2 * NF + 1:2 * NF + 1 + len(o) - 1] = r[o][1:] / (r[o][0] + 1e-9)
    return f

def upgrade(samples):
    for s in samples: s['phys'] = phys_features2(s['spots'], s['w'])
    return samples

def cnn_image(img, size=96):
    t = img.reshape(IMG, IMG)
    # area downsample 256 -> 96 via block mean on resampled grid
    import torch, torch.nn.functional as F
    x = torch.from_numpy(t)[None, None]
    return F.interpolate(x, size=(size, size), mode='area')[0, 0].numpy()

def make_sample(combo, rng, mode, noise, cal=0.03):
    pat = PATTERNS[combo]
    img, meta = (render_saed if mode == 'saed' else render_hrtem_fft)(pat, rng, noise, cal)
    xy, w, c = detect_spots(img, centre_search=(mode == 'saed'))
    return dict(combo=combo, mode=mode, noise=noise, img=cnn_image(img).astype(np.float16),
                spots=xy.astype(np.float32), w=w.astype(np.float32), centre=c,
                true_px=meta['px'].astype(np.float32), vis=meta['vis'],
                phys=phys_features2(xy, w))

# ---------------------------------------------------------------- indexing
def _search(M, wM, Qxy, a, flips, scales, thetas, tol):
    best = (-1, None)
    th = np.radians(thetas)
    Rm = np.stack([np.stack([np.cos(th), -np.sin(th)], -1), np.stack([np.sin(th), np.cos(th)], -1)], -2)  # (T,2,2)
    for fl in flips:
        Q = Qxy * np.array([1.0, fl])
        for s in scales:
            Qr = np.einsum('tij,nj->tni', Rm, Q * s)                 # (T,n,2)
            D = np.linalg.norm(Qr[:, :, None, :] - M[None, None, :, :], axis=3)  # (T,n,m)
            dj = D.min(1); di = D.min(2)
            mj = np.clip(1 - dj / tol, 0, None); mi = np.clip(1 - di / tol, 0, None)
            F = (mj * wM).sum(1) / wM.sum()
            G = (mi * a).sum(1) / a.sum()
            sc = F * G
            t = int(np.argmax(sc))
            if sc[t] > best[0]:
                best = (float(sc[t]), (fl, s, float(thetas[t])))
    return best

def index_pattern(xy, w, pat, tol=0.022):
    """assign (hkl) to detected spots for a given phase/zone-axis pattern."""
    if len(xy) < 3: return None, -1.0, None
    o = np.argsort(-w)[:30]
    M, wM = xy[o], w[o]
    rmax = np.hypot(M[:, 0], M[:, 1]).max()
    sel = np.nonzero(pat['g'] <= rmax * 1.05)[0][:80]
    if len(sel) < 3: return None, -1.0, None
    Q = pat['xy'][sel]; a = np.exp(-(pat['g'][sel] / 0.55) ** 2)
    sc, (fl, s, t) = _search(M, wM, Q, a, (1.0, -1.0), np.linspace(0.95, 1.05, 5), np.arange(0, 360, 3.0), tol)
    sc, (fl, s, t) = _search(M, wM, Q, a, (fl,), np.linspace(s - 0.0125, s + 0.0125, 5), np.arange(t - 3, t + 3.01, 1.0), tol)
    # final assignment for all detected spots
    Qf = pat['xy'] * np.array([1.0, fl]) * s
    Qr = Qf @ _rot(math.radians(t)).T
    D = np.linalg.norm(xy[:, None, :] - Qr[None, :, :], axis=2)
    j = D.argmin(1); dm = D.min(1)
    assign = np.where(dm < tol * 1.2, j, -1)
    return assign, sc, (fl, s, t)

def eval_indexing(sample, pat_true, pat_pred, assign, comparable=True, tol_q=0.02):
    """spot-level evaluation of an indexing result.
    returns (n_true_detected, n_correct_sym, n_correct_strict, n_assigned, pattern_ok)
    sym   : label correct up to a symmetry operation S of the 2-D pattern (S fitted over all spots)
    strict: label equals the generating hkl exactly
    comparable: predicted (phase, zone axis) share an hkl space with the truth (same phase/zone axis, or two orthoferrites with the same zone axis)"""
    xy, c = sample['spots'], sample['centre']
    det_px = xy * S_NOM + c
    tp, vis = sample['true_px'], sample['vis']
    cand = np.nonzero(vis)[0]
    if len(cand) == 0 or len(xy) == 0 or assign is None: return 0, 0, 0, 0, False
    D = np.linalg.norm(det_px[:, None, :] - tp[None, cand, :], axis=2)
    k = D.argmin(1); dk = D.min(1)
    ids = [i for i in np.nonzero(dk < 3.5)[0]]
    nT = len(ids)
    pairs = [(cand[k[i]], assign[i]) for i in ids if assign[i] >= 0]
    nas = len(pairs)
    if not comparable or nas < 3: return nT, 0, 0, nas, False
    ht, hp = pat_true['hkl'], pat_pred['hkl']
    cstr = sum(1 for t, a in pairs if np.array_equal(ht[t], hp[a]))
    P = pat_true['xy'][[t for t, a in pairs]]; Qa = pat_pred['xy'][[a for t, a in pairs]]
    U, _, Vt = np.linalg.svd(P.T @ Qa)
    S = (U @ Vt).T
    resid = np.linalg.norm(P @ S.T - Qa, axis=1)
    good = resid < tol_q
    rr = np.hypot(*Qa.T).max()
    allq = pat_pred['xy'][pat_pred['g'] <= rr]
    mapped = allq @ S.T
    dd = np.linalg.norm(mapped[:, None, :] - pat_pred['xy'][None, :, :], axis=2).min(1)
    is_sym = np.mean(dd < 0.012) > 0.95
    csym = int(good.sum()) if is_sym else cstr
    return nT, csym, cstr, nas, bool(is_sym)
