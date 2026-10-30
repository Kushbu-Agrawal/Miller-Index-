"""Inference: image (SAED, or FFT of an HRTEM image) -> phase, zone axis, (hkl) per spot."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import mi_core as m
from model import Net
from evalrun import predict

def load(tag='fused'):
    ck = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), tag + '.pt'), weights_only=False)
    net = Net(11, 13, bool(ck['use_img']), bool(ck['use_phys'])); net.load_state_dict(ck['state']); net.eval()
    return net, ck['mu'], ck['sd']

def prepare(arr, is_hrtem_fft=False):
    """arr: 2-D float image. Resized to 256 x 256, normalised to [0,1]."""
    import torch.nn.functional as F
    a = np.asarray(arr, np.float32)
    if a.ndim == 3: a = a.mean(2)
    t = torch.from_numpy(a)[None, None]
    a = F.interpolate(t, size=(m.IMG, m.IMG), mode='area' if a.shape[0] >= m.IMG else 'bilinear')[0, 0].numpy()
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    return a

def analyse(arr, model=None, topk=3, is_fft=False):
    net, mu, sd = model if model else load()
    img = prepare(arr)
    xy, w, c = m.detect_spots(img, centre_search=not is_fft)
    s = dict(img=m.cnn_image(img).astype(np.float16), phys=m.phys_features2(xy, w))
    cands = predict(net, mu, sd, [s], topk)[0]
    best = None
    for p, j, conf in cands:
        pat = m.PATTERNS[(p, j)]
        a, sc, T = m.index_pattern(xy, w, pat)
        if a is not None and (best is None or sc > best['score']):
            best = dict(phase=m.PHASE_NAMES[p], zone=pat['zone'], score=sc, conf=conf, pat=pat, assign=a, transform=T)
    return dict(img=img, spots=xy, weights=w, centre=c, best=best, candidates=[(m.PHASE_NAMES[p], m.ZA_LIST[p][j], conf) for p, j, conf in cands])

def table(res):
    b = res['best']; rows = []
    if b is None: return rows
    for (x, y), a in zip(res['spots'], b['assign']):
        if a < 0: continue
        g = float(np.hypot(x, y))
        rows.append(dict(hkl=tuple(int(v) for v in b['pat']['hkl'][a]), d_A=round(1 / g, 3), g_inv_A=round(g, 4)))
    return rows

def overlay(res, path):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 5.2)); ax.imshow(res['img'], cmap='gray'); ax.axis('off')
    b = res['best']
    if b is not None:
        for (x, y), a in zip(res['spots'], b['assign']):
            px = np.array([x, y]) * m.S_NOM + res['centre']
            if a >= 0:
                h, k, l = (int(v) for v in b['pat']['hkl'][a])
                ax.text(px[0] + 3, px[1] - 3, f'{h}{k}{l}'.replace('-', '̅') if False else f'({h} {k} {l})', color='#ffd24d', fontsize=6)
            ax.scatter(px[0], px[1], s=50, facecolors='none', edgecolors='#33ff77' if a >= 0 else '#ff4d4d', linewidths=0.8)
        ax.set_title(f"{b['phase']}  zone axis [{' '.join(map(str, b['zone']))}]", fontsize=9)
    fig.savefig(path, dpi=160, bbox_inches='tight'); plt.close(fig)
