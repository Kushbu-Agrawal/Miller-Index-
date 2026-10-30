import sys, pickle, time, json, numpy as np, torch, torch.nn as nn
sys.path.insert(0, '.')
import mi_core as mc
from model import Net
tag, use_img, use_phys, epochs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
torch.set_num_threads(int(sys.argv[5]) if len(sys.argv) > 5 else 1)
torch.manual_seed(0); np.random.seed(0)
def load(n):
    d = mc.upgrade(pickle.load(open(n + '.pkl', 'rb')))
    X = torch.from_numpy(np.stack([s['img'] for s in d]).astype(np.float32))[:, None]
    P = torch.from_numpy(np.stack([s['phys'] for s in d]))
    y = torch.tensor([s['combo'] for s in d])
    return X, P, y
Xtr, Ptr, ytr = load('train'); Xva, Pva, yva = load('val')
mu, sd = Ptr.mean(0), Ptr.std(0) + 1e-6
net = Net(11, 13, bool(use_img), bool(use_phys))
print(tag, 'params', sum(p.numel() for p in net.parameters()), flush=True)
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
bs = 64; steps = epochs * ((len(Xtr) + bs - 1) // bs)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.2)
ce = nn.CrossEntropyLoss(label_smoothing=0.05)
def evaluate(X, P, y):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            lp, lz = net(X[i:i+256], (P[i:i+256] - mu) / sd); out.append((lp.argmax(1), lz.argmax(1)))
    pp = torch.cat([o[0] for o in out]); pz = torch.cat([o[1] for o in out])
    return (pp == y[:, 0]).float().mean().item(), ((pp == y[:, 0]) & (pz == y[:, 1])).float().mean().item()
best, hist = -1, []
for ep in range(1, epochs + 1):
    net.train(); t0 = time.time(); perm = torch.randperm(len(Xtr)); tl = 0
    for i in range(0, len(Xtr), bs):
        idx = perm[i:i+bs]
        lp, lz = net(Xtr[idx], (Ptr[idx] - mu) / sd)
        loss = ce(lp, ytr[idx, 0]) + ce(lz, ytr[idx, 1])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += loss.item() * len(idx)
    pa, ja = evaluate(Xva, Pva, yva)
    hist.append(dict(epoch=ep, loss=tl / len(Xtr), val_phase=pa, val_joint=ja))
    print(f'{tag} ep {ep}/{epochs} loss {tl/len(Xtr):.4f} val phase {pa:.3f} joint {ja:.3f} ({time.time()-t0:.0f}s)', flush=True)
    if ja >= best:
        best = ja; torch.save(dict(state=net.state_dict(), mu=mu, sd=sd, use_img=use_img, use_phys=use_phys), f'{tag}.pt')
json.dump(hist, open(f'{tag}_hist.json', 'w'))
print('done best joint', best, flush=True)
