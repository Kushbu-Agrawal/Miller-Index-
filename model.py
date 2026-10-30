import torch, torch.nn as nn, torch.nn.functional as F
NF = 36
class ResBlock(nn.Module):
    def __init__(s, ch):
        super().__init__()
        s.b = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.GELU(),
                            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch))
        s.a = nn.GELU()
    def forward(s, x): return s.a(x + s.b(x))
class Encoder(nn.Module):
    def __init__(s, cin=2, w=(32, 64, 128)):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv2d(cin, w[0], 5, 2, 2, bias=False), nn.BatchNorm2d(w[0]), nn.GELU(), nn.MaxPool2d(2))
        s.l1 = nn.Sequential(ResBlock(w[0]), ResBlock(w[0]))
        s.d1 = nn.Sequential(nn.Conv2d(w[0], w[1], 3, 2, 1, bias=False), nn.BatchNorm2d(w[1]), nn.GELU())
        s.l2 = nn.Sequential(ResBlock(w[1]), ResBlock(w[1]))
        s.d2 = nn.Sequential(nn.Conv2d(w[1], w[2], 3, 2, 1, bias=False), nn.BatchNorm2d(w[2]), nn.GELU())
        s.l3 = nn.Sequential(ResBlock(w[2]), ResBlock(w[2]))
        s.out = 2 * w[2]
    def forward(s, x):
        x = s.l3(s.d2(s.l2(s.d1(s.l1(s.stem(x))))))
        return torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1)
def add_fft(img):   # img (B,1,H,W) -> (B,2,H,W), log-FFT channel as in the notebook
    f = torch.log(torch.abs(torch.fft.fftshift(torch.fft.fft2(img), dim=(-2, -1))) + 1e-6)
    mn = f.amin((2, 3), keepdim=True); mx = f.amax((2, 3), keepdim=True)
    return torch.cat([img, (f - mn) / (mx - mn + 1e-6)], 1)
class Net(nn.Module):
    def __init__(s, n_phase, n_za, use_img=True, use_phys=True):
        super().__init__()
        s.use_img, s.use_phys = use_img, use_phys
        s.enc = Encoder()
        d = (s.enc.out if use_img else 0)
        if use_phys:
            s.pm = nn.Sequential(nn.Linear(NF, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU()); d += 128
        s.trunk = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.2))
        s.hp = nn.Linear(256, n_phase); s.hz = nn.Linear(256, n_za)
    def forward(s, img, phys):
        parts = []
        if s.use_img: parts.append(s.enc(add_fft(img)))
        if s.use_phys: parts.append(s.pm(phys))
        h = s.trunk(torch.cat(parts, 1))
        return s.hp(h), s.hz(h)
