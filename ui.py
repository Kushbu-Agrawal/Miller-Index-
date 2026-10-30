"""Plots and HTML pieces for the Streamlit app. Dark navy, rose and gold, gradient colour maps."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter, sobel
import mi_core as m

BG, PANEL, LINE = '#0e1424', '#151d33', '#2a3350'
TEXT, MUTED = '#f1ede4', '#9aa3b8'
ROSE, GOLD, CRIMSON = '#e58a8f', '#d9b26f', '#a8232b'

# rose to gold, used for confidence bars and the d-spacing colour of the spots
GRAD = LinearSegmentedColormap.from_list('rosegold', ['#5a1020', CRIMSON, ROSE, GOLD, '#fff1c9'])
# deep navy to rose, used for the intensity map
INTENSITY = LinearSegmentedColormap.from_list('navyrose', ['#0b1020', '#1d2a55', '#6b2f6b', ROSE, GOLD, '#fff6dc'])

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"], .stApp { font-family: 'IBM Plex Sans', sans-serif; }
.stApp {
  background:
    radial-gradient(1200px 600px at 85% -10%, rgba(229,138,143,0.16), transparent 60%),
    radial-gradient(900px 500px at -10% 10%, rgba(217,178,111,0.10), transparent 55%),
    linear-gradient(180deg, #0e1424 0%, #0b101d 100%);
}
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 2.2rem; max-width: 1280px; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #121a30, #0d1322); border-right: 1px solid #2a3350; }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { font-family: 'EB Garamond', serif; font-weight: 500; letter-spacing: .3px; }

.hero { padding: 34px 40px; border-radius: 22px; border: 1px solid #2a3350;
  background: linear-gradient(120deg, rgba(229,138,143,0.14), rgba(21,29,51,0.9) 45%, rgba(217,178,111,0.10)); margin-bottom: 26px; position: relative; overflow: hidden; }
.hero:after { content: ""; position: absolute; right: -60px; top: -60px; width: 260px; height: 260px; border-radius: 50%;
  background: radial-gradient(circle, rgba(229,138,143,0.28), transparent 65%); }
.eyebrow { font-size: 12px; letter-spacing: 3px; text-transform: uppercase; color: #e58a8f; font-weight: 600; }
.hero h1 { font-family: 'EB Garamond', serif; font-weight: 500; font-size: 46px; line-height: 1.08; margin: 8px 0 10px;
  background: linear-gradient(90deg, #f1ede4 10%, #e58a8f 55%, #d9b26f 100%); -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero p { color: #9aa3b8; font-size: 16px; max-width: 720px; margin: 0; line-height: 1.55; }

.card { background: linear-gradient(160deg, #18213b, #121a30); border: 1px solid #2a3350; border-radius: 18px; padding: 20px 22px; height: 100%; }
.card .lab { font-size: 11px; letter-spacing: 2.2px; text-transform: uppercase; color: #9aa3b8; font-weight: 600; }
.card .val { font-family: 'EB Garamond', serif; font-size: 36px; line-height: 1.15; margin-top: 6px; color: #f1ede4; font-weight: 500; }
.card .sub { color: #9aa3b8; font-size: 13px; margin-top: 4px; }
.card.accent { border-color: rgba(229,138,143,0.55); background: linear-gradient(160deg, rgba(229,138,143,0.20), #131b32 70%); }

.sect { font-family: 'EB Garamond', serif; font-size: 28px; font-weight: 500; margin: 30px 0 4px; color: #f1ede4; }
.sect-sub { color: #9aa3b8; font-size: 14px; margin-bottom: 12px; }
.rule { height: 2px; width: 64px; border-radius: 2px; background: linear-gradient(90deg, #e58a8f, #d9b26f); margin: 6px 0 14px; }

.cand { margin: 14px 0; }
.cand .row { display: flex; justify-content: space-between; font-size: 14px; color: #f1ede4; margin-bottom: 6px; }
.cand .row span:last-child { color: #d9b26f; font-variant-numeric: tabular-nums; }
.track { height: 10px; border-radius: 6px; background: #0b101d; border: 1px solid #2a3350; overflow: hidden; }
.fill { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #5a1020, #a8232b, #e58a8f, #d9b26f); }

.note { border-left: 3px solid #e58a8f; background: rgba(229,138,143,0.07); padding: 12px 16px; border-radius: 0 12px 12px 0; color: #c9cfdd; font-size: 13.5px; }
.empty { text-align: center; padding: 70px 20px; border: 1px dashed #2a3350; border-radius: 20px; color: #9aa3b8; background: rgba(21,29,51,0.5); }
.empty .big { font-family: 'EB Garamond', serif; font-size: 28px; color: #f1ede4; margin-bottom: 6px; }

div[data-testid="stTabs"] button { font-weight: 500; letter-spacing: .2px; }
div[data-testid="stTabs"] [aria-selected="true"] { color: #e58a8f; }
div[data-testid="stFileUploader"] section { background: #151d33; border: 1px dashed #3a4468; border-radius: 14px; }
div[data-testid="stDataFrame"] { border: 1px solid #2a3350; border-radius: 14px; overflow: hidden; }
.stButton button, .stDownloadButton button { border-radius: 12px; border: 1px solid #e58a8f; background: transparent; color: #e58a8f; }
.stButton button:hover, .stDownloadButton button:hover { background: #e58a8f; color: #0e1424; }
.foot { color: #6b7590; font-size: 12px; margin-top: 40px; text-align: center; }
</style>
"""

HERO = """
<div class="hero">
  <div class="eyebrow">Electron diffraction &middot; deep learning</div>
  <h1>Miller index assignment<br/>for rare-earth orthoferrites</h1>
  <p>Drop in a SAED pattern or the FFT of an HRTEM image. The model names the phase and zone axis, then labels every diffraction spot with its (hkl).</p>
</div>
"""

def card(label, value, sub='', accent=False):
    return f'<div class="card{" accent" if accent else ""}"><div class="lab">{label}</div><div class="val">{value}</div><div class="sub">{sub}</div></div>'

def section(title, sub=''):
    return f'<div class="sect">{title}</div><div class="rule"></div>' + (f'<div class="sect-sub">{sub}</div>' if sub else '')

def candidate_bars(cands):
    top = max(c for _, _, c in cands) or 1
    out = []
    for ph, z, c in cands:
        w = max(2.0, 100 * c / top)
        out.append(f'<div class="cand"><div class="row"><span>{ph} &nbsp;<span style="color:#9aa3b8">[{" ".join(map(str, z))}]</span></span>'
                   f'<span>{100 * c:.1f}%</span></div><div class="track"><div class="fill" style="width:{w:.1f}%"></div></div></div>')
    return ''.join(out)

def _hkl_label(h):
    parts = [f'\\bar{{{abs(v)}}}' if v < 0 else str(v) for v in h]
    return '$(' + '\\,'.join(parts) + ')$'

def _style(fig, ax):
    fig.patch.set_facecolor(BG)
    for a in np.atleast_1d(ax):
        a.set_facecolor(BG)
        for s in a.spines.values():
            s.set_color(LINE)

def _spot_px(res):
    return np.array([[x, y] for x, y in res['spots']]) * m.S_NOM + res['centre'] if len(res['spots']) else np.zeros((0, 2))

def indexed_map(res):
    """Pattern on a gradient colour map, spots coloured by d-spacing, (hkl) labels on the indexed ones."""
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    _style(fig, ax)
    ax.imshow(res['img'] ** 0.6, cmap=INTENSITY, interpolation='bilinear')
    ax.set_xticks([]); ax.set_yticks([])
    b = res['best']; px = _spot_px(res)
    if b is not None and len(px):
        g = np.hypot(res['spots'][:, 0], res['spots'][:, 1])
        d = 1.0 / g
        idx = np.array(b['assign']) >= 0
        sc = ax.scatter(px[idx, 0], px[idx, 1], s=130, c=d[idx], cmap=GRAD.reversed(), vmin=d[idx].min(), vmax=d[idx].max(),
                        edgecolors='white', linewidths=0.9, zorder=3)
        ax.scatter(px[~idx, 0], px[~idx, 1], s=110, facecolors='none', edgecolors='#7f8aa8', linewidths=0.9, linestyle='--', zorder=3)
        for (x, y), a in zip(px, b['assign']):
            if a >= 0:
                ax.annotate(_hkl_label([int(v) for v in b['pat']['hkl'][a]]), (x, y), xytext=(7, -9), textcoords='offset points',
                            color=TEXT, fontsize=8, zorder=4,
                            bbox=dict(boxstyle='round,pad=0.18', fc=BG, ec='none', alpha=0.65))
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label('d-spacing (Å)', color=MUTED, fontsize=9)
        cb.ax.tick_params(colors=MUTED, labelsize=8); cb.outline.set_edgecolor(LINE)
    ax.plot(*res['centre'], marker='+', color=GOLD, ms=14, mew=1.4, zorder=4)
    fig.tight_layout()
    return fig

def gradient_map(res):
    """Edge strength of the smoothed pattern. Shows where intensity changes fastest, so spot cores and rings stand out."""
    img = gaussian_filter(res['img'], 1.2)
    gx, gy = sobel(img, axis=1), sobel(img, axis=0)
    mag = np.hypot(gx, gy); mag = mag / (mag.max() + 1e-9)
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    _style(fig, ax)
    im = ax.imshow(mag ** 0.5, cmap='magma', interpolation='bilinear')
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label('gradient magnitude (relative)', color=MUTED, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8); cb.outline.set_edgecolor(LINE)
    fig.tight_layout()
    return fig

def density_map(res):
    """Spot density in reciprocal space: each detected spot is a soft Gaussian weighted by its intensity."""
    px = _spot_px(res)
    n = m.IMG
    yy, xx = np.mgrid[:n, :n]
    z = np.zeros((n, n))
    for (x, y), w in zip(px, res['weights']):
        z += w * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 6.0 ** 2))
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    _style(fig, ax)
    im = ax.imshow(z, cmap=GRAD, interpolation='bicubic')
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label('spot intensity density', color=MUTED, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8); cb.outline.set_edgecolor(LINE)
    fig.tight_layout()
    return fig
