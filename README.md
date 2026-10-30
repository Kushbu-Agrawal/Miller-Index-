# Miller index assignment for rare-earth orthoferrites

Deep learning guided Miller index assignment from SAED patterns and FFTs of HRTEM lattice images.
Built by Kushbu Agrawal (guide: Dr. Rahee Walambe), Symbiosis Institute of Technology, Pune.

Upload a SAED pattern, or the FFT of an HRTEM image. The app detects the diffraction spots, predicts the phase and zone axis with a small CNN plus physics features, and then assigns (hkl) to each spot with a template search.

**Research prototype.** Every model here was trained and tested on simulated patterns only. It has not seen a real micrograph. Treat the output as decision support.

## Run locally

    pip install -r requirements.txt
    streamlit run app.py

Try the images in `examples/`.

## Files

| File | What it does |
|---|---|
| `app.py` | Streamlit front end |
| `ui.py` | Page styling, gradient colour maps and plots |
| `.streamlit/config.toml` | Dark theme settings |
| `miller_infer.py` | Inference: image to phase, zone axis and (hkl) per spot |
| `mi_core.py` | Crystal models, pattern simulation, spot detection, indexing search |
| `model.py` | Network definition (image branch plus physics branch) |
| `fused.pt` | Model C, image plus physics (used by the app) |
| `imgonly.pt`, `physonly.pt` | Models A and B for comparison |
| `gen.py`, `train.py`, `evalrun.py`, `baseline.py` | Data generation, training, evaluation, exhaustive-search baseline |
| `results/` | Saved evaluation numbers from the report |

## Retrain (optional)

    python gen.py          # writes train.pkl, val.pkl and the test sets (large files, not in git)
    python train.py fused 1 1 40   # tag, use_img, use_phys, epochs
    python evalrun.py

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repo.
2. Go to share.streamlit.io, click New app, pick the repo, branch `main`, main file `app.py`.
3. Deploy. The first build takes a few minutes because it installs CPU PyTorch.
