"""Streamlit front end:  streamlit run app.py"""
import io, numpy as np, streamlit as st
from PIL import Image
import miller_infer as mi

st.set_page_config(page_title='Miller index assignment', layout='wide')
st.title('Miller index assignment for rare-earth orthoferrites')
st.caption('Research prototype. Trained on simulated SAED and HRTEM-FFT patterns. Decision support only.')

@st.cache_resource
def get_model(): return mi.load()

up = st.file_uploader('SAED pattern or FFT of an HRTEM image (PNG, JPG, TIFF)', type=['png', 'jpg', 'jpeg', 'tif', 'tiff'])
is_fft = st.checkbox('This image is already an FFT of an HRTEM lattice image', value=False)
if up is not None:
    arr = np.array(Image.open(up).convert('L'))
    res = mi.analyse(arr, get_model(), is_fft=is_fft)
    b = res['best']
    c1, c2 = st.columns(2)
    if b is None:
        st.error('Not enough spots detected. Check contrast and centring.')
    else:
        mi.overlay(res, '/tmp/overlay.png')
        c1.image('/tmp/overlay.png', caption='Detected spots (green: indexed, red: not indexed)')
        c2.metric('Phase', b['phase']); c2.metric('Zone axis', '[' + ' '.join(map(str, b['zone'])) + ']')
        c2.metric('Match score', f"{b['score']:.2f}")
        c2.write('Top candidates'); c2.table([dict(phase=p, zone=str(z), confidence=round(cf, 3)) for p, z, cf in res['candidates']])
        st.subheader('Indexed reflections'); st.dataframe(mi.table(res))
