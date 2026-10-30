"""Streamlit front end:  streamlit run app.py"""
import os, time
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import miller_infer as mi
import ui

st.set_page_config(page_title='Miller index assignment', page_icon='◈', layout='wide')
st.markdown(ui.CSS, unsafe_allow_html=True)

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = {
    'SAED example 1 (LaFeO3)': 'saed_example_1.png',
    'SAED example 2 (DyFeO3)': 'saed_example_2.png',
    'HRTEM FFT example (SmFeO3)': 'hrtem_fft_example.png',
}

@st.cache_resource
def get_model():
    return mi.load()

@st.cache_data(show_spinner=False)
def run(arr_bytes, shape, is_fft):
    arr = np.frombuffer(arr_bytes, dtype=np.uint8).reshape(shape)
    t0 = time.time()
    res = mi.analyse(arr, get_model(), is_fft=is_fft)
    res['seconds'] = time.time() - t0
    return res

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('## Input')
    source = st.radio('Source', ['Upload my own', 'Try an example'], label_visibility='collapsed')
    img = None
    if source == 'Upload my own':
        up = st.file_uploader('SAED pattern or FFT of an HRTEM image', type=['png', 'jpg', 'jpeg', 'tif', 'tiff'])
        if up is not None:
            img = Image.open(up).convert('L'); name = up.name
    else:
        pick = st.selectbox('Example', list(EXAMPLES))
        name = EXAMPLES[pick]
        img = Image.open(os.path.join(HERE, 'examples', name)).convert('L')
    default_fft = source == 'Try an example' and 'hrtem' in (name if img else '')
    is_fft = st.toggle('Image is already an FFT of an HRTEM image', value=default_fft,
                       help='Leave off for SAED patterns. The app then finds the direct beam itself.')
    st.markdown('---')
    st.markdown('### About')
    st.caption('Kushbu Agrawal  \nGuide: Dr. Rahee Walambe  \nSymbiosis Institute of Technology, Pune')
    st.caption('Trained and tested on simulated patterns only. It has not seen a real micrograph.')

# ---------------------------------------------------------------- main
st.markdown(ui.HERO, unsafe_allow_html=True)

if img is None:
    st.markdown('<div class="empty"><div class="big">Add a pattern to begin</div>'
                'Upload a file in the sidebar, or pick one of the built-in examples.</div>', unsafe_allow_html=True)
else:
    arr = np.array(img)
    with st.spinner('Detecting spots and searching for the best match'):
        res = run(arr.tobytes(), arr.shape, is_fft)
    b = res['best']

    if b is None:
        st.error('Not enough spots detected. Check contrast, and make sure the pattern is centred.')
        st.image(img, caption='Your input', width=360)
    else:
        n_spots = len(res['spots'])
        n_idx = int(np.sum(np.array(b['assign']) >= 0))
        zone = '[' + ' '.join(map(str, b['zone'])) + ']'
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(ui.card('Phase', b['phase'], b['pat'] and 'best candidate', accent=True), unsafe_allow_html=True)
        c2.markdown(ui.card('Zone axis', zone, 'beam direction'), unsafe_allow_html=True)
        c3.markdown(ui.card('Match score', f"{b['score']:.2f}", 'higher is better'), unsafe_allow_html=True)
        c4.markdown(ui.card('Spots indexed', f'{n_idx} / {n_spots}', f"{res['seconds']:.2f} s"), unsafe_allow_html=True)

        st.markdown(ui.section('Pattern maps', 'Same pattern, three views. Spots are coloured by d-spacing.'), unsafe_allow_html=True)
        left, right = st.columns([1.15, 1])
        with left:
            t1, t2, t3 = st.tabs(['Indexed pattern', 'Gradient map', 'Spot density'])
            with t1: st.pyplot(ui.indexed_map(res), use_container_width=True)
            with t2: st.pyplot(ui.gradient_map(res), use_container_width=True)
            with t3: st.pyplot(ui.density_map(res), use_container_width=True)
        with right:
            st.markdown('<div class="card"><div class="lab">Top candidates</div>' + ui.candidate_bars(res['candidates']) + '</div>',
                        unsafe_allow_html=True)
            st.markdown('<div style="height:14px"></div><div class="note">Solid rings are spots that got an (hkl). Dashed grey rings are detected '
                        'spots that matched nothing in the chosen pattern.</div>', unsafe_allow_html=True)

        st.markdown(ui.section('Indexed reflections', 'Sorted from the largest d-spacing, which is the innermost ring.'), unsafe_allow_html=True)
        rows = mi.table(res)
        df = pd.DataFrame(rows)
        if not df.empty:
            df['hkl'] = df['hkl'].apply(lambda t: '(' + ' '.join(map(str, t)) + ')')
            df = df.sort_values('d_A', ascending=False).reset_index(drop=True)
            df.columns = ['hkl', 'd (Å)', '|g| (1/Å)']
            styled = df.style.background_gradient(subset=['d (Å)'], cmap=ui.GRAD.reversed()).format({'d (Å)': '{:.3f}', '|g| (1/Å)': '{:.4f}'})
            st.dataframe(styled, use_container_width=True, hide_index=True, height=min(420, 38 * (len(df) + 1)))
            st.download_button('Download table as CSV', df.to_csv(index=False).encode(), file_name='indexed_reflections.csv', mime='text/csv')

st.markdown('<div class="foot">Research prototype. Decision support only. Trained on simulated SAED and HRTEM-FFT patterns.</div>', unsafe_allow_html=True)
