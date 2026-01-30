import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
from datetime import datetime

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="PhytoScan Pro | Master 2 IA",
    page_icon="🌿",
    layout="wide"
)

# --- INJECTION DU STYLE AVANCÉ ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    @media (prefers-color-scheme: dark) {
        .stApp { background: linear-gradient(135deg, #1e1e2f 0%, #121212 100%); }
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        margin-bottom: 20px;
    }
    @media (prefers-color-scheme: dark) {
        .glass-card { background: rgba(0, 0, 0, 0.3); color: white; }
    }

    .status-healthy { color: #2ecc71; font-weight: 800; font-size: 1.8rem; }
    .status-warning { color: #e67e22; font-weight: 800; font-size: 1.8rem; }
    
    .centered-header { text-align: center; padding: 30px 0; }
    </style>
    """, unsafe_allow_html=True)

# --- INITIALISATION & CHARGEMENT ---
if 'history' not in st.session_state:
    st.session_state.history = []

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('cnn_model.h5')

model = load_my_model()

class_names = {
    0: 'Alternaria Leaf Spot', 1: 'Bacterial Blight', 
    2: 'Fusarium Wilt', 3: 'Healthy Leaf', 4: 'Verticillium Wilt'
}

# --- HEADER CENTRÉ ---
st.markdown(f"""
    <div class="centered-header">
        <h1 style='font-size: 3.2em; font-weight: 900; color: #2E7D32; margin-bottom:0;'>PHYTO<span style='color: #1b5e20;'>SCAN</span> IA</h1>
        <p style='font-size: 1.2em; opacity: 0.8;'>Expertise Pathologique par Vision Artificielle</p>
        <p style='font-size: 0.9em; font-style: italic;'>Session active : {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
    </div>
    """, unsafe_allow_html=True)

# --- ZONE DE TRAVAIL ---
tab1, tab2 = st.tabs(["🔍 Analyse en temps réel", "📊 Historique & Benchmarking"])

with tab1:
    col_input, col_result = st.columns([1, 1.2], gap="large")

    with col_input:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📸 Acquisition de l'échantillon")
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_container_width=True, caption="Échantillon chargé")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("⚡ Résultats du Diagnostic")
        
        if uploaded_file:
            with st.spinner('Calcul des probabilités...'):
                # Inférence
                img_res = image.resize((64, 64))
                img_arr = tf.keras.preprocessing.image.img_to_array(img_res) / 255.0
                preds = model.predict(np.expand_dims(img_arr, axis=0))
                
                # Conversion des probabilités
                probs = tf.nn.softmax(preds[0]).numpy()
                idx = np.argmax(probs)
                label = class_names[idx]
                conf = probs[idx] * 100

                # Affichage Diagnostic
                status_class = "status-healthy" if idx == 3 else "status-warning"
                st.markdown(f"<div class='{status_class}'>{label}</div>", unsafe_allow_html=True)
                st.write(f"Confiance du modèle : **{conf:.2f}%**")

                # --- TABLEAU COMPARATIF DES CLASSES (CORRECTION ICI) ---
                st.write("---")
                st.markdown("### 📊 Comparaison entre les classes")
                
                # Création d'un dictionnaire pour le tableau
                data_comp = {
                    "Pathologie": list(class_names.values()),
                    "Probabilité (%)": [round(p * 100, 2) for p in probs]
                }
                df_comp = pd.DataFrame(data_comp).sort_values(by="Probabilité (%)", ascending=False)
                
                # Affichage d'un tableau propre plutôt qu'un graphique qui peut bugger
                st.table(df_comp)

                st.session_state.history.append({
                    "Heure": datetime.now().strftime("%H:%M"),
                    "Diagnostic": label,
                    "Confiance": f"{conf:.2f}%"
                })
        else:
            st.info("Veuillez charger une image pour activer l'analyse comparative.")
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📋 Rapport & Logs")
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history).iloc[::-1], use_container_width=True)
    else:
        st.warning("Historique vide.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<p style='text-align: center; color: gray;'>M2 IA & Big Data - 2026</p>", unsafe_allow_html=True)