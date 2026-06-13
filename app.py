"""
DataForge EDU — Entry Point
Plataforma de Aprendizagem de Machine Learning para Universidades
"""

import streamlit as st
import os, sys

sys.path.insert(0, os.path.dirname(__file__))

# ── PAGE CONFIG ──────────────────────────────────────
st.set_page_config(
    page_title="DataForge EDU",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from modules.utils import inject_css, C_TEXT_MUTE, C_BORDER, C_SURFACE, C_ACCENT, C_TEXT_SEC, C_TEXT
from modules.auth import render_login_page, render_sidebar_user, get_user_role

# ── CSS GLOBAL ───────────────────────────────────────
inject_css()

# ══════════════════════════════════════════════════════
# PÁGINAS AUXILIARES (definidas antes de serem usadas)
# ══════════════════════════════════════════════════════
def _render_em_breve(titulo: str, descricao: str):
    from modules.utils import page_header
    page_header(titulo, "Em desenvolvimento", "🚧")
    st.markdown(f"""
    <div style="background:{C_SURFACE};border:1px solid {C_BORDER};border-radius:16px;
    padding:3rem;text-align:center;margin-top:2rem;">
        <div style="font-size:48px;margin-bottom:1rem;">🚧</div>
        <div style="font-size:20px;font-weight:700;color:#E8EBF0;margin-bottom:.6rem;">{titulo}</div>
        <div style="font-size:14px;color:{C_TEXT_MUTE};">{descricao}</div>
        <div style="font-size:12px;color:{C_TEXT_MUTE};margin-top:1rem;">
            Volta ao Dashboard para ver o percurso de aprendizagem completo.
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_historico(username: str):
    from modules.utils import (page_header, section_title, get_historico_modelos)
    page_header("Historial de Modelos", "Todos os modelos que treinaste", "📊")

    historico = get_historico_modelos(username)
    if not historico:
        st.markdown(f"""
        <div style="text-align:center;padding:3rem;color:{C_TEXT_MUTE};">
            Ainda sem modelos treinados. Vai a Classificação ou Regressão para começar! 🚀
        </div>
        """, unsafe_allow_html=True)
        return

    for h in reversed(historico[-20:]):
        tipo = h.get("tipo", "")
        algo = h.get("algo", h.get("algoritmo", ""))
        data = h.get("data", "")[:16].replace("T", " ")

        metricas = ""
        if tipo == "classificacao" and h.get("accuracy"):
            metricas = f"Accuracy: {h.get('accuracy',0):.1%} · F1: {h.get('f1',0):.1%}"
        elif tipo == "regressao" and h.get("r2"):
            metricas = f"R²: {h.get('r2',0):.4f} · RMSE: {h.get('rmse',0):.4f}"

        icone = "🎯" if tipo == "classificacao" else "📈" if tipo == "regressao" else "🔵"

        st.markdown(f"""
        <div style="background:{C_SURFACE};border:1px solid {C_BORDER};border-radius:10px;
        padding:.8rem 1.1rem;margin-bottom:.6rem;display:flex;align-items:center;gap:12px;">
            <span style="font-size:22px;">{icone}</span>
            <div style="flex:1;">
                <div style="font-size:14px;font-weight:600;color:{C_TEXT};">{algo}</div>
                <div style="font-size:12px;color:{C_TEXT_MUTE};">{tipo.capitalize()} · {data}</div>
                <div style="font-size:12px;color:{C_TEXT_SEC};margin-top:2px;">{metricas}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── AUTENTICAÇÃO ─────────────────────────────────────
authenticator, auth_status, username, config = render_login_page()

if auth_status is not True:
    st.stop()

name = st.session_state.get("name", username)
role = get_user_role(username, config)

# ── SIDEBAR ──────────────────────────────────────────
with st.sidebar:
    render_sidebar_user(name, username, role, authenticator)

    st.markdown(f'<div style="font-size:11px;font-weight:700;letter-spacing:.08em;color:{C_TEXT_MUTE};padding:4px 0 8px;text-transform:uppercase;">Menu Principal</div>', unsafe_allow_html=True)

    MENU_ITEMS = [
        ("🏠 Dashboard",               "dashboard"),
        ("── SUPERVISIONADO",           None),
        ("🎯 Classificação",            "classificacao"),
        ("📈 Regressão",                "regressao"),
        ("── NÃO SUPERVISIONADO",       None),
        ("🔵 Clustering",               "clustering"),
        ("🔻 Redução Dimensionalidade", "reducao"),
        ("🚨 Detecção de Anomalias",    "anomalias"),
        ("── REDES NEURAIS",            None),
        ("🧠 MLP / Redes Neurais",      "mlp"),
        ("── DEEP LEARNING",            None),
        ("👁️ Visão Computacional",      "visao"),
        ("📝 NLP",                      "nlp"),
        ("⏱️ Séries Temporais",         "series"),
        ("🎨 Modelos Generativos",      "generativo"),
        ("── REFORÇO",                  None),
        ("🎮 Aprendizagem p/ Reforço",  "reforco"),
        ("── FERRAMENTAS",              None),
        ("⚔️ Comparador de Modelos",    "comparador"),
        ("📊 Meu Historial",            "historico"),
        ("🏆 Certificados",             "certificados"),
    ]

    if "pagina" not in st.session_state:
        st.session_state.pagina = "dashboard"

    for label, key in MENU_ITEMS:
        if key is None:
            secao = label.replace("── ", "")
            st.markdown(f"""
            <div style="font-size:10px;font-weight:700;letter-spacing:.1em;
            color:{C_TEXT_MUTE};padding:10px 0 3px;text-transform:uppercase;
            border-top:1px solid {C_BORDER};margin-top:6px;">{secao}</div>
            """, unsafe_allow_html=True)
        else:
            ativo = st.session_state.pagina == key
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state.pagina = key
                if "guiado_step" in st.session_state:
                    st.session_state.guiado_step = 1
                st.rerun()

    st.markdown("---")
    st.markdown(f'<div style="font-size:11px;color:{C_TEXT_MUTE};text-align:center;">DataForge EDU v1.0 · Angola 🇦🇴</div>', unsafe_allow_html=True)


# ── ROTEADOR DE PÁGINAS ──────────────────────────────
pagina = st.session_state.get("pagina", "dashboard")

if pagina == "dashboard":
    from modules.dashboard import render_dashboard
    render_dashboard(name, username)

elif pagina == "classificacao":
    from modules.supervisionado.classificacao import render_classificacao
    render_classificacao(username)

elif pagina == "regressao":
    from modules.supervisionado.regressao import render_regressao
    render_regressao(username)

elif pagina == "clustering":
    _render_em_breve("🔵 Clustering", "KMeans, DBSCAN, HDBSCAN, Agglomerative, GMM — Fase 2")

elif pagina == "reducao":
    _render_em_breve("🔻 Redução de Dimensionalidade", "PCA, t-SNE, UMAP, ICA — Fase 2")

elif pagina == "anomalias":
    _render_em_breve("🚨 Detecção de Anomalias", "Isolation Forest, LOF, One-Class SVM — Fase 2")

elif pagina == "mlp":
    _render_em_breve("🧠 MLP / Redes Neurais", "MLP Classifier/Regressor, PyTorch personalizado — Fase 2")

elif pagina == "visao":
    _render_em_breve("👁️ Visão Computacional", "CNN, ResNet, VGG16, MobileNetV2, EfficientNet — Fase 2")

elif pagina == "nlp":
    _render_em_breve("📝 NLP", "BoW, TF-IDF, LSTM, Transformer, BERT — Fase 3")

elif pagina == "series":
    _render_em_breve("⏱️ Séries Temporais", "LSTM, GRU, Prophet, ARIMA — Fase 3")

elif pagina == "generativo":
    _render_em_breve("🎨 Modelos Generativos", "VAE, GAN, DCGAN — Fase 3")

elif pagina == "reforco":
    _render_em_breve("🎮 Aprendizagem por Reforço", "Q-Learning, DQN, Policy Gradient, CartPole — Fase 4")

elif pagina == "comparador":
    _render_em_breve("⚔️ Comparador de Modelos", "Treina e compara até 3 algoritmos lado a lado — Fase 2")

elif pagina == "historico":
    _render_historico(username)

elif pagina == "certificados":
    _render_em_breve("🏆 Certificados", "Certificados PDF com QR code verificável — Fase 2")

else:
    from modules.dashboard import render_dashboard
    render_dashboard(name, username)
