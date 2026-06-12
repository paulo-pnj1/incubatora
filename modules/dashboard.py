"""
DataForge EDU — Dashboard Principal
Visão geral, progresso, percurso de aprendizagem
"""

import streamlit as st
from modules.utils import (
    page_header, section_title, progresso_bar, badge,
    get_user_progress, C_TEXT, C_TEXT_SEC, C_TEXT_MUTE,
    C_SURFACE, C_SURFACE2, C_BORDER, C_ACCENT, C_GREEN,
    C_AMBER, C_ACCENT2, C_TEAL, C_RED, PALETTE
)

# Mapa completo do percurso de aprendizagem
PERCURSO = [
    {
        "id": "supervisionado_classificacao",
        "titulo": "Classificação",
        "categoria": "Supervisionado",
        "icone": "🎯",
        "cor": C_ACCENT,
        "nivel": "Iniciante",
        "algoritmos": 10,
        "descricao": "KNN, Decision Tree, Random Forest, SVM, Gradient Boosting, XGBoost, LightGBM, AdaBoost, Naive Bayes, Logistic Regression",
        "prerequisitos": []
    },
    {
        "id": "supervisionado_regressao",
        "titulo": "Regressão",
        "categoria": "Supervisionado",
        "icone": "📈",
        "cor": C_GREEN,
        "nivel": "Iniciante",
        "algoritmos": 8,
        "descricao": "Linear, Ridge, Lasso, ElasticNet, SVR, KNN Regressor, RF Regressor, GB Regressor",
        "prerequisitos": ["supervisionado_classificacao"]
    },
    {
        "id": "nao_supervisionado_clustering",
        "titulo": "Clustering",
        "categoria": "Não Supervisionado",
        "icone": "🔵",
        "cor": C_TEAL,
        "nivel": "Intermédio",
        "algoritmos": 7,
        "descricao": "KMeans, DBSCAN, HDBSCAN, Agglomerative, Birch, Mean Shift, GMM",
        "prerequisitos": ["supervisionado_classificacao"]
    },
    {
        "id": "nao_supervisionado_reducao",
        "titulo": "Redução de Dimensionalidade",
        "categoria": "Não Supervisionado",
        "icone": "🔻",
        "cor": C_ACCENT2,
        "nivel": "Intermédio",
        "algoritmos": 6,
        "descricao": "PCA, Kernel PCA, t-SNE, UMAP, LDA, ICA",
        "prerequisitos": ["supervisionado_classificacao"]
    },
    {
        "id": "nao_supervisionado_anomalias",
        "titulo": "Detecção de Anomalias",
        "categoria": "Não Supervisionado",
        "icone": "🚨",
        "cor": C_AMBER,
        "nivel": "Intermédio",
        "algoritmos": 5,
        "descricao": "Isolation Forest, LOF, One-Class SVM, Elliptic Envelope, Autoencoder",
        "prerequisitos": ["nao_supervisionado_clustering"]
    },
    {
        "id": "redes_neurais_mlp",
        "titulo": "Redes Neurais (MLP)",
        "categoria": "Redes Neurais",
        "icone": "🧠",
        "cor": C_ACCENT,
        "nivel": "Avançado",
        "algoritmos": 4,
        "descricao": "MLP Classifier, MLP Regressor, Rede Personalizada PyTorch, Dropout + BatchNorm",
        "prerequisitos": ["supervisionado_regressao"]
    },
    {
        "id": "deep_learning_visao",
        "titulo": "Visão Computacional",
        "categoria": "Deep Learning",
        "icone": "👁️",
        "cor": C_ACCENT2,
        "nivel": "Avançado",
        "algoritmos": 6,
        "descricao": "CNN personalizada, ResNet, VGG16, MobileNetV2, EfficientNet, U-Net",
        "prerequisitos": ["redes_neurais_mlp"]
    },
    {
        "id": "deep_learning_nlp",
        "titulo": "Processamento de Linguagem",
        "categoria": "Deep Learning",
        "icone": "📝",
        "cor": C_GREEN,
        "nivel": "Avançado",
        "algoritmos": 7,
        "descricao": "BoW, TF-IDF, Word2Vec, LSTM, GRU, Transformer, BERT",
        "prerequisitos": ["redes_neurais_mlp"]
    },
    {
        "id": "deep_learning_series",
        "titulo": "Séries Temporais",
        "categoria": "Deep Learning",
        "icone": "⏱️",
        "cor": C_TEAL,
        "nivel": "Avançado",
        "algoritmos": 5,
        "descricao": "LSTM, GRU, TCN, Prophet, ARIMA",
        "prerequisitos": ["redes_neurais_mlp"]
    },
    {
        "id": "deep_learning_generativo",
        "titulo": "Modelos Generativos",
        "categoria": "Deep Learning",
        "icone": "🎨",
        "cor": C_AMBER,
        "nivel": "Expert",
        "algoritmos": 3,
        "descricao": "Autoencoder Variacional (VAE), GAN, DCGAN",
        "prerequisitos": ["deep_learning_visao"]
    },
    {
        "id": "reforco",
        "titulo": "Aprendizagem por Reforço",
        "categoria": "Reforço",
        "icone": "🎮",
        "cor": C_RED,
        "nivel": "Expert",
        "algoritmos": 5,
        "descricao": "Q-Learning, DQN, Policy Gradient, Actor-Critic, CartPole",
        "prerequisitos": ["redes_neurais_mlp"]
    },
]

NIVEL_CORES = {
    "Iniciante": C_GREEN,
    "Intermédio": C_AMBER,
    "Avançado": C_ACCENT,
    "Expert": C_RED,
}

CATEGORIA_ICONES = {
    "Supervisionado": "🎯",
    "Não Supervisionado": "🔵",
    "Redes Neurais": "🧠",
    "Deep Learning": "⚡",
    "Reforço": "🎮",
}


def render_dashboard(name: str, username: str):
    page_header(
        "Dashboard",
        f"Bem-vindo de volta, {name.split()[0]}! Continua o teu percurso em ML.",
        "🏠"
    )

    progress = get_user_progress(username)
    completados = progress.get("completados", [])
    pontos = progress.get("pontos", 0)
    modulos_data = progress.get("modulos", {})

    total = len(PERCURSO)
    feitos = len(completados)
    pct = feitos / total if total > 0 else 0

    # ── KPIs ──────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Módulos Concluídos", f"{feitos}/{total}")
    with c2:
        st.metric("Pontuação Total", f"⭐ {pontos}")
    with c3:
        algos_total = sum(m["algoritmos"] for m in PERCURSO)
        st.metric("Algoritmos Disponíveis", f"50+")
    with c4:
        nivel_actual = "Iniciante" if feitos < 2 else "Intermédio" if feitos < 5 else "Avançado" if feitos < 9 else "Expert"
        st.metric("Nível Actual", nivel_actual)

    # ── BARRA DE PROGRESSO GERAL ───────────────────────
    st.markdown("<div style='margin:1.2rem 0 .4rem;'>", unsafe_allow_html=True)
    progresso_bar(pct, "Progresso geral do percurso")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── PERCURSO DE APRENDIZAGEM ───────────────────────
    section_title("PERCURSO DE APRENDIZAGEM")

    categorias = {}
    for m in PERCURSO:
        cat = m["categoria"]
        categorias.setdefault(cat, []).append(m)

    for cat, modulos in categorias.items():
        icone_cat = CATEGORIA_ICONES.get(cat, "📦")
        st.markdown(f"""
        <div style="font-size:14px;font-weight:700;color:{C_TEXT_SEC};
        margin:1.4rem 0 .8rem;display:flex;align-items:center;gap:8px;">
            <span>{icone_cat}</span> {cat}
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns(min(len(modulos), 3))
        for i, modulo in enumerate(modulos):
            col = cols[i % 3]
            with col:
                _render_modulo_card(modulo, completados, username)

    # ── HISTÓRICO DE ACTIVIDADE ────────────────────────
    section_title("ACTIVIDADE RECENTE")
    historico = progress.get("historico_pontos", [])
    if historico:
        ultimos = list(reversed(historico[-5:]))
        for h in ultimos:
            data = h.get("data", "")[:10]
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;
            padding:.6rem 0;border-bottom:1px solid {C_BORDER};">
                <span style="font-size:18px;">⭐</span>
                <div style="flex:1;">
                    <div style="font-size:13px;color:{C_TEXT};">{h.get("motivo","")}</div>
                    <div style="font-size:11px;color:{C_TEXT_MUTE};">{data}</div>
                </div>
                <div style="font-size:13px;font-weight:700;color:{C_ACCENT};">+{h.get("pontos",0)} pts</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="text-align:center;padding:2rem;color:{C_TEXT_MUTE};font-size:14px;">
            Ainda sem actividade registada. Começa por Classificação! 🚀
        </div>
        """, unsafe_allow_html=True)

    # ── DICA DO DIA ────────────────────────────────────
    import random
    dicas = [
        "💡 Normaliza sempre as features antes de usar KNN ou SVM.",
        "💡 Random Forest raramente precisa de tuning intensivo — é um excelente ponto de partida.",
        "💡 Overfitting? Adiciona regularização (Ridge/Lasso) ou reduz a complexidade do modelo.",
        "💡 Usa curvas de aprendizagem para diagnosticar overfitting vs underfitting.",
        "💡 Para dados desbalanceados, usa F1-score em vez de accuracy.",
        "💡 t-SNE é óptimo para visualizar — nunca uses as suas coordenadas como features.",
        "💡 Gradient Boosting com learning_rate baixo e mais árvores tende a generalizar melhor.",
        "💡 Em séries temporais, nunca mistures dados de treino e teste por ordem cronológica.",
    ]
    dica = dicas[hash(username) % len(dicas)]
    st.markdown(f"""
    <div style="background:rgba(79,142,247,.08);border:1px solid rgba(79,142,247,.2);
    border-radius:12px;padding:1rem 1.2rem;margin-top:1.4rem;font-size:14px;color:{C_TEXT_SEC};">
        {dica}
    </div>
    """, unsafe_allow_html=True)


def _render_modulo_card(modulo: dict, completados: list, username: str):
    mid = modulo["id"]
    feito = mid in completados
    cor = modulo["cor"]
    nivel_cor = NIVEL_CORES.get(modulo["nivel"], C_ACCENT)
    prereqs_ok = all(p in completados for p in modulo["prerequisitos"])
    bloqueado = not prereqs_ok and modulo["prerequisitos"]

    status_icon = "✅" if feito else ("🔒" if bloqueado else "▶️")
    opacidade = "0.45" if bloqueado else "1"
    border_cor = cor if feito else (C_BORDER if bloqueado else C_BORDER)
    border_left = f"4px solid {cor}" if feito else f"1px solid {border_cor}"

    st.markdown(f"""
    <div style="background:{C_SURFACE};border:{border_left};
    border-radius:12px;padding:1rem 1.1rem;margin-bottom:.8rem;
    opacity:{opacidade};transition:.2s;">
        <div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:.5rem;">
            <span style="font-size:22px;">{modulo['icone']}</span>
            <span style="font-size:16px;">{status_icon}</span>
        </div>
        <div style="font-size:14px;font-weight:700;color:{C_TEXT};margin-bottom:3px;">
            {modulo['titulo']}
        </div>
        <div style="font-size:11px;color:{C_TEXT_MUTE};margin-bottom:.6rem;
        display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;">
            {modulo['descricao']}
        </div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;">
            <span style="font-size:10px;font-weight:600;padding:2px 8px;border-radius:10px;
            background:rgba(255,255,255,.05);color:{nivel_cor};border:1px solid {nivel_cor}33;">
                {modulo['nivel']}
            </span>
            <span style="font-size:10px;padding:2px 8px;border-radius:10px;
            background:rgba(255,255,255,.05);color:{C_TEXT_MUTE};">
                {modulo['algoritmos']} algoritmos
            </span>
        </div>
        {"<div style='font-size:11px;color:"+C_TEXT_MUTE+";margin-top:.5rem;'>🔒 Completa os módulos anteriores primeiro</div>" if bloqueado else ""}
    </div>
    """, unsafe_allow_html=True)
