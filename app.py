"""
================================================================
DataForge ML — Plataforma de Machine Learning
Módulos: Treino Geral (CSV) + Visão Computacional (CNN)
================================================================
Execute com:  streamlit run app.py
Dependências: pip install streamlit pandas numpy matplotlib scikit-learn pillow
Opcional:     pip install torch torchvision
================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pickle
import os
from PIL import Image
warnings.filterwarnings('ignore')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    silhouette_score, calinski_harabasz_score,
    mean_squared_error, mean_absolute_error, r2_score
)

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DataForge ML",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
# PALETA — LIGHT / ACESSÍVEL  (contraste WCAG AAA ≥ 7:1)
# ══════════════════════════════════════════════════════════════
C_BG        = "#F8F9FC"
C_SURFACE   = "#FFFFFF"
C_SURFACE2  = "#EEF1F8"
C_BORDER    = "#C2C9D6"
C_ACCENT    = "#1A4FD6"
C_ACCENT2   = "#B5200E"
C_GREEN     = "#0D6B3B"
C_AMBER     = "#7A4800"
C_RED       = "#B5200E"
C_TEXT      = "#0D1117"
C_TEXT_SEC  = "#2D3A52"
C_TEXT_MUTE = "#4A5568"

PALETTE_CATS = ["#1A4FD6", "#0D6B3B", "#7A4800", "#B5200E", "#5B2D8E", "#005F73"]

plt.rcParams.update({
    'figure.facecolor':  C_SURFACE,
    'axes.facecolor':    "#F0F3FA",
    'axes.edgecolor':    C_BORDER,
    'axes.labelcolor':   C_TEXT_SEC,
    'axes.titlecolor':   C_TEXT,
    'xtick.color':       C_TEXT_MUTE,
    'ytick.color':       C_TEXT_MUTE,
    'text.color':        C_TEXT,
    'grid.color':        "#D8DDE8",
    'grid.linestyle':    '--',
    'grid.linewidth':    0.7,
    'legend.facecolor':  C_SURFACE,
    'legend.edgecolor':  C_BORDER,
    'legend.labelcolor': C_TEXT_SEC,
    'font.family':       'sans-serif',
    'font.size':         13,
    'axes.titlesize':    15,
    'axes.labelsize':    13,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    1.2,
})

# ══════════════════════════════════════════════════════════════
# CSS — ACESSÍVEL
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&family=Source+Sans+3:wght@400;600;700&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

html, body, [class*="css"], .stApp, .main {{
    background-color: {C_BG} !important;
    color: {C_TEXT} !important;
    font-family: 'Atkinson Hyperlegible', 'Source Sans 3', sans-serif !important;
    font-size: 18px !important;
    line-height: 1.7 !important;
}}

#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1300px !important;
}}

section[data-testid="stSidebar"] {{
    background: {C_SURFACE} !important;
    border-right: 2px solid {C_BORDER} !important;
}}
section[data-testid="stSidebar"] > div {{
    padding: 1.5rem 1.2rem 2rem;
}}

.brand-wrap {{
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 6px;
    padding: 8px 0;
}}
.brand-icon {{
    width: 44px; height: 44px; flex-shrink: 0;
    background: {C_ACCENT};
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
}}
.brand-name {{
    font-family: 'Atkinson Hyperlegible', sans-serif;
    font-size: 20px; font-weight: 700;
    color: {C_TEXT};
    line-height: 1.15;
}}
.brand-sub {{
    font-size: 13px;
    color: {C_TEXT_MUTE};
    font-weight: 400;
    margin-top: 2px;
}}

.sidebar-div {{
    border: none;
    border-top: 2px solid {C_BORDER};
    margin: 1.2rem 0;
}}
.nav-label {{
    font-size: 13px; font-weight: 700;
    letter-spacing: 0.08em; color: {C_TEXT_MUTE};
    text-transform: uppercase;
    padding: 0 4px; margin-bottom: 10px;
}}

div[data-testid="stRadio"] > label {{ display: none !important; }}
div[data-testid="stRadio"] label[data-baseweb="radio"] {{
    display: flex !important;
    align-items: center !important;
    padding: 14px 16px !important;
    border-radius: 10px !important;
    margin-bottom: 6px !important;
    cursor: pointer !important;
    font-size: 17px !important;
    font-weight: 600 !important;
    color: {C_TEXT_SEC} !important;
    border: 2px solid transparent !important;
    transition: all .15s ease !important;
    min-height: 52px !important;
    background: transparent !important;
}}
div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {{
    background: {C_SURFACE2} !important;
    color: {C_ACCENT} !important;
    border-color: {C_ACCENT} !important;
}}

.page-header {{
    border-bottom: 3px solid {C_BORDER};
    padding-bottom: 1.2rem;
    margin-bottom: 1.8rem;
}}
.page-tag {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: {C_ACCENT};
    background: #E8EEFA;
    border: 2px solid {C_ACCENT};
    border-radius: 6px;
    padding: 4px 14px;
    margin-bottom: 12px;
    text-transform: uppercase;
}}
.page-title {{
    font-family: 'Atkinson Hyperlegible', sans-serif;
    font-size: 32px;
    font-weight: 700;
    color: {C_TEXT};
    line-height: 1.2;
    margin: 0;
}}
.page-title span {{ color: {C_ACCENT}; }}

.info-box {{
    background: #E8EEFA;
    border: 2px solid #A0B4E8;
    border-left: 5px solid {C_ACCENT};
    border-radius: 0 8px 8px 0;
    padding: 14px 20px;
    font-size: 16px;
    color: {C_TEXT_SEC};
    margin: 10px 0 20px;
    line-height: 1.8;
}}
.info-box strong {{ color: {C_TEXT}; font-weight: 700; }}
.warn-box {{
    background: #FFF3E0;
    border: 2px solid #E6A020;
    border-left: 5px solid {C_AMBER};
    border-radius: 0 8px 8px 0;
    padding: 14px 20px;
    font-size: 16px;
    color: {C_TEXT_SEC};
    margin: 10px 0;
    line-height: 1.8;
}}
.explain-box {{
    background: #F0F3FA;
    border: 2px solid {C_BORDER};
    border-left: 5px solid {C_GREEN};
    border-radius: 0 8px 8px 0;
    padding: 14px 20px;
    font-size: 15px;
    color: {C_TEXT_SEC};
    margin: 15px 0;
    line-height: 1.7;
    font-style: normal;
}}
.explain-box strong {{ color: {C_TEXT}; font-weight: 700; }}
.explain-box .emph {{ color: {C_GREEN}; font-weight: 700; }}

.metric-row {{
    display: flex;
    gap: 12px;
    margin: 18px 0;
    flex-wrap: wrap;
}}
.metric-card {{
    flex: 1; min-width: 120px;
    background: {C_SURFACE};
    border: 2px solid {C_BORDER};
    border-top: 5px solid {C_ACCENT};
    border-radius: 10px;
    padding: 16px 18px;
    text-align: center;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}}
.metric-label {{
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: {C_TEXT_MUTE};
    text-transform: uppercase;
    margin-bottom: 10px;
}}
.metric-value {{
    font-size: 28px;
    font-weight: 700;
    color: {C_ACCENT};
    line-height: 1;
}}
.metric-card.green {{ border-top-color: {C_GREEN}; }}
.metric-card.green .metric-value {{ color: {C_GREEN}; }}
.metric-card.amber {{ border-top-color: {C_AMBER}; }}
.metric-card.amber .metric-value {{ color: {C_AMBER}; }}
.metric-card.red {{ border-top-color: {C_RED}; }}
.metric-card.red .metric-value {{ color: {C_RED}; }}

.section-title {{
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: {C_TEXT_SEC};
    text-transform: uppercase;
    margin: 2rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 12px;
}}
.section-title::after {{
    content: '';
    flex: 1;
    height: 2px;
    background: {C_BORDER};
}}

.stButton > button {{
    background: {C_ACCENT} !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Atkinson Hyperlegible', sans-serif !important;
    font-size: 17px !important;
    font-weight: 700 !important;
    padding: 12px 28px !important;
    min-height: 52px !important;
    cursor: pointer !important;
    transition: background .15s ease !important;
    letter-spacing: 0.02em !important;
}}
.stButton > button:hover {{
    background: #1240B0 !important;
    outline: 3px solid {C_ACCENT} !important;
    outline-offset: 2px !important;
}}

[data-testid="stDownloadButton"] button {{
    background: {C_GREEN} !important;
    color: #FFFFFF !important;
    border: none !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    min-height: 48px !important;
    border-radius: 8px !important;
}}

[data-testid="stFileUploader"] {{
    background: {C_SURFACE} !important;
    border: 2px dashed {C_ACCENT} !important;
    border-radius: 10px !important;
    padding: 12px !important;
}}

hr {{
    border: none !important;
    border-top: 2px solid {C_BORDER} !important;
    margin: 1.2rem 0 !important;
}}

.sidebar-footer {{
    margin-top: 2rem;
    font-size: 13px;
    color: {C_TEXT_MUTE};
    border-top: 2px solid {C_BORDER};
    padding-top: 14px;
    line-height: 1.9;
}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# UTILITÁRIOS
# ══════════════════════════════════════════════════════════════
def info(msg):
    st.markdown(f'<div class="info-box">{msg}</div>', unsafe_allow_html=True)

def warn(msg):
    st.markdown(f'<div class="warn-box">⚠ {msg}</div>', unsafe_allow_html=True)

def explain(msg, title="📚 Aprenda Machine Learning"):
    st.markdown(f'''
    <div class="explain-box">
        <strong>{title}</strong><br><br>
        {msg}
    </div>
    ''', unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

def page_header(tag, title, highlight=""):
    full = title.replace(highlight, f"<span>{highlight}</span>") if highlight else title
    st.markdown(f"""
    <div class="page-header">
        <div class="page-tag">// {tag}</div>
        <div class="page-title">{full}</div>
    </div>
    """, unsafe_allow_html=True)

def metrics_row(items):
    color_cycle = ["", "green", "amber", "red", "", "green"]
    html = '<div class="metric-row">'
    for i, (lbl, val) in enumerate(items):
        c = color_cycle[i % len(color_cycle)]
        html += f"""
        <div class="metric-card {c}">
            <div class="metric-label">{lbl}</div>
            <div class="metric-value">{val}</div>
        </div>"""
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def encode_dataframe(df):
    df_enc = df.copy()
    le_dict = {}
    for col in df_enc.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        le_dict[col] = le
    return df_enc, le_dict

def plot_confusion_matrix(cm, label_names, title="Matriz de Confusão"):
    n = len(label_names)
    fig, ax = plt.subplots(figsize=(max(4, n * 1.2), max(3.5, n * 1.0)))
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.04)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=13, fontweight='bold',
                    color='white' if cm[i, j] > cm.max() / 2 else C_TEXT)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    short = [str(l)[:12] for l in label_names]
    ax.set_xticklabels(short, rotation=30, ha='right', fontsize=10)
    ax.set_yticklabels(short, fontsize=10)
    ax.set_xlabel('Previsto'); ax.set_ylabel('Real')
    ax.set_title(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div class="brand-wrap">
        <div class="brand-icon">⚡</div>
        <div>
            <div class="brand-name">DataForge ML</div>
            <div class="brand-sub">machine learning platform</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<hr class="sidebar-div">', unsafe_allow_html=True)
    st.markdown('<div class="nav-label">// módulos</div>', unsafe_allow_html=True)

    modulo = st.radio(
        "nav",
        [
            "⚙  Treino Geral — CSV",
            "👁  Visão Computacional",
        ],
        label_visibility="collapsed"
    )

    st.markdown('<hr class="sidebar-div">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sidebar-footer">
        v1.0.0 · DataForge ML<br>
        Classificação · Regressão<br>
        Clustering · CNN · Visão
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# MÓDULO 01 — TREINO GERAL (QUALQUER CSV)
# ══════════════════════════════════════════════════════════════
if modulo == "⚙  Treino Geral — CSV":
    page_header("treino_geral", "Treino com Qualquer Dataset", "Qualquer Dataset")

    info("Carregue <strong>qualquer ficheiro CSV</strong>. O sistema detecta automaticamente "
         "as colunas, o tipo de problema (classificação / regressão / clustering) "
         "e oferece todos os algoritmos disponíveis.")

    uploaded = st.file_uploader(
        "Carregar ficheiro CSV",
        type="csv", key="general",
        help="Ficheiro CSV com dados numéricos e/ou categóricos"
    )

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.success(f"✔ {df_raw.shape[0]} linhas × {df_raw.shape[1]} colunas carregadas")

        with st.expander("📊 Explorar dados"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Primeiras linhas**")
                st.dataframe(df_raw.head(10).astype(str), use_container_width=True)
            with c2:
                st.markdown("**Tipos e valores em falta**")
                info_df = pd.DataFrame({
                    'Tipo': df_raw.dtypes.astype(str),
                    'Nulos': df_raw.isnull().sum(),
                    'Únicos': df_raw.nunique()
                })
                st.dataframe(info_df, use_container_width=True)
            st.markdown("**Estatísticas descritivas**")
            st.dataframe(df_raw.describe(include='all').astype(str), use_container_width=True)

        st.markdown("---")
        section("configurar problema")

        task_type = st.radio(
            "Tipo de tarefa",
            ["🔵 Classificação", "🟢 Regressão", "🟠 Clustering (não supervisionado)"],
            horizontal=True
        )

        if "🔵" in task_type:
            explain("""
            <span class="emph">🔵 Classificação</span> — tarefa de <strong>aprendizagem supervisionada</strong> onde o objetivo é 
            prever uma <strong>categoria ou classe discreta</strong>. Exemplos: prever se um e-mail é spam ou não, 
            classificar uma flor como Iris-setosa/versicolor/virginica, ou diagnosticar uma doença como presente/ausente.
            
            O modelo aprende a partir de exemplos rotulados (dados de treino) e depois generaliza para novos casos.
            As métricas principais são: <strong>Acurácia</strong> (percentual de acertos), <strong>Precisão</strong> (qualidade dos positivos),
            <strong>Recall</strong> (sensibilidade para encontrar todos os positivos) e <strong>F1-Score</strong> (média harmónica entre precisão e recall).
            """)
        elif "🟢" in task_type:
            explain("""
            <span class="emph">🟢 Regressão</span> — tarefa de <strong>aprendizagem supervisionada</strong> onde o objetivo é 
            prever um <strong>valor numérico contínuo</strong>. Exemplos: preço de uma casa, temperatura amanhã, 
            salário de um funcionário, ou tempo de entrega de um pedido.
            
            Diferente da classificação (que prevê categorias), aqui a saída é um número real.
            As métricas principais incluem <strong>R²</strong> (proporção da variância explicada pelo modelo),
            <strong>RMSE</strong> (raiz do erro quadrático médio) e <strong>MAE</strong> (erro absoluto médio).
            """)
        else:
            explain("""
            <span class="emph">🟠 Clustering (não supervisionado)</span> — tarefa de <strong>aprendizagem não supervisionada</strong> 
            onde NÃO existem rótulos pré-definidos. O objetivo é <strong>descobrir grupos naturais</strong> nos dados com base 
            em similaridades intrínsecas.
            
            Exemplos: segmentação de clientes por comportamento de compra, agrupamento de documentos por tema,
            ou identificação de padrões em dados genéticos. Como não há "resposta certa", avaliamos com métricas
            internas como <strong>Silhouette Score</strong> (quão coeso e separado cada cluster é).
            """)

        section("pré-processamento")
        pc1, pc2, pc3 = st.columns(3)

        fill_strategy = pc1.selectbox(
            "Valores em falta",
            ["Média (numérico) / Moda (cat.)", "Mediana / Moda", "Remover linhas com nulos"]
        )
        scaler_choice = pc2.selectbox(
            "Normalização",
            ["StandardScaler (Z-score)", "MinMaxScaler (0-1)", "Nenhum"]
        )
        cols_to_drop_user = pc3.multiselect(
            "Colunas a remover (IDs, etc.)",
            df_raw.columns.tolist()
        )

        explain("""
        <strong>Pré-processamento — por que é fundamental?</strong><br><br>
        
        • <strong>Valores em falta</strong>: Dados reais frequentemente têm lacunas. Ignorá-los pode enviesar o modelo.
          - <em>Média/Mediana</em>: substitui valores nulos pela média (ou mediana) da coluna, preservando a distribuição geral.
          - <em>Remover linhas</em>: elimina observações incompletas, mas perde dados se muitas linhas tiverem nulos.
        
        • <strong>Normalização/Escalamento</strong>: Algoritmos como SVM, KNN e redes neurais são sensíveis à escala das variáveis.
          - <em>StandardScaler</em>: transforma para média 0 e desvio padrão 1. Assume distribuição aproximadamente normal.
          - <em>MinMaxScaler</em>: redimensiona para o intervalo [0,1]. Útil quando valores precisam estar em limites definidos.
        
        • <strong>Remover colunas</strong>: IDs, nomes, ou variáveis irrelevantes devem ser excluídas para não confundir o modelo.
        """)

        if "🔵" in task_type or "🟢" in task_type:
            target_col = st.selectbox(
                "Coluna alvo (label / variável dependente)",
                [c for c in df_raw.columns if c not in cols_to_drop_user]
            )

        section("algoritmo")

        if "🔵" in task_type:
            algo = st.selectbox("Algoritmo de classificação", [
                "Random Forest", "Árvore de Decisão", "Gradient Boosting",
                "AdaBoost", "KNN", "Regressão Logística", "SVM (RBF)", "Naive Bayes",
            ])
            
            algo_explanations = {
                "Random Forest": "🌲 <strong>Random Forest</strong>: Conjunto (ensemble) de muitas árvores de decisão. "
                                "Cada árvore vota, e a classe mais votada vence. Reduz overfitting (sobreajuste) e é muito robusta.",
                "Árvore de Decisão": "🌳 <strong>Árvore de Decisão</strong>: Modelo hierárquico que faz perguntas sucessivas sobre os dados. "
                                    "Muito interpretável (pode ser visualizada), mas propensa a overfitting com poucos dados.",
                "Gradient Boosting": "📈 <strong>Gradient Boosting</strong>: Treina árvores sequencialmente, cada uma corrigindo erros da anterior. "
                                    "Muito preciso, mas mais lento e sensível a hiperparâmetros.",
                "AdaBoost": "⚡ <strong>AdaBoost</strong>: Ajusta pesos das amostras, focando nos exemplos mais difíceis. "
                           "Bom para dados desbalanceados.",
                "KNN": "👥 <strong>K-Nearest Neighbors</strong>: Classifica baseado nos K vizinhos mais próximos. "
                      "Simples e não paramétrico, mas computacionalmente intenso para dados grandes.",
                "Regressão Logística": "📐 <strong>Regressão Logística</strong>: Apesar do nome, é um classificador linear. "
                                      "Modela a probabilidade logarítmica da classe. Bom baseline.",
                "SVM (RBF)": "🔮 <strong>SVM com kernel RBF</strong>: Encontra o hiperplano que maximiza a margem entre classes. "
                            "Muito poderoso para dados de alta dimensionalidade.",
                "Naive Bayes": "📊 <strong>Naive Bayes</strong>: Baseado no Teorema de Bayes, assume independência entre features. "
                              "Rápido e funciona bem com texto (classificação de documentos/spam)."
            }
            explain(algo_explanations.get(algo, "Algoritmo de classificação supervisionada."))
            
        elif "🟢" in task_type:
            algo = st.selectbox("Algoritmo de regressão", [
                "Random Forest Regressor", "Gradient Boosting Regressor",
                "Regressão Linear", "Ridge Regression", "Lasso Regression",
                "KNN Regressor", "Árvore de Decisão Regressor",
            ])
            
            reg_explanations = {
                "Random Forest Regressor": "🌲 <strong>Random Forest Regressor</strong>: Ensemble de árvores que calcula a média das previsões individuais. "
                                          "Lida bem com não-linearidades e outliers.",
                "Gradient Boosting Regressor": "📈 <strong>Gradient Boosting Regressor</strong>: Sequência de árvores otimizando erro residual. "
                                              "Excelente para problemas complexos.",
                "Regressão Linear": "📏 <strong>Regressão Linear</strong>: Assume relação linear entre features e target. "
                                   "Simples, interpretável, mas limitada para padrões complexos.",
                "Ridge Regression": "🔄 <strong>Ridge (L2)</strong>: Regressão linear com penalidade L2 (soma dos quadrados dos coeficientes). "
                                   "Reduz overfitting encolhendo coeficientes, mas nenhum zera.",
                "Lasso Regression": "✂️ <strong>Lasso (L1)</strong>: Regressão linear com penalidade L1 (soma dos valores absolutos). "
                                   "Promove esparsidade — pode zerar coeficientes irrelevantes (seleção automática de features).",
                "KNN Regressor": "👥 <strong>KNN Regressor</strong>: Média dos valores dos K vizinhos mais próximos. "
                                "Não paramétrico, mas caro para grandes datasets.",
                "Árvore de Decisão Regressor": "🌳 <strong>Árvore de Decisão Regressor</strong>: Segmenta o espaço em regiões retangulares "
                                              "e atribui a média dos valores em cada região."
            }
            explain(reg_explanations.get(algo, "Algoritmo de regressão supervisionada."))
            
        else:
            algo = st.selectbox("Algoritmo de clustering", [
                "K-Means", "DBSCAN", "Agglomerative Clustering",
            ])
            
            cluster_explanations = {
                "K-Means": "🎯 <strong>K-Means</strong>: Particiona os dados em K clusters, minimizando a distância intra-cluster. "
                          "Rápido e eficiente, mas requer especificar K e assume clusters esféricos.",
                "DBSCAN": "📍 <strong>DBSCAN</strong>: Agrupa pontos baseado em densidade. Detecta outliers (ruído) e "
                         "não requer número de clusters. Funciona bem com formas arbitrárias.",
                "Agglomerative Clustering": "🌱 <strong>Agglomerative Clustering</strong>: Abordagem hierárquica que começa com cada ponto como um cluster "
                                           "e funde os mais similares iterativamente. Gera uma árvore de clusters (dendrograma)."
            }
            explain(cluster_explanations.get(algo, "Algoritmo de clustering não supervisionado."))

        section("hiperparâmetros")
        ha, hb, hc = st.columns(3)

        hyper = {}
        if algo in ["Random Forest", "Random Forest Regressor"]:
            hyper['n_estimators'] = ha.slider("Nº de árvores", 10, 500, 100, step=10)
            hyper['max_depth'] = hb.slider("Profundidade máx.", 1, 30, 8)
            hyper['min_samples_split'] = hc.slider("Min. amostras split", 2, 20, 2)
            explain("""
            <strong>🌲 Hiperparâmetros do Random Forest:</strong><br><br>
            • <strong>n_estimators</strong>: Número de árvores na floresta. Mais árvores geralmente melhoram a precisão, mas aumentam o tempo de treino.
            • <strong>max_depth</strong>: Profundidade máxima de cada árvore. Limitar evita overfitting.
            • <strong>min_samples_split</strong>: Mínimo de amostras necessárias para dividir um nó. Valores maiores criam árvores mais conservadoras.
            """)
        elif algo in ["Gradient Boosting", "Gradient Boosting Regressor"]:
            hyper['n_estimators'] = ha.slider("Nº de estimadores", 50, 500, 100, step=50)
            hyper['learning_rate'] = hb.select_slider("Taxa de aprendizagem", [0.01, 0.05, 0.1, 0.2, 0.5], value=0.1)
            hyper['max_depth'] = hc.slider("Profundidade máx.", 1, 10, 3)
            explain("""
            <strong>📈 Hiperparâmetros do Gradient Boosting:</strong><br><br>
            • <strong>n_estimators</strong>: Número de árvores sequenciais.
            • <strong>learning_rate</strong> (taxa de aprendizagem): Controla a contribuição de cada árvore.
            • <strong>max_depth</strong>: Profundidade de cada árvore. Valores pequenos (3-5) são comuns para evitar overfitting.
            """)
        elif algo == "K-Means":
            hyper['n_clusters'] = ha.slider("K clusters", 2, 15, 3)
            hyper['n_init'] = hb.slider("N inicializações", 5, 30, 10)
            explain("""
            <strong>🎯 Hiperparâmetros do K-Means:</strong><br><br>
            • <strong>K clusters</strong>: Número de grupos a encontrar.
            • <strong>N inicializações</strong>: K-Means é sensível à inicialização. Múltiplas inicializações reduzem mínimos locais.
            """)
        elif algo == "DBSCAN":
            hyper['eps'] = ha.slider("Epsilon (eps)", 0.1, 5.0, 0.5, step=0.1)
            hyper['min_samples'] = hb.slider("Min. amostras", 2, 20, 5)
            explain("""
            <strong>📍 Hiperparâmetros do DBSCAN:</strong><br><br>
            • <strong>epsilon (eps)</strong>: Raio máximo da vizinhança para considerar dois pontos como conectados.
            • <strong>min_samples</strong>: Número mínimo de pontos para formar uma região densa.
            """)

        if "🟠" not in task_type:
            test_size_g = st.slider("% dados de teste", 10, 40, 25, key="ts_g") / 100
            cv_folds = st.slider("Folds — validação cruzada", 3, 10, 5, key="cv_g")

        if st.button("▶ Treinar / Executar", key="btn_general"):
            with st.spinner("A pré-processar e treinar..."):
                explain("""
                <strong>⚙️ Processo de Treino em Execução:</strong><br><br>
                1. <strong>Pré-processamento</strong>: Tratamento de valores nulos, remoção de colunas, codificação.
                2. <strong>Divisão dos dados</strong>: Separação entre treino (aprende padrões) e teste (avalia generalização).
                3. <strong>Normalização</strong>: Padronização das escalas das features (se selecionada).
                4. <strong>Treino do modelo</strong>: O algoritmo encontra a melhor função features → target.
                5. <strong>Avaliação</strong>: Métricas e visualizações para entender o desempenho.
                """)

                df_work = df_raw.drop(columns=cols_to_drop_user, errors='ignore').copy()

                for col in df_work.columns:
                    if df_work[col].isnull().any():
                        if fill_strategy == "Remover linhas com nulos":
                            df_work = df_work.dropna()
                            break
                        elif not pd.api.types.is_numeric_dtype(df_work[col]):
                            mode_vals = df_work[col].mode()
                            df_work[col] = df_work[col].fillna(mode_vals[0] if len(mode_vals) > 0 else "")
                        else:
                            if "Mediana" in fill_strategy:
                                df_work[col] = df_work[col].fillna(df_work[col].median())
                            else:
                                df_work[col] = df_work[col].fillna(df_work[col].mean())

                df_enc, le_dict = encode_dataframe(df_work)

                # CLUSTERING
                if "🟠" in task_type:
                    X_cl = df_enc.values
                    if scaler_choice != "Nenhum":
                        sc = StandardScaler() if "Standard" in scaler_choice else MinMaxScaler()
                        X_cl = sc.fit_transform(X_cl)

                    if algo == "K-Means":
                        model_cl = KMeans(**hyper, random_state=42)
                    elif algo == "DBSCAN":
                        model_cl = DBSCAN(**hyper)
                    else:
                        model_cl = AgglomerativeClustering(**hyper)

                    labels_cl = model_cl.fit_predict(X_cl)
                    df_work['Cluster'] = labels_cl
                    n_clusters_found = len(set(labels_cl)) - (1 if -1 in labels_cl else 0)

                    try:
                        sil = silhouette_score(X_cl, labels_cl)
                    except:
                        sil = float('nan')
                    try:
                        ch = calinski_harabasz_score(X_cl, labels_cl)
                    except:
                        ch = float('nan')

                    section("resultados — clustering")
                    metrics_row([
                        ("Clusters", str(n_clusters_found)),
                        ("Silhouette", f"{sil:.4f}" if not np.isnan(sil) else "N/A"),
                        ("Calinski-Harabasz", f"{ch:.1f}" if not np.isnan(ch) else "N/A"),
                        ("Amostras", str(len(labels_cl))),
                    ])
                    
                    explain(f"""
                    <strong>📊 Interpretação dos Resultados — Clustering:</strong><br><br>
                    • <strong>Silhouette Score = {sil:.4f if not np.isnan(sil) else 'N/A'}</strong>: 
                      Valores próximos de 1 = clusters bem definidos; próximos de 0 = clusters sobrepostos.
                    • <strong>Calinski-Harabasz Index = {ch:.1f if not np.isnan(ch) else 'N/A'}</strong>: 
                      Quanto MAIOR, melhor a separação dos clusters.
                    """)

                    section("distribuição dos clusters")
                    vc_cl = pd.Series(labels_cl).value_counts().sort_index()
                    fig, ax = plt.subplots(figsize=(8, 3.5))
                    ax.bar([f"Cluster {k}" for k in vc_cl.index], vc_cl.values,
                           color=PALETTE_CATS[:len(vc_cl)], edgecolor='none', width=0.5)
                    for i, v in enumerate(vc_cl.values):
                        ax.text(i, v + 1, str(v), ha='center', fontsize=12, fontweight='bold', color=C_TEXT)
                    ax.set_title("Distribuição por Cluster", fontsize=14, fontweight='bold')
                    ax.grid(axis='y', alpha=0.4)
                    fig.tight_layout(); st.pyplot(fig); plt.close()

                    if X_cl.shape[1] >= 2:
                        section("projecção PCA 2D")
                        pca = PCA(n_components=2, random_state=42)
                        X_pca = pca.fit_transform(X_cl)
                        fig, ax = plt.subplots(figsize=(8, 5))
                        for c_idx in sorted(set(labels_cl)):
                            mask = labels_cl == c_idx
                            lbl = "Ruído" if c_idx == -1 else f"Cluster {c_idx}"
                            color = C_TEXT_MUTE if c_idx == -1 else PALETTE_CATS[c_idx % len(PALETTE_CATS)]
                            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], color=color,
                                       alpha=0.7, s=50, edgecolors='none', label=lbl)
                        ve = pca.explained_variance_ratio_
                        ax.set_xlabel(f"PC1 — {ve[0]*100:.1f}%"); ax.set_ylabel(f"PC2 — {ve[1]*100:.1f}%")
                        ax.set_title("Clusters — Projecção PCA", fontsize=14, fontweight='bold')
                        ax.legend(fontsize=11); ax.grid(True, alpha=0.4)
                        fig.tight_layout(); st.pyplot(fig); plt.close()

                    section("perfil médio por cluster")
                    numeric_cols = df_work.select_dtypes(include=np.number).columns.tolist()
                    if 'Cluster' in numeric_cols:
                        numeric_cols.remove('Cluster')
                    profile_cl = df_work.groupby('Cluster')[numeric_cols].mean().round(3)
                    st.dataframe(profile_cl.reset_index(), use_container_width=True)

                # CLASSIFICAÇÃO / REGRESSÃO
                else:
                    X = df_enc.drop(columns=[target_col], errors='ignore')
                    y = df_enc[target_col]

                    is_regression = "🟢" in task_type
                    stratify_y = y if not is_regression and y.nunique() <= 20 else None

                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X, y, test_size=test_size_g, random_state=42, stratify=stratify_y)

                    if scaler_choice != "Nenhum":
                        sc = StandardScaler() if "Standard" in scaler_choice else MinMaxScaler()
                        X_tr = sc.fit_transform(X_tr)
                        X_te = sc.transform(X_te)

                    # Modelo
                    if algo == "Random Forest":
                        model = RandomForestClassifier(**hyper, random_state=42, n_jobs=-1)
                    elif algo == "Random Forest Regressor":
                        from sklearn.ensemble import RandomForestRegressor
                        model = RandomForestRegressor(**hyper, random_state=42, n_jobs=-1)
                    elif algo == "Gradient Boosting":
                        model = GradientBoostingClassifier(**hyper, random_state=42)
                    elif algo == "Gradient Boosting Regressor":
                        from sklearn.ensemble import GradientBoostingRegressor
                        model = GradientBoostingRegressor(**hyper, random_state=42)
                    elif algo == "AdaBoost":
                        model = AdaBoostClassifier(**hyper, random_state=42)
                    elif algo == "KNN":
                        model = KNeighborsClassifier(**hyper)
                    elif algo == "KNN Regressor":
                        from sklearn.neighbors import KNeighborsRegressor
                        model = KNeighborsRegressor(**hyper)
                    elif algo == "Árvore de Decisão":
                        model = DecisionTreeClassifier(**hyper, random_state=42)
                    elif algo == "Árvore de Decisão Regressor":
                        from sklearn.tree import DecisionTreeRegressor
                        model = DecisionTreeRegressor(**hyper, random_state=42)
                    elif algo == "Regressão Logística":
                        model = LogisticRegression(**hyper, random_state=42)
                    elif algo == "SVM (RBF)":
                        model = SVC(**hyper, probability=True, random_state=42)
                    elif algo == "Naive Bayes":
                        model = GaussianNB()
                    elif algo == "Regressão Linear":
                        model = LinearRegression()
                    elif algo == "Ridge Regression":
                        model = Ridge(**hyper)
                    elif algo == "Lasso Regression":
                        model = Lasso(**hyper)

                    model.fit(X_tr, y_tr)
                    y_pred = model.predict(X_te)

                    # REGRESSÃO
                    if is_regression:
                        mse = mean_squared_error(y_te, y_pred)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_te, y_pred)
                        r2 = r2_score(y_te, y_pred)

                        section("métricas — regressão")
                        metrics_row([
                            ("R² Score", f"{r2:.4f}"),
                            ("RMSE", f"{rmse:.4f}"),
                            ("MAE", f"{mae:.4f}"),
                            ("MSE", f"{mse:.4f}"),
                        ])
                        
                        explain(f"""
                        <strong>📈 Interpretação das Métricas de Regressão:</strong><br><br>
                        • <strong>R² Score = {r2:.4f}</strong>: Proporção da variância explicada pelo modelo.
                        • <strong>RMSE = {rmse:.4f}</strong>: Raiz do Erro Quadrático Médio (dá mais peso a erros grandes).
                        • <strong>MAE = {mae:.4f}</strong>: Erro Absoluto Médio (robusto a outliers).
                        """)

                        col_r1, col_r2 = st.columns(2)
                        with col_r1:
                            section("real vs previsto")
                            fig, ax = plt.subplots(figsize=(5, 4))
                            ax.scatter(y_te, y_pred, alpha=0.5, color=C_ACCENT, s=30, edgecolors='none')
                            mn = min(y_te.min(), y_pred.min()); mx = max(y_te.max(), y_pred.max())
                            ax.plot([mn, mx], [mn, mx], color=C_RED, lw=2, linestyle='--', label='Ideal')
                            ax.set_xlabel("Valor Real"); ax.set_ylabel("Valor Previsto")
                            ax.set_title("Real vs Previsto", fontsize=14, fontweight='bold')
                            ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
                            fig.tight_layout(); st.pyplot(fig); plt.close()

                        with col_r2:
                            section("distribuição dos resíduos")
                            residuals = y_te - y_pred
                            fig, ax = plt.subplots(figsize=(5, 4))
                            ax.hist(residuals, bins=30, color=C_ACCENT, edgecolor='none', alpha=0.85)
                            ax.axvline(0, color=C_RED, lw=2, linestyle='--')
                            ax.set_xlabel("Resíduo"); ax.set_ylabel("Frequência")
                            ax.set_title("Distribuição dos Resíduos", fontsize=14, fontweight='bold')
                            ax.grid(axis='y', alpha=0.3)
                            fig.tight_layout(); st.pyplot(fig); plt.close()

                        section(f"validação cruzada — {cv_folds} folds (R²)")
                        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                        cv_scores = cross_val_score(model, X_tr, y_tr, cv=kf, scoring='r2')
                        fig, ax = plt.subplots(figsize=(7, 2.8))
                        bar_col = [C_ACCENT if s >= cv_scores.mean() else C_TEXT_MUTE for s in cv_scores]
                        ax.bar(range(1, cv_folds + 1), cv_scores, color=bar_col, width=0.5, edgecolor='none')
                        ax.axhline(cv_scores.mean(), color=C_AMBER, lw=2, linestyle='--',
                                   label=f'Média = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')
                        for i, s in enumerate(cv_scores):
                            ax.text(i + 1, s + 0.01, f'{s:.3f}', ha='center', fontsize=11, fontweight='bold', color=C_TEXT)
                        ax.set_xlabel("Fold"); ax.set_ylabel("R²")
                        ax.set_title("Validação Cruzada", fontsize=14, fontweight='bold')
                        ax.legend(fontsize=12); ax.grid(axis='y', alpha=0.3)
                        fig.tight_layout(); st.pyplot(fig); plt.close()

                    else:
                        # CLASSIFICAÇÃO
                        acc = accuracy_score(y_te, y_pred)
                        f1 = f1_score(y_te, y_pred, average='weighted', zero_division=0)
                        prec = precision_score(y_te, y_pred, average='weighted', zero_division=0)
                        rec = recall_score(y_te, y_pred, average='weighted', zero_division=0)

                        section("métricas — classificação")
                        metrics_row([
                            ("Acurácia", f"{acc:.4f}"),
                            ("F1-Score", f"{f1:.4f}"),
                            ("Precisão", f"{prec:.4f}"),
                            ("Recall", f"{rec:.4f}"),
                        ])
                        
                        explain(f"""
                        <strong>🎯 Interpretação das Métricas de Classificação:</strong><br><br>
                        • <strong>Acurácia = {acc:.4f}</strong>: Proporção de previsões corretas.
                        • <strong>Precisão = {prec:.4f}</strong>: Qualidade dos positivos previstos.
                        • <strong>Recall = {rec:.4f}</strong>: Sensibilidade para encontrar positivos.
                        • <strong>F1-Score = {f1:.4f}</strong>: Média harmônica entre precisão e recall.
                        """)

                        labels_u = sorted(y.unique())
                        le_t = le_dict.get(target_col)
                        label_names_g = le_t.classes_.tolist() if le_t else [str(l) for l in labels_u]
                        cm = confusion_matrix(y_te, y_pred, labels=labels_u)

                        col_cm_g, col_report = st.columns(2)
                        with col_cm_g:
                            section("matriz de confusão")
                            fig = plot_confusion_matrix(cm, label_names_g, algo)
                            st.pyplot(fig); plt.close()
                        with col_report:
                            section("relatório de classificação")
                            report = classification_report(y_te, y_pred, target_names=label_names_g,
                                                           output_dict=True, zero_division=0)
                            st.dataframe(pd.DataFrame(report).T.round(3).astype(str), use_container_width=True)

                        if y.nunique() == 2 and hasattr(model, 'predict_proba'):
                            try:
                                y_prob_g = model.predict_proba(X_te)[:, 1]
                                auc = roc_auc_score(y_te, y_prob_g)
                                fpr, tpr, _ = roc_curve(y_te, y_prob_g)
                                section("curva ROC")
                                fig, ax = plt.subplots(figsize=(6, 4))
                                ax.fill_between(fpr, tpr, alpha=0.1, color=C_ACCENT)
                                ax.plot(fpr, tpr, color=C_ACCENT, lw=2.5, label=f'AUC = {auc:.3f}')
                                ax.plot([0, 1], [0, 1], color=C_TEXT_MUTE, lw=1.5, linestyle='--', label='Aleatório')
                                ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
                                ax.set_title('Curva ROC', fontsize=14, fontweight='bold')
                                ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
                                fig.tight_layout(); st.pyplot(fig); plt.close()
                            except:
                                pass

                        section(f"validação cruzada — {cv_folds} folds")
                        skf_g = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42) if y.nunique() <= 20 else KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                        try:
                            cv_scores_g = cross_val_score(model, X_tr, y_tr, cv=skf_g, scoring='accuracy')
                            fig, ax = plt.subplots(figsize=(7, 2.8))
                            bar_col = [C_ACCENT if s >= cv_scores_g.mean() else C_TEXT_MUTE for s in cv_scores_g]
                            ax.bar(range(1, cv_folds + 1), cv_scores_g, color=bar_col, width=0.5, edgecolor='none')
                            ax.axhline(cv_scores_g.mean(), color=C_AMBER, lw=2, linestyle='--',
                                       label=f'Média = {cv_scores_g.mean():.4f} ± {cv_scores_g.std():.4f}')
                            for i, s in enumerate(cv_scores_g):
                                ax.text(i + 1, s + 0.005, f'{s:.3f}', ha='center', fontsize=11, fontweight='bold', color=C_TEXT)
                            ax.set_ylim(0, 1.1)
                            ax.set_xlabel("Fold"); ax.set_ylabel("Acurácia")
                            ax.set_title("Validação Cruzada", fontsize=14, fontweight='bold')
                            ax.legend(fontsize=12); ax.grid(axis='y', alpha=0.3)
                            fig.tight_layout(); st.pyplot(fig); plt.close()
                        except Exception as e:
                            st.warning(f"Validação cruzada não disponível: {e}")

                    # Importância de variáveis
                    if hasattr(model, 'feature_importances_'):
                        section("importância das variáveis — top 15")
                        feat_names = X.columns if hasattr(X, 'columns') else [f'var_{i}' for i in range(X.shape[1])]
                        fi = pd.Series(model.feature_importances_, index=feat_names).sort_values().tail(15)
                        fig, ax = plt.subplots(figsize=(8, max(3, len(fi) * 0.4)))
                        colors_fi = [C_ACCENT if v >= fi.max() * 0.6 else C_TEXT_MUTE for v in fi.values]
                        ax.barh(fi.index, fi.values, color=colors_fi, edgecolor='none', height=0.65)
                        for v, lbl in zip(fi.values, fi.index):
                            ax.text(v + 0.001, lbl, f'{v:.3f}', va='center', fontsize=11, color=C_TEXT)
                        ax.set_xlabel("Importância")
                        ax.set_title("Top 15 Variáveis Mais Importantes", fontsize=14, fontweight='bold')
                        ax.grid(axis='x', alpha=0.3)
                        fig.tight_layout(); st.pyplot(fig); plt.close()

                    # Download modelo
                    model_bytes = pickle.dumps(model)
                    st.download_button(
                        "⬇ Descarregar modelo (.pkl)", model_bytes,
                        f"modelo_{algo.lower().replace(' ', '_')}.pkl",
                        mime="application/octet-stream"
                    )

                    section("previsão interactiva — nova amostra")
                    feat_cols = X.columns if hasattr(X, 'columns') else [f'var_{i}' for i in range(X.shape[1])]
                    user_inputs = {}
                    cols_input = st.columns(min(4, len(feat_cols)))
                    for idx, col_name in enumerate(feat_cols):
                        col_widget = cols_input[idx % len(cols_input)]
                        orig_col = df_raw[col_name] if col_name in df_raw.columns else None
                        if orig_col is not None and not pd.api.types.is_numeric_dtype(orig_col):
                            options = orig_col.dropna().unique().tolist()
                            val_sel = col_widget.selectbox(col_name, options, key=f"inp_{col_name}")
                            le_c = le_dict.get(col_name)
                            if le_c:
                                try:
                                    val_enc = le_c.transform([val_sel])[0]
                                except:
                                    val_enc = 0
                            else:
                                val_enc = 0
                            user_inputs[col_name] = val_enc
                        else:
                            min_v = float(df_raw[col_name].min()) if col_name in df_raw.columns else 0.0
                            max_v = float(df_raw[col_name].max()) if col_name in df_raw.columns else 100.0
                            med_v = float(df_raw[col_name].mean()) if col_name in df_raw.columns else 0.0
                            user_inputs[col_name] = col_widget.number_input(
                                col_name, min_value=min_v, max_value=max_v, value=med_v, key=f"inp_{col_name}")

                    if st.button("▶ Prever nova amostra", key="btn_predict_new"):
                        try:
                            nova = np.array([[user_inputs[c] for c in feat_cols]])
                            if scaler_choice != "Nenhum":
                                nova = sc.transform(nova)
                            pred_new = model.predict(nova)[0]
                            if not is_regression and le_dict.get(target_col):
                                pred_label = le_dict[target_col].inverse_transform([int(pred_new)])[0]
                            else:
                                pred_label = pred_new
                            if is_regression:
                                st.success(f"📊 Previsão: **{pred_label:.4f}**")
                            else:
                                st.success(f"🏷️ Classe prevista: **{pred_label}**")
                                if hasattr(model, 'predict_proba'):
                                    probas = model.predict_proba(nova)[0]
                                    le_t2 = le_dict.get(target_col)
                                    class_names = le_t2.classes_.tolist() if le_t2 else [str(i) for i in range(len(probas))]
                                    prob_df = pd.DataFrame({'Classe': class_names, 'Probabilidade': probas}).sort_values('Probabilidade', ascending=False)
                                    fig, ax = plt.subplots(figsize=(6, 2.5))
                                    ax.barh(prob_df['Classe'], prob_df['Probabilidade'],
                                            color=PALETTE_CATS[:len(prob_df)], edgecolor='none', height=0.5)
                                    ax.set_xlim(0, 1); ax.set_xlabel("Probabilidade")
                                    ax.set_title("Probabilidades por classe", fontsize=13, fontweight='bold')
                                    ax.grid(axis='x', alpha=0.3)
                                    fig.tight_layout(); st.pyplot(fig); plt.close()
                        except Exception as ex:
                            st.error(f"Erro na previsão: {ex}")
    else:
        warn("Carregue um <strong>ficheiro CSV</strong> para começar. "
             "Suporta classificação, regressão e clustering com detecção automática.")


# ══════════════════════════════════════════════════════════════
# MÓDULO 02 — VISÃO COMPUTACIONAL
# ══════════════════════════════════════════════════════════════
elif modulo == "👁  Visão Computacional":
    page_header("visao_computacional", "Visão Computacional — CNN", "CNN")

    info("Treine uma <strong>Rede Neuronal Convolucional (CNN)</strong> com as suas próprias imagens. "
         "Carregue imagens por classe, configure a arquitectura e treine o modelo.")
    
    explain("""
    <strong>👁️ O que é Visão Computacional com CNNs?</strong><br><br>
    Uma <strong>Rede Neural Convolucional (CNN)</strong> é um tipo especial de rede neural projetada para processar imagens.
    
    <strong>🔬 Como funciona:</strong><br>
    1. <strong>Camadas Convolucionais</strong>: Detectam características como bordas, texturas, padrões.
    2. <strong>Camadas de Pooling</strong>: Reduzem dimensionalidade mantendo características importantes.
    3. <strong>Camadas Densas</strong>: Combinam características para classificar a imagem.
    """)

    tf_available = False
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        tf_available = True
    except ImportError:
        st.error("PyTorch não está instalado. Execute: `pip install torch torchvision pillow`")

    if tf_available:
        section("1 · definir classes e carregar imagens")

        num_classes_input = st.number_input(
            "Número de classes", min_value=2, max_value=20, value=2, step=1
        )

        class_names_input = []
        class_images = {}
        cols_cls = st.columns(min(4, int(num_classes_input)))

        for i in range(int(num_classes_input)):
            col_c = cols_cls[i % len(cols_cls)]
            cls_name = col_c.text_input(f"Nome da classe {i+1}", value=f"classe_{i+1}", key=f"cls_{i}")
            class_names_input.append(cls_name)
            imgs = col_c.file_uploader(
                f"Imagens — {cls_name}",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                accept_multiple_files=True,
                key=f"imgs_{i}"
            )
            class_images[cls_name] = imgs if imgs else []

        total_imgs = sum(len(v) for v in class_images.values())
        classes_with_data = [c for c, imgs in class_images.items() if len(imgs) > 0]

        if total_imgs > 0:
            st.success(f"✔ {total_imgs} imagens carregadas em {len(classes_with_data)} classe(s): "
                       + ", ".join([f"{c} ({len(class_images[c])})" for c in classes_with_data]))

            section("2 · configurar arquitectura CNN")

            ca1, ca2, ca3 = st.columns(3)
            img_size = ca1.select_slider("Tamanho da imagem (px)", [32, 48, 64, 96, 128, 224], value=64)
            epochs = ca2.slider("Épocas de treino", 3, 100, 15, step=1)
            batch_size = ca3.select_slider("Batch size", [8, 16, 32, 64, 128], value=16)

            cb1, cb2, cb3 = st.columns(3)
            lr = cb1.select_slider("Taxa de aprendizagem", [0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
            dropout = cb2.slider("Dropout", 0.0, 0.7, 0.3, step=0.05)
            test_split = cb3.slider("% Teste", 10, 40, 20) / 100

            section("camadas convolucionais")
            cc1, cc2, cc3 = st.columns(3)
            n_conv_blocks = cc1.slider("Nº de blocos convolucionais", 1, 4, 2)
            filters_base = cc2.select_slider("Filtros no 1º bloco", [16, 32, 64, 128], value=32)
            dense_units = cc3.select_slider("Nós na camada densa", [64, 128, 256, 512], value=128)

            augment = st.checkbox("Activar Data Augmentation (flip, rotação, zoom)", value=True)

            section("3 · treinar modelo CNN")

            if st.button("▶ Treinar CNN", key="btn_cnn"):
                with st.spinner("A carregar e pré-processar imagens..."):
                    X_all, y_all = [], []
                    label_map = {cls: idx for idx, cls in enumerate(classes_with_data)}

                    for cls_name in classes_with_data:
                        for img_file in class_images[cls_name]:
                            try:
                                pil_img = Image.open(img_file).convert('RGB')
                                pil_img = pil_img.resize((img_size, img_size))
                                arr = np.array(pil_img, dtype=np.float32) / 255.0
                                X_all.append(arr)
                                y_all.append(label_map[cls_name])
                            except Exception as e:
                                st.warning(f"Erro ao carregar {img_file.name}: {e}")

                    X_arr = np.array(X_all, dtype=np.float32)
                    y_arr = np.array(y_all, dtype=np.int64)
                    n_classes_actual = len(classes_with_data)

                    st.success(f"✔ {len(X_arr)} imagens processadas — shape: {X_arr.shape}")

                    if len(X_arr) < 4:
                        st.error("❌ São necessárias pelo menos 4 imagens para treinar.")
                    else:
                        X_tr_cnn, X_te_cnn, y_tr_cnn, y_te_cnn = train_test_split(
                            X_arr, y_arr, test_size=test_split, random_state=42,
                            stratify=y_arr if len(set(y_arr)) > 1 else None
                        )

                        X_tr_t = torch.from_numpy(X_tr_cnn.transpose(0, 3, 1, 2))
                        X_te_t = torch.from_numpy(X_te_cnn.transpose(0, 3, 1, 2))
                        y_tr_t = torch.from_numpy(y_tr_cnn)
                        y_te_t = torch.from_numpy(y_te_cnn)

                        def augment_batch(x):
                            if augment and torch.rand(1).item() > 0.5:
                                x = torch.flip(x, dims=[3])
                            return x

                        train_ds = TensorDataset(X_tr_t, y_tr_t)
                        test_ds  = TensorDataset(X_te_t, y_te_t)
                        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
                        test_loader  = DataLoader(test_ds,  batch_size=batch_size)

                        class CNNModel(nn.Module):
                            def __init__(self):
                                super().__init__()
                                self.conv_blocks = nn.ModuleList()
                                in_ch = 3
                                for block in range(n_conv_blocks):
                                    out_ch = filters_base * (2 ** block)
                                    self.conv_blocks.append(nn.Sequential(
                                        nn.Conv2d(in_ch, out_ch, 3, padding=1),
                                        nn.BatchNorm2d(out_ch),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(out_ch, out_ch, 3, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(2),
                                        nn.Dropout2d(dropout * 0.5),
                                    ))
                                    in_ch = out_ch
                                self.gap = nn.AdaptiveAvgPool2d(1)
                                self.fc  = nn.Sequential(
                                    nn.Flatten(),
                                    nn.Linear(in_ch, dense_units),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(dropout),
                                    nn.Linear(dense_units, n_classes_actual),
                                )
                            def forward(self, x):
                                for blk in self.conv_blocks:
                                    x = blk(x)
                                x = self.gap(x)
                                return self.fc(x)

                        device = torch.device("cpu")
                        model_cnn = CNNModel().to(device)

                        criterion = nn.CrossEntropyLoss()
                        optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=lr)
                        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer_cnn, factor=0.5, patience=5)

                        with st.expander("🔎 Arquitectura do modelo"):
                            st.code(str(model_cnn), language='text')

                        prog_cnn = st.progress(0, text="A treinar CNN...")
                        history_data = {'loss': [], 'val_loss': [],
                                        'accuracy': [], 'val_accuracy': []}

                        best_val_acc = -1
                        best_weights = None
                        patience_count = 0
                        early_stop_patience = 10

                        for epoch in range(epochs):
                            # Treino
                            model_cnn.train()
                            ep_loss, ep_correct, ep_total = 0.0, 0, 0
                            for xb, yb in train_loader:
                                xb = augment_batch(xb.to(device))
                                yb = yb.to(device)
                                optimizer_cnn.zero_grad()
                                out = model_cnn(xb)
                                loss = criterion(out, yb)
                                loss.backward()
                                optimizer_cnn.step()
                                ep_loss   += loss.item() * xb.size(0)
                                ep_correct += (out.argmax(1) == yb).sum().item()
                                ep_total   += xb.size(0)
                            train_loss = ep_loss / ep_total
                            train_acc  = ep_correct / ep_total

                            # Validação
                            model_cnn.eval()
                            val_loss, val_correct, val_total = 0.0, 0, 0
                            with torch.no_grad():
                                for xb, yb in test_loader:
                                    xb, yb = xb.to(device), yb.to(device)
                                    out = model_cnn(xb)
                                    val_loss    += criterion(out, yb).item() * xb.size(0)
                                    val_correct += (out.argmax(1) == yb).sum().item()
                                    val_total   += xb.size(0)
                            val_loss /= val_total
                            val_acc   = val_correct / val_total

                            history_data['loss'].append(train_loss)
                            history_data['val_loss'].append(val_loss)
                            history_data['accuracy'].append(train_acc)
                            history_data['val_accuracy'].append(val_acc)

                            scheduler.step(val_loss)

                            prog_cnn.progress(
                                (epoch + 1) / epochs,
                                text=f"Época {epoch+1}/{epochs} — "
                                     f"loss: {train_loss:.4f} — "
                                     f"acc: {train_acc:.4f} — "
                                     f"val_acc: {val_acc:.4f}"
                            )

                            if val_acc > best_val_acc:
                                best_val_acc = val_acc
                                best_weights = {k: v.clone() for k, v in model_cnn.state_dict().items()}
                                patience_count = 0
                            else:
                                patience_count += 1
                                if patience_count >= early_stop_patience:
                                    st.info(f"Early stopping na época {epoch+1} — melhor val_acc: {best_val_acc:.4f}")
                                    break

                        if best_weights:
                            model_cnn.load_state_dict(best_weights)

                        prog_cnn.progress(1.0, text="Treino concluído ✔")

                        # Métricas finais
                        model_cnn.eval()
                        all_preds, all_probs = [], []
                        test_loss_total, test_correct, test_total = 0.0, 0, 0
                        with torch.no_grad():
                            for xb, yb in test_loader:
                                xb, yb = xb.to(device), yb.to(device)
                                out = model_cnn(xb)
                                test_loss_total += criterion(out, yb).item() * xb.size(0)
                                test_correct    += (out.argmax(1) == yb).sum().item()
                                test_total      += xb.size(0)
                                all_preds.extend(out.argmax(1).cpu().numpy())
                                all_probs.extend(torch.softmax(out, 1).cpu().numpy())
                        test_loss = test_loss_total / test_total
                        test_acc  = test_correct / test_total
                        y_pred_cnn = np.array(all_preds)

                        f1_cnn   = f1_score(y_te_cnn, y_pred_cnn, average='weighted', zero_division=0)
                        prec_cnn = precision_score(y_te_cnn, y_pred_cnn, average='weighted', zero_division=0)
                        rec_cnn  = recall_score(y_te_cnn, y_pred_cnn, average='weighted', zero_division=0)

                        section("métricas finais")
                        metrics_row([
                            ("Acurácia Teste", f"{test_acc:.4f}"),
                            ("F1-Score", f"{f1_cnn:.4f}"),
                            ("Precisão", f"{prec_cnn:.4f}"),
                            ("Recall", f"{rec_cnn:.4f}"),
                            ("Loss Teste", f"{test_loss:.4f}"),
                        ])

                        section("curvas de aprendizagem")
                        col_loss, col_acc = st.columns(2)
                        h = history_data

                        with col_loss:
                            fig, ax = plt.subplots(figsize=(5, 4))
                            ax.plot(h['loss'], color=C_ACCENT, lw=2, label='Treino')
                            ax.plot(h['val_loss'], color=C_RED, lw=2, linestyle='--', label='Validação')
                            ax.set_xlabel("Época"); ax.set_ylabel("Loss")
                            ax.set_title("Curva de Loss", fontsize=14, fontweight='bold')
                            ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
                            fig.tight_layout(); st.pyplot(fig); plt.close()

                        with col_acc:
                            fig, ax = plt.subplots(figsize=(5, 4))
                            ax.plot(h['accuracy'], color=C_GREEN, lw=2, label='Treino')
                            ax.plot(h['val_accuracy'], color=C_AMBER, lw=2, linestyle='--', label='Validação')
                            ax.set_xlabel("Época"); ax.set_ylabel("Acurácia")
                            ax.set_title("Curva de Acurácia", fontsize=14, fontweight='bold')
                            ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
                            fig.tight_layout(); st.pyplot(fig); plt.close()

                        section("matriz de confusão")
                        cm_cnn = confusion_matrix(y_te_cnn, y_pred_cnn)
                        fig = plot_confusion_matrix(cm_cnn, classes_with_data, "CNN")
                        st.pyplot(fig); plt.close()

                        section("relatório de classificação")
                        report_cnn = classification_report(y_te_cnn, y_pred_cnn,
                                                            target_names=classes_with_data,
                                                            output_dict=True, zero_division=0)
                        st.dataframe(pd.DataFrame(report_cnn).T.round(3).astype(str), use_container_width=True)

                        section("amostras do conjunto de teste")
                        n_show = min(12, len(X_te_cnn))
                        indices_show = np.random.choice(len(X_te_cnn), n_show, replace=False)
                        n_cols_show = min(6, n_show)
                        n_rows_show = (n_show + n_cols_show - 1) // n_cols_show
                        fig, axes = plt.subplots(n_rows_show, n_cols_show,
                                                 figsize=(n_cols_show * 2.5, n_rows_show * 2.8))
                        if n_rows_show == 1 and n_cols_show == 1:
                            axes = np.array([[axes]])
                        elif n_rows_show == 1:
                            axes = axes[np.newaxis, :]
                        elif n_cols_show == 1:
                            axes = axes[:, np.newaxis]

                        correct_count = 0
                        for idx_plot, idx_data in enumerate(indices_show):
                            r_p = idx_plot // n_cols_show
                            c_p = idx_plot % n_cols_show
                            ax = axes[r_p, c_p]
                            ax.imshow(X_te_cnn[idx_data])
                            true_cls = classes_with_data[y_te_cnn[idx_data]]
                            pred_cls = classes_with_data[y_pred_cnn[idx_data]]
                            correct = true_cls == pred_cls
                            if correct:
                                correct_count += 1
                            for spine in ax.spines.values():
                                spine.set_edgecolor(C_GREEN if correct else C_RED)
                                spine.set_linewidth(3)
                            ax.set_title(f"Real: {true_cls}\nPrev: {pred_cls}",
                                         fontsize=9,
                                         color=C_GREEN if correct else C_RED,
                                         fontweight='bold')
                            ax.axis('off')

                        for idx_extra in range(n_show, n_rows_show * n_cols_show):
                            r_p = idx_extra // n_cols_show
                            c_p = idx_extra % n_cols_show
                            axes[r_p, c_p].axis('off')

                        fig.suptitle(f"Predições no conjunto de teste — {correct_count}/{n_show} corretas",
                                     fontsize=13, fontweight='bold', color=C_TEXT)
                        fig.tight_layout(); st.pyplot(fig); plt.close()

                        st.session_state['cnn_model'] = model_cnn
                        st.session_state['cnn_classes'] = classes_with_data
                        st.session_state['cnn_img_size'] = img_size

                        try:
                            model_path_cnn = "/tmp/modelo_cnn.pt"
                            torch.save(model_cnn.state_dict(), model_path_cnn)
                            with open(model_path_cnn, 'rb') as f:
                                st.download_button("⬇ Descarregar modelo CNN (.pt)", f.read(),
                                                   "modelo_cnn.pt", mime="application/octet-stream")
                        except Exception:
                            pass

            st.markdown("---")
            section("4 · inferência — classificar nova imagem")

            if 'cnn_model' in st.session_state:
                inf_img = st.file_uploader(
                    "Carregar imagem para classificar",
                    type=["jpg", "jpeg", "png", "bmp", "webp"],
                    key="inf_img"
                )
                if inf_img:
                    m_inf = st.session_state['cnn_model']
                    cls_inf = st.session_state['cnn_classes']
                    sz_inf = st.session_state['cnn_img_size']

                    pil_inf = Image.open(inf_img).convert('RGB')
                    pil_rs = pil_inf.resize((sz_inf, sz_inf))
                    arr_inf = np.array(pil_rs, dtype=np.float32) / 255.0
                    t_inf = torch.from_numpy(arr_inf.transpose(2, 0, 1)).unsqueeze(0)

                    m_inf.eval()
                    with torch.no_grad():
                        out_inf = m_inf(t_inf)
                        probas_inf = torch.softmax(out_inf, 1)[0].numpy().tolist()
                    pred_idx = int(np.argmax(probas_inf))

                    col_img_inf, col_res_inf = st.columns([1, 2])
                    with col_img_inf:
                        st.image(pil_inf, caption="Imagem carregada", width=None)
                    with col_res_inf:
                        st.markdown(f"""
                        <div style="background:{C_GREEN}15;border:1px solid {C_GREEN}44;
                                    border-radius:8px;padding:16px 20px;margin:10px 0;">
                            <div style="font-family:'Space Mono',monospace;font-size:10px;
                                        color:{C_GREEN};font-weight:700;letter-spacing:0.12em;
                                        text-transform:uppercase;margin-bottom:8px;">// resultado</div>
                            <div style="font-family:'Space Mono',monospace;font-size:26px;
                                        font-weight:700;color:{C_TEXT};">
                                {cls_inf[pred_idx]}
                            </div>
                            <div style="font-size:14px;color:{C_TEXT_SEC};margin-top:6px;">
                                Confiança: <strong style="color:{C_GREEN};">{probas_inf[pred_idx]*100:.1f}%</strong>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        section("probabilidades por classe")
                        fig, ax = plt.subplots(figsize=(5, 2.5))
                        colors_p = [C_GREEN if i == pred_idx else C_TEXT_MUTE for i in range(len(cls_inf))]
                        ax.barh(cls_inf, probas_inf, color=colors_p, edgecolor='none', height=0.5)
                        for i, v in enumerate(probas_inf):
                            ax.text(v + 0.01, i, f'{v*100:.1f}%', va='center', fontsize=12, fontweight='bold', color=C_TEXT)
                        ax.set_xlim(0, 1.15)
                        ax.set_xlabel("Probabilidade")
                        ax.grid(axis='x', alpha=0.3)
                        fig.tight_layout(); st.pyplot(fig); plt.close()
            else:
                warn("Treine primeiro o modelo CNN (passo 3) para activar a inferência.")

        else:
            warn("Carregue imagens nas classes definidas acima para continuar.")