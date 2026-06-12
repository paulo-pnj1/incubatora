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
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder
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
# PALETA — LIGHT / ACESSÍVEL
# ══════════════════════════════════════════════════════════════
C_BG        = "#F8F9FC"
C_SURFACE   = "#FFFFFF"
C_SURFACE2  = "#EEF1F8"
C_BORDER    = "#C2C9D6"
C_ACCENT    = "#1A4FD6"
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
})

# ══════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&display=swap');

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

.brand-wrap {{
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 6px;
    padding: 8px 0;
}}
.brand-icon {{
    width: 44px; height: 44px;
    background: {C_ACCENT};
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
}}
.brand-name {{
    font-size: 20px; font-weight: 700;
    color: {C_TEXT};
}}
.brand-sub {{
    font-size: 13px;
    color: {C_TEXT_MUTE};
}}

.page-header {{
    border-bottom: 3px solid {C_BORDER};
    padding-bottom: 1.2rem;
    margin-bottom: 1.8rem;
}}
.page-tag {{
    display: inline-flex;
    font-size: 13px;
    font-weight: 700;
    color: {C_ACCENT};
    background: #E8EEFA;
    border: 2px solid {C_ACCENT};
    border-radius: 6px;
    padding: 4px 14px;
    margin-bottom: 12px;
}}
.page-title {{
    font-size: 32px;
    font-weight: 700;
    color: {C_TEXT};
}}
.page-title span {{ color: {C_ACCENT}; }}

.info-box {{
    background: #E8EEFA;
    border: 2px solid #A0B4E8;
    border-left: 5px solid {C_ACCENT};
    border-radius: 0 8px 8px 0;
    padding: 14px 20px;
    margin: 10px 0 20px;
}}
.warn-box {{
    background: #FFF3E0;
    border: 2px solid #E6A020;
    border-left: 5px solid {C_AMBER};
    border-radius: 0 8px 8px 0;
    padding: 14px 20px;
    margin: 10px 0;
}}
.explain-box {{
    background: #F0F3FA;
    border: 2px solid {C_BORDER};
    border-left: 5px solid {C_GREEN};
    border-radius: 0 8px 8px 0;
    padding: 14px 20px;
    margin: 15px 0;
}}
.explain-box strong {{ color: {C_TEXT}; }}
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
}}
.metric-label {{
    font-size: 13px;
    font-weight: 700;
    color: {C_TEXT_MUTE};
    text-transform: uppercase;
    margin-bottom: 10px;
}}
.metric-value {{
    font-size: 28px;
    font-weight: 700;
    color: {C_ACCENT};
}}
.metric-card.green {{ border-top-color: {C_GREEN}; }}
.metric-card.green .metric-value {{ color: {C_GREEN}; }}
.metric-card.amber {{ border-top-color: {C_AMBER}; }}
.metric-card.amber .metric-value {{ color: {C_AMBER}; }}

.section-title {{
    font-size: 13px;
    font-weight: 700;
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
    color: white !important;
    font-size: 17px !important;
    font-weight: 700 !important;
    padding: 12px 28px !important;
    min-height: 52px !important;
    border-radius: 8px !important;
}}
.stButton > button:hover {{
    background: #1240B0 !important;
}}

[data-testid="stDownloadButton"] button {{
    background: {C_GREEN} !important;
    color: white !important;
}}

.sidebar-footer {{
    margin-top: 2rem;
    font-size: 13px;
    color: {C_TEXT_MUTE};
    border-top: 2px solid {C_BORDER};
    padding-top: 14px;
}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# UTILITÁRIOS
# ══════════════════════════════════════════════════════════════
def info(msg):
    st.markdown(f'<div class="info-box">{msg}</div>', unsafe_allow_html=True)

def warn(msg):
    st.markdown(f'<div class="warn-box">⚠️ {msg}</div>', unsafe_allow_html=True)

def explain(msg, title="📚 Aprenda Machine Learning"):
    st.markdown(f'<div class="explain-box"><strong>{title}</strong><br><br>{msg}</div>', unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

def page_header(tag, title, highlight=""):
    full = title.replace(highlight, f"<span>{highlight}</span>") if highlight else title
    st.markdown(f'<div class="page-header"><div class="page-tag">// {tag}</div><div class="page-title">{full}</div></div>', unsafe_allow_html=True)

def metrics_row(items):
    color_cycle = ["", "green", "amber", "green", ""]
    html = '<div class="metric-row">'
    for i, (lbl, val) in enumerate(items):
        c = color_cycle[i % len(color_cycle)]
        html += f'<div class="metric-card {c}"><div class="metric-label">{lbl}</div><div class="metric-value">{val}</div></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def safe_plot(fig):
    """Garante que o gráfico seja exibido corretamente"""
    st.pyplot(fig)
    plt.close(fig)

def encode_dataframe(df, encoding_method="label"):
    """Codifica variáveis categóricas"""
    df_enc = df.copy()
    le_dict = {}
    
    if encoding_method == "onehot":
        # One-Hot Encoding
        categorical_cols = df_enc.select_dtypes(include='object').columns
        for col in categorical_cols:
            dummies = pd.get_dummies(df_enc[col], prefix=col, drop_first=True)
            df_enc = pd.concat([df_enc.drop(columns=[col]), dummies], axis=1)
        return df_enc, le_dict
    else:
        # Label Encoding
        for col in df_enc.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
            le_dict[col] = le
        return df_enc, le_dict

def plot_confusion_matrix(cm, label_names, title="Matriz de Confusão"):
    n = len(label_names)
    fig, ax = plt.subplots(figsize=(max(5, n), max(4, n)))
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    plt.colorbar(im, ax=ax)
    
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                   fontsize=12, fontweight='bold',
                   color='white' if cm[i, j] > cm.max() / 2 else C_TEXT)
    
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short_names = [str(l)[:15] for l in label_names]
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(short_names, fontsize=10)
    ax.set_xlabel('Previsto', fontsize=12)
    ax.set_ylabel('Real', fontsize=12)
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
    
    st.markdown("---")
    
    modulo = st.radio(
        "Módulos",
        ["⚙ Treino Geral — CSV", "👁 Visão Computacional"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown(f"""
    <div class="sidebar-footer">
        v1.0.0 · DataForge ML<br>
        Classificação · Regressão · Clustering · CNN
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# MÓDULO 01 — TREINO GERAL
# ══════════════════════════════════════════════════════════════
if modulo == "⚙ Treino Geral — CSV":
    page_header("treino_geral", "Treino com Qualquer Dataset", "Qualquer Dataset")
    
    info("Carregue um <strong>ficheiro CSV</strong>. O sistema detecta automaticamente o tipo de problema.")
    
    uploaded = st.file_uploader("Carregar CSV", type="csv", key="general")
    
    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.success(f"✔ {df_raw.shape[0]} linhas × {df_raw.shape[1]} colunas")
        
        with st.expander("📊 Explorar dados"):
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(df_raw.head(10).astype(str), use_container_width=True)
            with col2:
                info_df = pd.DataFrame({
                    'Tipo': df_raw.dtypes.astype(str),
                    'Nulos': df_raw.isnull().sum(),
                    'Únicos': df_raw.nunique()
                })
                st.dataframe(info_df, use_container_width=True)
            
            # Mostrar distribuição da variável alvo se for classificação
            st.markdown("**Estatísticas descritivas**")
            st.dataframe(df_raw.describe(include='all').astype(str), use_container_width=True)
        
        st.markdown("---")
        section("configurar problema")
        
        task_type = st.radio(
            "Tipo de tarefa",
            ["🔵 Classificação", "🟢 Regressão", "🟠 Clustering"],
            horizontal=True
        )
        
        if "🔵" in task_type:
            explain("""
            <span class="emph">🔵 Classificação</span> — prevê uma <strong>categoria discreta</strong>.
            Exemplos: spam ou não spam, tipo de flor, diagnóstico de doença.
            Métricas: Acurácia, Precisão, Recall, F1-Score.
            """)
        elif "🟢" in task_type:
            explain("""
            <span class="emph">🟢 Regressão</span> — prevê um <strong>valor numérico contínuo</strong>.
            Exemplos: preço de casa, temperatura, salário.
            Métricas: R², RMSE, MAE.
            """)
        else:
            explain("""
            <span class="emph">🟠 Clustering</span> — descobre <strong>grupos naturais</strong> nos dados.
            Não usa rótulos. Exemplos: segmentação de clientes, agrupamento de documentos.
            """)
        
        section("pré-processamento")
        
        col1, col2, col3, col4 = st.columns(4)
        
        fill_strategy = col1.selectbox(
            "Valores em falta",
            ["Média/Moda", "Mediana/Moda", "Remover linhas"]
        )
        scaler_choice = col2.selectbox(
            "Normalização",
            ["StandardScaler", "MinMaxScaler", "Nenhum"]
        )
        encoding_method = col3.selectbox(
            "Encoding categórico",
            ["Label Encoding", "One-Hot Encoding"]
        )
        cols_to_drop = col4.multiselect(
            "Remover colunas",
            df_raw.columns.tolist()
        )
        
        if "🔵" in task_type or "🟢" in task_type:
            target_col = st.selectbox(
                "🎯 Variável alvo",
                [c for c in df_raw.columns if c not in cols_to_drop]
            )
            
            # Mostrar distribuição da variável alvo
            if "🔵" in task_type:
                st.write("**Distribuição das classes:**")
                target_dist = df_raw[target_col].value_counts()
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.bar(range(len(target_dist)), target_dist.values, color=PALETTE_CATS[:len(target_dist)])
                ax.set_xticks(range(len(target_dist)))
                ax.set_xticklabels(target_dist.index, rotation=45, ha='right')
                ax.set_ylabel("Frequência")
                ax.set_title(f"Distribuição de {target_col}")
                safe_plot(fig)
        
        section("algoritmo")
        
        if "🔵" in task_type:
            algo = st.selectbox("Algoritmo", [
                "Random Forest", "Gradient Boosting", "Árvore de Decisão",
                "KNN", "SVM", "Regressão Logística", "Naive Bayes", "AdaBoost"
            ])
        elif "🟢" in task_type:
            algo = st.selectbox("Algoritmo", [
                "Random Forest", "Gradient Boosting", "Regressão Linear",
                "Ridge", "Lasso", "KNN", "Árvore de Decisão"
            ])
        else:
            algo = st.selectbox("Algoritmo", [
                "K-Means", "DBSCAN", "Agglomerative"
            ])
        
        section("hiperparâmetros")
        
        hyper = {}
        
        if algo in ["Random Forest", "Random Forest Regressor"]:
            col1, col2, col3 = st.columns(3)
            hyper['n_estimators'] = col1.slider("Árvores", 50, 300, 100, 25)
            hyper['max_depth'] = col2.slider("Profundidade", 3, 20, 10)
            hyper['min_samples_split'] = col3.slider("Min split", 2, 20, 5)
            hyper['class_weight'] = 'balanced' if "🔵" in task_type else None
        elif algo in ["Gradient Boosting", "Gradient Boosting Regressor"]:
            col1, col2, col3 = st.columns(3)
            hyper['n_estimators'] = col1.slider("Estimadores", 50, 200, 100, 25)
            hyper['learning_rate'] = col2.select_slider("Learning rate", [0.01, 0.05, 0.1, 0.2])
            hyper['max_depth'] = col3.slider("Profundidade", 2, 10, 4)
        elif algo == "KNN":
            col1, col2 = st.columns(2)
            hyper['n_neighbors'] = col1.slider("K vizinhos", 3, 31, 5, 2)
            hyper['weights'] = col2.selectbox("Pesos", ['uniform', 'distance'])
        elif algo == "SVM":
            col1, col2 = st.columns(2)
            hyper['C'] = col1.select_slider("C", [0.1, 1.0, 10.0, 100.0])
            hyper['gamma'] = col2.selectbox("Gamma", ['scale', 'auto'])
        elif algo == "K-Means":
            hyper['n_clusters'] = st.slider("K clusters", 2, 10, 3)
        elif algo == "DBSCAN":
            col1, col2 = st.columns(2)
            hyper['eps'] = col1.slider("Eps", 0.1, 2.0, 0.5, 0.1)
            hyper['min_samples'] = col2.slider("Min samples", 2, 10, 4)
        
        if "🟠" not in task_type:
            col1, col2 = st.columns(2)
            test_size = col1.slider("% Teste", 10, 40, 25) / 100
            cv_folds = col2.slider("Folds CV", 3, 10, 5)
        
        if st.button("▶ Treinar Modelo", key="btn_train", use_container_width=True):
            with st.spinner("Processando e treinando..."):
                # Preparar dados
                df_work = df_raw.drop(columns=cols_to_drop, errors='ignore').copy()
                
                # Tratar valores nulos
                for col in df_work.columns:
                    if df_work[col].isnull().any():
                        if fill_strategy == "Remover linhas":
                            df_work = df_work.dropna()
                            break
                        elif pd.api.types.is_numeric_dtype(df_work[col]):
                            if "Mediana" in fill_strategy:
                                df_work[col] = df_work[col].fillna(df_work[col].median())
                            else:
                                df_work[col] = df_work[col].fillna(df_work[col].mean())
                        else:
                            mode_val = df_work[col].mode()
                            df_work[col] = df_work[col].fillna(mode_val[0] if len(mode_val) > 0 else "Unknown")
                
                # Codificar
                enc_method = "label" if "Label" in encoding_method else "onehot"
                df_enc, le_dict = encode_dataframe(df_work, enc_method)
                
                # CLUSTERING
                if "🟠" in task_type:
                    X_clust = df_enc.values
                    
                    if scaler_choice != "Nenhum":
                        scaler = StandardScaler() if scaler_choice == "StandardScaler" else MinMaxScaler()
                        X_clust = scaler.fit_transform(X_clust)
                    
                    if algo == "K-Means":
                        model = KMeans(**hyper, random_state=42, n_init=10)
                    elif algo == "DBSCAN":
                        model = DBSCAN(**hyper)
                    else:
                        model = AgglomerativeClustering(**hyper)
                    
                    labels = model.fit_predict(X_clust)
                    df_work['Cluster'] = labels
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    
                    # Métricas
                    try:
                        sil_score = silhouette_score(X_clust, labels)
                    except:
                        sil_score = float('nan')
                    
                    section("resultados — clustering")
                    metrics_row([
                        ("Clusters", str(n_clusters)),
                        ("Silhouette", f"{sil_score:.4f}" if not np.isnan(sil_score) else "N/A"),
                        ("Amostras", str(len(labels)))
                    ])
                    
                    # Visualização
                    if X_clust.shape[1] >= 2:
                        pca = PCA(n_components=2, random_state=42)
                        X_pca = pca.fit_transform(X_clust)
                        fig, ax = plt.subplots(figsize=(8, 6))
                        unique_labels = sorted(set(labels))
                        for label in unique_labels:
                            mask = labels == label
                            color = PALETTE_CATS[label % len(PALETTE_CATS)] if label != -1 else C_TEXT_MUTE
                            name = f"Cluster {label}" if label != -1 else "Ruído"
                            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=name, alpha=0.7, s=50)
                        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
                        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
                        ax.set_title("Visualização dos Clusters (PCA)")
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        safe_plot(fig)
                    
                    # Perfil dos clusters
                    numeric_cols = df_work.select_dtypes(include=[np.number]).columns
                    if 'Cluster' in numeric_cols:
                        numeric_cols = [c for c in numeric_cols if c != 'Cluster']
                    if len(numeric_cols) > 0:
                        section("perfil médio por cluster")
                        profile = df_work.groupby('Cluster')[numeric_cols].mean().round(3)
                        st.dataframe(profile, use_container_width=True)
                
                # CLASSIFICAÇÃO/REGRESSÃO
                else:
                    # Separar features e target
                    X = df_enc.drop(columns=[target_col], errors='ignore')
                    y = df_enc[target_col]
                    
                    # Verificar target
                    is_classification = "🔵" in task_type
                    n_classes = y.nunique()
                    
                    if is_classification:
                        st.info(f"🎯 Classes encontradas: {n_classes} - {sorted(y.unique())}")
                        if n_classes > 20:
                            st.warning("Muitas classes únicas! Considere Regressão ou agrupar classes.")
                    
                    # Dividir dados
                    stratify = y if is_classification and n_classes <= 20 else None
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=stratify
                    )
                    
                    # Escalar
                    if scaler_choice != "Nenhum":
                        scaler = StandardScaler() if scaler_choice == "StandardScaler" else MinMaxScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)
                    
                    # Criar modelo
                    if is_classification:
                        if algo == "Random Forest":
                            model = RandomForestClassifier(**{k:v for k,v in hyper.items() if v is not None}, random_state=42, n_jobs=-1)
                        elif algo == "Gradient Boosting":
                            model = GradientBoostingClassifier(**hyper, random_state=42)
                        elif algo == "Árvore de Decisão":
                            model = DecisionTreeClassifier(random_state=42, max_depth=hyper.get('max_depth', 10))
                        elif algo == "KNN":
                            model = KNeighborsClassifier(**hyper)
                        elif algo == "SVM":
                            model = SVC(**hyper, probability=True, random_state=42)
                        elif algo == "Regressão Logística":
                            model = LogisticRegression(max_iter=1000, random_state=42)
                        elif algo == "Naive Bayes":
                            model = GaussianNB()
                        elif algo == "AdaBoost":
                            model = AdaBoostClassifier(random_state=42)
                        else:
                            model = RandomForestClassifier(random_state=42)
                    else:
                        if algo == "Random Forest":
                            model = RandomForestRegressor(**{k:v for k,v in hyper.items() if v is not None}, random_state=42, n_jobs=-1)
                        elif algo == "Gradient Boosting":
                            model = GradientBoostingRegressor(**hyper, random_state=42)
                        elif algo == "Regressão Linear":
                            model = LinearRegression()
                        elif algo == "Ridge":
                            model = Ridge(alpha=hyper.get('alpha', 1.0))
                        elif algo == "Lasso":
                            model = Lasso(alpha=hyper.get('alpha', 0.1))
                        elif algo == "KNN":
                            model = KNeighborsRegressor(**hyper)
                        elif algo == "Árvore de Decisão":
                            model = DecisionTreeRegressor(random_state=42, max_depth=hyper.get('max_depth', 10))
                        else:
                            model = RandomForestRegressor(random_state=42)
                    
                    # Treinar
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # Avaliar
                    if is_classification:
                        acc = accuracy_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                        
                        section("métricas — classificação")
                        metrics_row([
                            ("Acurácia", f"{acc:.4f}"),
                            ("F1-Score", f"{f1:.4f}"),
                            ("Precisão", f"{prec:.4f}"),
                            ("Recall", f"{rec:.4f}")
                        ])
                        
                        # Matriz de confusão
                        labels_sorted = sorted(y_test.unique())
                        label_names = le_dict.get(target_col, {}).classes_.tolist() if target_col in le_dict else [str(l) for l in labels_sorted]
                        cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
                        fig = plot_confusion_matrix(cm, label_names, f"{algo} - Matriz de Confusão")
                        safe_plot(fig)
                        
                        # Classification report
                        with st.expander("📋 Relatório detalhado por classe"):
                            report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True, zero_division=0)
                            report_df = pd.DataFrame(report).T.round(3)
                            st.dataframe(report_df, use_container_width=True)
                        
                        # Curva ROC (apenas binário)
                        if n_classes == 2 and hasattr(model, 'predict_proba'):
                            try:
                                y_prob = model.predict_proba(X_test)[:, 1]
                                auc = roc_auc_score(y_test, y_prob)
                                fpr, tpr, _ = roc_curve(y_test, y_prob)
                                
                                fig, ax = plt.subplots(figsize=(6, 5))
                                ax.plot(fpr, tpr, color=C_ACCENT, lw=2, label=f'AUC = {auc:.3f}')
                                ax.plot([0, 1], [0, 1], color=C_TEXT_MUTE, lw=1.5, linestyle='--', label='Aleatório')
                                ax.fill_between(fpr, tpr, alpha=0.1, color=C_ACCENT)
                                ax.set_xlabel('FPR (1 - Especificidade)')
                                ax.set_ylabel('TPR (Recall/Sensibilidade)')
                                ax.set_title('Curva ROC')
                                ax.legend()
                                ax.grid(True, alpha=0.3)
                                safe_plot(fig)
                            except:
                                pass
                        
                        # Validação cruzada
                        try:
                            cv_stratify = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42) if n_classes <= 20 else KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_stratify, scoring='accuracy')
                            
                            fig, ax = plt.subplots(figsize=(8, 4))
                            bars = ax.bar(range(1, cv_folds+1), cv_scores, color=PALETTE_CATS[:cv_folds])
                            ax.axhline(cv_scores.mean(), color=C_RED, linestyle='--', linewidth=2, label=f'Média: {cv_scores.mean():.4f}')
                            ax.fill_between(range(1, cv_folds+1), cv_scores.mean() - cv_scores.std(), cv_scores.mean() + cv_scores.std(), alpha=0.2, color=C_ACCENT)
                            ax.set_xlabel('Fold')
                            ax.set_ylabel('Acurácia')
                            ax.set_title(f'Validação Cruzada ({cv_folds} folds)')
                            ax.legend()
                            ax.grid(True, alpha=0.3, axis='y')
                            for i, (bar, score) in enumerate(zip(bars, cv_scores)):
                                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{score:.3f}', ha='center', fontsize=10)
                            safe_plot(fig)
                        except Exception as e:
                            warn(f"CV não disponível: {str(e)[:100]}")
                        
                    else:  # Regressão
                        r2 = r2_score(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        mae = mean_absolute_error(y_test, y_pred)
                        
                        section("métricas — regressão")
                        metrics_row([
                            ("R² Score", f"{r2:.4f}"),
                            ("RMSE", f"{rmse:.4f}"),
                            ("MAE", f"{mae:.4f}")
                        ])
                        
                        # Real vs Previsto
                        fig, ax = plt.subplots(figsize=(6, 5))
                        ax.scatter(y_test, y_pred, alpha=0.5, color=C_ACCENT, s=30)
                        min_val = min(y_test.min(), y_pred.min())
                        max_val = max(y_test.max(), y_pred.max())
                        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal')
                        ax.set_xlabel('Valor Real')
                        ax.set_ylabel('Valor Previsto')
                        ax.set_title('Real vs Previsto')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        safe_plot(fig)
                        
                        # Resíduos
                        residuals = y_test - y_pred
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.hist(residuals, bins=30, color=C_ACCENT, alpha=0.7, edgecolor='black')
                        ax.axvline(0, color='red', linestyle='--', linewidth=2)
                        ax.set_xlabel('Resíduo')
                        ax.set_ylabel('Frequência')
                        ax.set_title('Distribuição dos Resíduos')
                        ax.grid(True, alpha=0.3, axis='y')
                        safe_plot(fig)
                        
                        # Validação cruzada
                        try:
                            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2')
                            fig, ax = plt.subplots(figsize=(8, 4))
                            bars = ax.bar(range(1, cv_folds+1), cv_scores, color=PALETTE_CATS[:cv_folds])
                            ax.axhline(cv_scores.mean(), color=C_RED, linestyle='--', linewidth=2, label=f'Média: {cv_scores.mean():.4f}')
                            ax.set_xlabel('Fold')
                            ax.set_ylabel('R²')
                            ax.set_title(f'Validação Cruzada ({cv_folds} folds)')
                            ax.legend()
                            ax.grid(True, alpha=0.3, axis='y')
                            for i, (bar, score) in enumerate(zip(bars, cv_scores)):
                                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{score:.3f}', ha='center', fontsize=10)
                            safe_plot(fig)
                        except Exception as e:
                            warn(f"CV não disponível: {str(e)[:100]}")
                    
                    # Feature Importance (para modelos baseados em árvore)
                    if hasattr(model, 'feature_importances_'):
                        section("importância das variáveis")
                        feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f'Feature_{i}' for i in range(X.shape[1])]
                        importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
                        
                        fig, ax = plt.subplots(figsize=(8, max(5, len(importance)*0.3)))
                        top_n = min(15, len(importance))
                        top_features = importance.tail(top_n)
                        colors = [C_ACCENT if i % 2 == 0 else C_GREEN for i in range(len(top_features))]
                        ax.barh(range(len(top_features)), top_features.values, color=colors)
                        ax.set_yticks(range(len(top_features)))
                        ax.set_yticklabels(top_features.index)
                        ax.set_xlabel('Importância')
                        ax.set_title(f'Top {top_n} Variáveis Mais Importantes')
                        ax.grid(True, alpha=0.3, axis='x')
                        safe_plot(fig)
                    
                    # Download do modelo
                    model_bytes = pickle.dumps(model)
                    st.download_button(
                        "⬇ Descarregar Modelo (.pkl)",
                        model_bytes,
                        f"modelo_{algo.lower().replace(' ', '_')}.pkl",
                        mime="application/octet-stream"
                    )
                    
                    # Previsão interativa
                    section("previsão interativa")
                    st.info("Digite valores para uma nova amostra e veja a previsão do modelo")
                    
                    input_cols = st.columns(min(4, len(feature_names)))
                    user_input = {}
                    
                    for idx, feat in enumerate(feature_names):
                        col = input_cols[idx % len(input_cols)]
                        if feat in df_raw.columns and df_raw[feat].dtype == 'object':
                            options = df_raw[feat].dropna().unique().tolist()
                            user_input[feat] = col.selectbox(feat, options, key=f"inp_{feat}")
                        else:
                            min_val = float(df_raw[feat].min()) if feat in df_raw.columns else 0.0
                            max_val = float(df_raw[feat].max()) if feat in df_raw.columns else 100.0
                            default = float(df_raw[feat].mean()) if feat in df_raw.columns else 0.0
                            user_input[feat] = col.number_input(feat, min_value=min_val, max_value=max_val, value=default, key=f"inp_{feat}")
                    
                    if st.button("🔮 Prever", key="btn_predict"):
                        try:
                            # Codificar input
                            input_df = pd.DataFrame([user_input])
                            input_enc, _ = encode_dataframe(input_df, enc_method)
                            input_values = input_enc.values
                            
                            if scaler_choice != "Nenhum":
                                input_values = scaler.transform(input_values)
                            
                            prediction = model.predict(input_values)[0]
                            
                            if is_classification and target_col in le_dict:
                                prediction = le_dict[target_col].inverse_transform([int(prediction)])[0]
                            
                            st.success(f"🎯 **Previsão:** {prediction}")
                            
                            if is_classification and n_classes == 2 and hasattr(model, 'predict_proba'):
                                proba = model.predict_proba(input_values)[0]
                                st.write(f"**Probabilidades:** {dict(zip(label_names, proba))}")
                        except Exception as e:
                            st.error(f"Erro na previsão: {str(e)}")
    else:
        warn("Carregue um ficheiro CSV para começar.")


# ══════════════════════════════════════════════════════════════
# MÓDULO 02 — VISÃO COMPUTACIONAL (opcional)
# ══════════════════════════════════════════════════════════════
elif modulo == "👁 Visão Computacional":
    page_header("visao_computacional", "Visão Computacional com CNN", "CNN")
    
    info("Treine uma Rede Neural Convolucional (CNN) com suas próprias imagens")
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        TORCH_AVAILABLE = True
    except ImportError:
        TORCH_AVAILABLE = False
        st.error("PyTorch não instalado. Execute: `pip install torch torchvision`")
    
    if TORCH_AVAILABLE:
        section("1 · definir classes")
        
        n_classes = st.number_input("Número de classes", min_value=2, max_value=10, value=2, step=1)
        
        class_images = {}
        class_names = []
        
        cols = st.columns(min(4, int(n_classes)))
        
        for i in range(int(n_classes)):
            col = cols[i % len(cols)]
            name = col.text_input(f"Classe {i+1}", value=f"classe_{i+1}", key=f"name_{i}")
            class_names.append(name)
            images = col.file_uploader(
                f"Imagens - {name}",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key=f"imgs_{i}"
            )
            class_images[name] = images if images else []
        
        total_images = sum(len(v) for v in class_images.values())
        valid_classes = [c for c, imgs in class_images.items() if len(imgs) > 0]
        
        if total_images > 0:
            st.success(f"✔ {total_images} imagens em {len(valid_classes)} classes")
            
            section("2 · configuração")
            
            col1, col2, col3 = st.columns(3)
            img_size = col1.select_slider("Tamanho", [32, 64, 128], value=64)
            epochs = col2.slider("Épocas", 5, 50, 15)
            batch_size = col3.select_slider("Batch", [8, 16, 32], value=16)
            
            col1, col2, col3 = st.columns(3)
            lr = col1.select_slider("Learning rate", [0.0001, 0.001, 0.01], value=0.001)
            dropout = col2.slider("Dropout", 0.0, 0.7, 0.3, 0.05)
            test_split = col3.slider("% Teste", 10, 40, 20) / 100
            
            if st.button("▶ Treinar CNN", key="btn_cnn"):
                with st.spinner("Processando imagens..."):
                    X, y = [], []
                    label_map = {name: idx for idx, name in enumerate(valid_classes)}
                    
                    for name in valid_classes:
                        for img_file in class_images[name]:
                            try:
                                img = Image.open(img_file).convert('RGB').resize((img_size, img_size))
                                arr = np.array(img, dtype=np.float32) / 255.0
                                X.append(arr)
                                y.append(label_map[name])
                            except:
                                st.warning(f"Erro: {img_file.name}")
                    
                    X = np.array(X)
                    y = np.array(y)
                    
                    if len(X) < 4:
                        st.error("Precisa de pelo menos 4 imagens")
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_split, random_state=42, stratify=y
                        )
                        
                        # Converter para PyTorch tensors
                        X_train_t = torch.from_numpy(X_train.transpose(0, 3, 1, 2))
                        X_test_t = torch.from_numpy(X_test.transpose(0, 3, 1, 2))
                        y_train_t = torch.from_numpy(y_train)
                        y_test_t = torch.from_numpy(y_test)
                        
                        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
                        test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size)
                        
                        # CNN simples
                        class SimpleCNN(nn.Module):
                            def __init__(self):
                                super().__init__()
                                self.conv1 = nn.Sequential(
                                    nn.Conv2d(3, 32, 3, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2),
                                    nn.Dropout2d(dropout)
                                )
                                self.conv2 = nn.Sequential(
                                    nn.Conv2d(32, 64, 3, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2),
                                    nn.Dropout2d(dropout)
                                )
                                self.fc = nn.Sequential(
                                    nn.Flatten(),
                                    nn.Linear(64 * (img_size//4) * (img_size//4), 128),
                                    nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(128, len(valid_classes))
                                )
                            
                            def forward(self, x):
                                x = self.conv1(x)
                                x = self.conv2(x)
                                x = self.fc(x)
                                return x
                        
                        device = torch.device("cpu")
                        model = SimpleCNN().to(device)
                        criterion = nn.CrossEntropyLoss()
                        optimizer = optim.Adam(model.parameters(), lr=lr)
                        
                        progress_bar = st.progress(0)
                        history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}
                        
                        for epoch in range(epochs):
                            model.train()
                            train_loss, train_correct = 0, 0
                            for xb, yb in train_loader:
                                xb, yb = xb.to(device), yb.to(device)
                                optimizer.zero_grad()
                                out = model(xb)
                                loss = criterion(out, yb)
                                loss.backward()
                                optimizer.step()
                                train_loss += loss.item()
                                train_correct += (out.argmax(1) == yb).sum().item()
                            
                            model.eval()
                            val_loss, val_correct = 0, 0
                            with torch.no_grad():
                                for xb, yb in test_loader:
                                    xb, yb = xb.to(device), yb.to(device)
                                    out = model(xb)
                                    val_loss += criterion(out, yb).item()
                                    val_correct += (out.argmax(1) == yb).sum().item()
                            
                            train_acc = train_correct / len(y_train)
                            val_acc = val_correct / len(y_test)
                            
                            history['loss'].append(train_loss / len(train_loader))
                            history['val_loss'].append(val_loss / len(test_loader))
                            history['acc'].append(train_acc)
                            history['val_acc'].append(val_acc)
                            
                            progress_bar.progress((epoch + 1) / epochs)
                            st.caption(f"Época {epoch+1}/{epochs} - Loss: {history['loss'][-1]:.4f} - Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")
                        
                        section("resultados")
                        metrics_row([
                            ("Acurácia", f"{history['val_acc'][-1]:.4f}"),
                            ("Melhor Val Acc", f"{max(history['val_acc']):.4f}"),
                            ("Épocas", str(epochs))
                        ])
                        
                        # Curvas
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                        ax1.plot(history['loss'], label='Treino')
                        ax1.plot(history['val_loss'], label='Validação')
                        ax1.set_xlabel('Época')
                        ax1.set_ylabel('Loss')
                        ax1.set_title('Curva de Loss')
                        ax1.legend()
                        ax1.grid(True, alpha=0.3)
                        
                        ax2.plot(history['acc'], label='Treino')
                        ax2.plot(history['val_acc'], label='Validação')
                        ax2.set_xlabel('Época')
                        ax2.set_ylabel('Acurácia')
                        ax2.set_title('Curva de Acurácia')
                        ax2.legend()
                        ax2.grid(True, alpha=0.3)
                        
                        safe_plot(fig)
                        
                        st.session_state['cnn_model'] = model
                        st.session_state['cnn_classes'] = valid_classes
                        st.session_state['cnn_size'] = img_size
                        
            section("3 · classificar nova imagem")
            
            if 'cnn_model' in st.session_state:
                test_img = st.file_uploader("Carregar imagem", type=["jpg", "jpeg", "png"], key="test_cnn")
                
                if test_img:
                    img = Image.open(test_img).convert('RGB').resize((st.session_state['cnn_size'], st.session_state['cnn_size']))
                    arr = np.array(img, dtype=np.float32) / 255.0
                    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
                    
                    model = st.session_state['cnn_model']
                    model.eval()
                    with torch.no_grad():
                        out = model(tensor)
                        probs = torch.softmax(out, 1)[0].numpy()
                        pred_idx = np.argmax(probs)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(img, caption="Imagem", width=200)
                    with col2:
                        st.success(f"**Predição:** {st.session_state['cnn_classes'][pred_idx]}")
                        st.write(f"**Confiança:** {probs[pred_idx]*100:.1f}%")
                        
                        fig, ax = plt.subplots(figsize=(6, 3))
                        ax.barh(st.session_state['cnn_classes'], probs, color=PALETTE_CATS[:len(probs)])
                        ax.set_xlim(0, 1)
                        ax.set_xlabel("Probabilidade")
                        safe_plot(fig)
        else:
            warn("Carregue imagens para começar.")
    else:
        st.info("PyTorch não instalado. Use `pip install torch torchvision` para ativar CNNs.")