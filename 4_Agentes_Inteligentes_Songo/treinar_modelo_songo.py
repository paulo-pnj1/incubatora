"""
=============================================================
  SISTEMA DE ML - SEGURANÇA NO TRÂNSITO - MUNICÍPIO DO SONGO
  Universidade Kimpa Vita | Engenharia Informática
=============================================================
  Objectivo: Prever se um cidadão acredita que Agentes
  Inteligentes podem reduzir acidentes no trânsito.

  Como usar:
    1. Instalar dependências:
       pip install scikit-learn pandas numpy matplotlib seaborn

    2. Garantir que o ficheiro CSV está na mesma pasta:
       inquerito_songo_500.csv

    3. Executar:
       python treinar_modelo_songo.py
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
import os

warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score, roc_curve
)

# ─────────────────────────────────────────
# 1. CARREGAR DADOS
# ─────────────────────────────────────────

CSV_PATH = 'inquerito_songo_500.csv'

print("=" * 60)
print("  SISTEMA ML - SEGURANÇA NO TRÂNSITO - SONGO")
print("=" * 60)

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Ficheiro não encontrado: {CSV_PATH}\n"
                            "Certifique-se que o CSV está na mesma pasta.")

df = pd.read_csv(CSV_PATH)
print(f"\n✔ Dados carregados: {len(df)} registos, {len(df.columns)} colunas")
print(f"\nColunas: {list(df.columns)}")
print(f"\nAmostra:\n{df.head(3).to_string()}")

# ─────────────────────────────────────────
# 2. PRÉ-PROCESSAMENTO
# ─────────────────────────────────────────

print("\n\n[1/5] PRÉ-PROCESSAMENTO DOS DADOS...")

cat_cols = [
    'Categoria',
    'Presenciou_Acidente',
    'Frequencia_Acidentes',
    'Principal_Causa',
    'Conhece_IA',
    'IA_Pode_Reduzir_Acidentes',
    'Apoia_Modernizacao_Transito'
]

encoders = {}
df_enc = df.copy()

for col in cat_cols:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col])
    encoders[col] = le
    print(f"   {col}: {list(le.classes_)}")

# Features (entradas) e Target (saída)
FEATURES = [
    'Categoria',
    'Presenciou_Acidente',
    'Frequencia_Acidentes',
    'Principal_Causa',
    'Conhece_IA',
    'Apoia_Modernizacao_Transito'
]
TARGET = 'IA_Pode_Reduzir_Acidentes'

X = df_enc[FEATURES]
y = df_enc[TARGET]

print(f"\n   Features: {FEATURES}")
print(f"   Target  : {TARGET}")
print(f"   Distribuição do target: {dict(zip(encoders[TARGET].classes_, y.value_counts().values))}")

# Divisão treino/teste (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n   Treino: {len(X_train)} | Teste: {len(X_test)}")

# ─────────────────────────────────────────
# 3. TREINAR MODELOS
# ─────────────────────────────────────────

print("\n\n[2/5] TREINAMENTO DOS MODELOS...")

models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=6,
        class_weight='balanced', random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1,
        max_depth=4, random_state=42
    ),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=5, class_weight='balanced', random_state=42
    ),
    'Logistic Regression': LogisticRegression(
        class_weight='balanced', max_iter=1000, random_state=42
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    acc  = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    auc  = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0

    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'acc': acc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'auc': auc,
    }

    print(f"\n   [{name}]")
    print(f"     Acurácia Teste : {acc:.4f}")
    print(f"     Cross-Val Mean : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"     AUC-ROC        : {auc:.4f}")

# Melhor modelo por AUC
best_name = max(results, key=lambda k: results[k]['auc'])
best = results[best_name]
print(f"\n   ★ Melhor Modelo: {best_name} (AUC={best['auc']:.4f})")

# ─────────────────────────────────────────
# 4. RELATÓRIO DETALHADO DO MELHOR MODELO
# ─────────────────────────────────────────

print(f"\n\n[3/5] RELATÓRIO DO MELHOR MODELO ({best_name})")
print("-" * 50)
target_names = list(encoders[TARGET].classes_)
print(classification_report(y_test, best['y_pred'], target_names=target_names))

# ─────────────────────────────────────────
# 5. VISUALIZAÇÕES / DASHBOARD
# ─────────────────────────────────────────

print("\n[4/5] GERANDO DASHBOARD DE RESULTADOS...")

fig = plt.figure(figsize=(18, 13))
fig.patch.set_facecolor('#F0F4F8')
gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.4)

# --- Banner título ---
ax0 = fig.add_subplot(gs[0, :])
ax0.set_xlim(0, 1); ax0.set_ylim(0, 1); ax0.axis('off')
ax0.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor='#1A237E', transform=ax0.transAxes))
ax0.text(0.5, 0.68,
         '🚦 Machine Learning — Segurança no Trânsito',
         ha='center', va='center', fontsize=16, fontweight='bold',
         color='white', transform=ax0.transAxes)
ax0.text(0.5, 0.28,
         f'Município do Songo  |  Melhor Modelo: {best_name}  |  '
         f'Acurácia: {best["acc"]:.1%}  |  AUC: {best["auc"]:.3f}  |  500 entrevistados',
         ha='center', va='center', fontsize=10,
         color='#90CAF9', transform=ax0.transAxes)

# --- Comparação de modelos ---
ax1 = fig.add_subplot(gs[1, 0])
names  = list(results.keys())
accs   = [results[n]['acc']     for n in names]
cvs    = [results[n]['cv_mean'] for n in names]
aucs   = [results[n]['auc']     for n in names]
x_pos  = np.arange(len(names))
w = 0.28
ax1.bar(x_pos - w, accs, w, label='Acurácia', color='#1565C0', alpha=0.85)
ax1.bar(x_pos,     cvs,  w, label='Cross-Val', color='#2E7D32', alpha=0.85)
ax1.bar(x_pos + w, aucs, w, label='AUC',       color='#E65100', alpha=0.85)
ax1.set_xticks(x_pos)
ax1.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=7)
ax1.set_ylim(0.4, 1.05)
ax1.set_title('Comparação de Modelos', fontweight='bold', fontsize=9)
ax1.set_ylabel('Score')
ax1.legend(fontsize=7)
ax1.axhline(0.8, color='red', linestyle='--', alpha=0.3)

# --- Matriz de confusão ---
ax2 = fig.add_subplot(gs[1, 1])
cm = confusion_matrix(y_test, best['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=target_names, yticklabels=target_names,
            linewidths=1, annot_kws={'size': 13})
ax2.set_title(f'Matriz de Confusão\n({best_name})', fontweight='bold', fontsize=9)
ax2.set_ylabel('Real'); ax2.set_xlabel('Previsto')

# --- Curva ROC ---
ax3 = fig.add_subplot(gs[1, 2])
for name, r in results.items():
    if r['y_proba'] is not None:
        fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
        ax3.plot(fpr, tpr, label=f"{name.split()[0]} ({r['auc']:.2f})", linewidth=1.5)
ax3.plot([0, 1], [0, 1], 'k--', alpha=0.4)
ax3.set_title('Curva ROC', fontweight='bold', fontsize=9)
ax3.set_xlabel('Taxa Falsos Positivos')
ax3.set_ylabel('Taxa Verdadeiros Positivos')
ax3.legend(fontsize=7)

# --- Importância das features (melhor modelo) ---
ax4 = fig.add_subplot(gs[2, :2])
if hasattr(best['model'], 'feature_importances_'):
    imp = best['model'].feature_importances_
    feat_labels = [
        'Categoria', 'Presenciou Acidente',
        'Frequência Acidentes', 'Causa Principal',
        'Conhece IA', 'Apoia Modernização'
    ]
    idx = np.argsort(imp)
    palette = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(imp)))
    bars = ax4.barh([feat_labels[i] for i in idx], imp[idx], color=palette)
    ax4.set_title(f'Importância das Variáveis — {best_name}', fontweight='bold', fontsize=9)
    ax4.set_xlabel('Importância')
    for bar, val in zip(bars, imp[idx]):
        ax4.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                 f'{val:.3f}', va='center', fontsize=8)

# --- Pizza: distribuição real ---
ax5 = fig.add_subplot(gs[2, 2])
counts = y.value_counts()
labels_pie = [encoders[TARGET].classes_[i] for i in counts.index]
ax5.pie(counts.values, labels=labels_pie, autopct='%1.1f%%',
        colors=['#1565C0', '#C62828'], startangle=140,
        wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'})
ax5.set_title('Distribuição Real\n(Aceita IA?)', fontweight='bold', fontsize=9)

plt.savefig('ml_dashboard_songo.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print("   ✔ Dashboard salvo: ml_dashboard_songo.png")

# ─────────────────────────────────────────
# 6. SALVAR MODELO E ENCODERS
# ─────────────────────────────────────────

print("\n[5/5] SALVANDO MODELO TREINADO...")

with open('modelo_songo.pkl', 'wb') as f:
    pickle.dump({'model': best['model'], 'encoders': encoders, 'features': FEATURES}, f)
print("   ✔ Modelo salvo: modelo_songo.pkl")

# ─────────────────────────────────────────
# 7. FUNÇÃO DE PREVISÃO INTERACTIVA
# ─────────────────────────────────────────

def prever(categoria, presenciou_acidente, frequencia_acidentes,
           causa_principal, conhece_ia, apoia_modernizacao):
    """
    Prevê se um cidadão acredita que IA pode reduzir acidentes.

    Parâmetros:
        categoria            : 'Mototaxista' | 'Motorista' | 'Peão'
        presenciou_acidente  : 'Sim' | 'Não'
        frequencia_acidentes : 'Muito frequente' | 'Frequente' | 'Raro'
        causa_principal      : 'Excesso de velocidade' | 'Falta de sinalização'
                               'Condução imprudente' | 'Consumo de álcool'
        conhece_ia           : 'Sim' | 'Não'
        apoia_modernizacao   : 'Sim' | 'Não'

    Retorna:
        dict com 'previsao' e 'confianca'
    """
    dados = {
        'Categoria': categoria,
        'Presenciou_Acidente': presenciou_acidente,
        'Frequencia_Acidentes': frequencia_acidentes,
        'Principal_Causa': causa_principal,
        'Conhece_IA': conhece_ia,
        'Apoia_Modernizacao_Transito': apoia_modernizacao,
    }
    row = []
    for col in FEATURES:
        val = encoders[col].transform([dados[col]])[0]
        row.append(val)

    proba  = best['model'].predict_proba([row])[0]
    pred   = best['model'].predict([row])[0]
    label  = encoders[TARGET].inverse_transform([pred])[0]
    confianca = max(proba)

    return {'previsao': label, 'confianca': f'{confianca:.1%}'}


# ─────────────────────────────────────────
# 8. DEMO DE PREVISÕES
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("  DEMO: PREVISÕES PARA NOVOS CIDADÃOS")
print("=" * 60)

casos_demo = [
    {
        'categoria': 'Mototaxista',
        'presenciou_acidente': 'Sim',
        'frequencia_acidentes': 'Muito frequente',
        'causa_principal': 'Excesso de velocidade',
        'conhece_ia': 'Sim',
        'apoia_modernizacao': 'Sim',
    },
    {
        'categoria': 'Peão',
        'presenciou_acidente': 'Não',
        'frequencia_acidentes': 'Raro',
        'causa_principal': 'Falta de sinalização',
        'conhece_ia': 'Não',
        'apoia_modernizacao': 'Sim',
    },
    {
        'categoria': 'Motorista',
        'presenciou_acidente': 'Sim',
        'frequencia_acidentes': 'Frequente',
        'causa_principal': 'Condução imprudente',
        'conhece_ia': 'Sim',
        'apoia_modernizacao': 'Não',
    },
]

for i, c in enumerate(casos_demo, 1):
    resultado = prever(**c)
    print(f"\n  Caso {i}: {c['categoria']} | "
          f"Conhece IA: {c['conhece_ia']} | "
          f"Frequência: {c['frequencia_acidentes']}")
    print(f"  → Previsão : {resultado['previsao']}")
    print(f"  → Confiança: {resultado['confianca']}")

print("\n" + "=" * 60)
print("  TREINAMENTO CONCLUÍDO COM SUCESSO!")
print(f"  Melhor Modelo : {best_name}")
print(f"  Acurácia      : {best['acc']:.1%}")
print(f"  AUC-ROC       : {best['auc']:.3f}")
print("  Ficheiros gerados:")
print("    • ml_dashboard_songo.png")
print("    • modelo_songo.pkl")
print("=" * 60)
