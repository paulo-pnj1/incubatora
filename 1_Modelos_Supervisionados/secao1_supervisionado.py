"""
=============================================================
GUIÃO PRÁTICO - CAPÍTULO III: APRENDIZAGEM AUTOMÁTICA (ML)
SECÇÃO 1: MODELOS SUPERVISIONADOS
Universidade Kimpa Vita - Curso de Engenharia Informática
Disciplina: Inteligência Artificial II
Autor do Guião: Prof. [Nome]
=============================================================

OBJECTIVOS:
  - Treinar modelos de classificação: KNN, Árvore de Decisão,
    Random Forest, SVM e Floresta Aleatória
  - Treinar modelo de regressão linear
  - Avaliar modelos com métricas: Accuracy, Precision, Recall,
    F1-Score, AUC-ROC, Curva ROC, Matriz de Confusão
  - Seleccionar o melhor modelo com base em critérios científicos

DATASET: credito_bancario.csv (simulado)
  - Variáveis: Idade, Pontuacao_Credito, Rendimento_Mensal,
               Anos_Historico, Divida_Total
  - Target: Aprovado (1 = Sim, 0 = Não)
"""

# ================================================================
# PASSO 0: IMPORTAR BIBLIOTECAS
# ================================================================
# Explicação: Importamos todas as ferramentas necessárias.
# O scikit-learn é a biblioteca padrão de ML em Python,
# amplamente usada na academia e indústria.

import pandas as pd           # Manipulação de dados tabulares
import numpy as np            # Operações numéricas
import matplotlib.pyplot as plt  # Visualizações
import seaborn as sns         # Gráficos estatísticos avançados
import os, warnings
warnings.filterwarnings('ignore')

# Modelos de ML
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, LogisticRegression

# Pré-processamento
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

# Métricas de avaliação
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

# Configurações de visualização
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
FIGDIR = "./figuras"
os.makedirs(FIGDIR, exist_ok=True)

print("=" * 65)
print("  SECÇÃO 1 — MODELOS SUPERVISIONADOS")
print("  Capítulo III: Aprendizagem Automática")
print("=" * 65)

# ================================================================
# PASSO 1: CARREGAR E EXPLORAR OS DADOS
# ================================================================
# Explicação: Antes de qualquer modelo, o cientista de dados deve
# compreender os dados (EDA - Análise Exploratória de Dados).
# Mitchell (1997) define AA como aprendizagem de experiência E:
# os dados são essa experiência.

print("\n[PASSO 1] Carregamento e Exploração dos Dados")
print("-" * 50)

df = pd.read_csv("./datasets/credito_bancario.csv")
print(f"→ Dimensão do dataset: {df.shape[0]} amostras × {df.shape[1]} variáveis")
print(f"\n→ Primeiras 5 linhas:")
print(df.head())

print(f"\n→ Estatísticas descritivas:")
print(df.describe().round(2))

print(f"\n→ Valores nulos por coluna:")
print(df.isnull().sum())

print(f"\n→ Distribuição da variável-alvo (Aprovado):")
vc = df['Aprovado'].value_counts()
print(f"   Recusado (0): {vc[0]} amostras ({vc[0]/len(df)*100:.1f}%)")
print(f"   Aprovado (1): {vc[1]} amostras ({vc[1]/len(df)*100:.1f}%)")

# Visualização da distribuição
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Distribuição das Variáveis — Dataset Crédito Bancário", fontsize=14, fontweight='bold')
cols = ['Idade', 'Pontuacao_Credito', 'Rendimento_Mensal', 'Anos_Historico', 'Divida_Total']
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
for i, (col, color) in enumerate(zip(cols, colors)):
    ax = axes[i//3][i%3]
    ax.hist(df[col], bins=25, color=color, edgecolor='white', alpha=0.8)
    ax.set_title(col, fontsize=10, fontweight='bold')
    ax.set_xlabel('Valor')
    ax.set_ylabel('Frequência')
axes[1][2].axis('off')
plt.tight_layout()
plt.savefig(f"{FIGDIR}/1a_distribuicao_variaveis.png", dpi=120, bbox_inches='tight')
plt.close()
print(f"\n→ Figura salva: 1a_distribuicao_variaveis.png")

# ================================================================
# PASSO 2: PRÉ-PROCESSAMENTO DOS DADOS
# ================================================================
# Explicação: Os dados brutos raramente estão prontos para treinar
# modelos. O pré-processamento inclui:
#   - Selecção de features (X) e target (y)
#   - Normalização (StandardScaler) — essencial para KNN e SVM,
#     pois esses algoritmos são sensíveis à escala das variáveis
#   - Divisão treino/teste (70%/30%) com aleatoriedade garantida
#     pelo parâmetro random_state — reproducibilidade é fundamental
#     em ciência!

print("\n[PASSO 2] Pré-Processamento dos Dados")
print("-" * 50)

X = df.drop('Aprovado', axis=1)  # Features (variáveis independentes)
y = df['Aprovado']               # Target (variável dependente/rótulo)

print(f"→ Features (X): {list(X.columns)}")
print(f"→ Target (y): Aprovado (binário: 0/1)")

# Divisão treino (70%) e teste (30%)
# Estratificação (stratify=y) garante mesma proporção de classes
# em treino e teste — boa prática académica
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print(f"\n→ Divisão dos dados:")
print(f"   Treino: {X_train.shape[0]} amostras ({X_train.shape[0]/len(df)*100:.0f}%)")
print(f"   Teste:  {X_test.shape[0]} amostras ({X_test.shape[0]/len(df)*100:.0f}%)")

# Normalização com StandardScaler (z-score: µ=0, σ=1)
# IMPORTANTE: fit apenas no treino para evitar data leakage!
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # Aprende e transforma
X_test_sc  = scaler.transform(X_test)        # Só transforma (sem re-aprender)
print(f"\n→ Normalização aplicada (StandardScaler)")
print(f"   Média (treino): {X_train_sc.mean(axis=0).round(3)}")
print(f"   Desvio padrão:  {X_train_sc.std(axis=0).round(3)}")

# ================================================================
# PASSO 3: TREINAR E AVALIAR MODELOS DE CLASSIFICAÇÃO
# ================================================================

print("\n[PASSO 3] Treinamento dos Modelos de Classificação")
print("-" * 50)

# ----------------------------------------------------------------
# MODELO 1: KNN (K-Vizinhos Mais Próximos)
# ----------------------------------------------------------------
# Explicação: KNN é um algoritmo baseado em instâncias que classifica
# uma nova amostra pela maioria dos K vizinhos mais próximos no espaço
# de features. Usa distância euclidiana por padrão.
# Escolha de K=5: compromisso entre bias e variância — K muito pequeno
# causa overfitting, K muito grande causa underfitting.
# ATENÇÃO: KNN requer normalização — sem ela, variáveis de escala maior
# (ex: Rendimento_Mensal) dominariam a distância.

print("\n  ► Modelo 1: KNN (K=5)")
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train_sc, y_train)  # Treina com dados normalizados
y_pred_knn = knn.predict(X_test_sc)
y_prob_knn = knn.predict_proba(X_test_sc)[:, 1]
print(f"    Acurácia no treino: {knn.score(X_train_sc, y_train):.4f}")
print(f"    Acurácia no teste:  {accuracy_score(y_test, y_pred_knn):.4f}")

# ----------------------------------------------------------------
# MODELO 2: ÁRVORE DE DECISÃO
# ----------------------------------------------------------------
# Explicação: Uma Árvore de Decisão divide o espaço de features
# recursivamente com base em perguntas (splits). O critério 'gini'
# mede a impureza dos nós — mínimo impuro = melhor divisão.
# max_depth=5 limita a profundidade para evitar overfitting.
# Vantagem: interpretável (podemos visualizar a árvore).
# Desvantagem: instável — pequenas mudanças nos dados alteram muito
# a árvore (alta variância).

print("\n  ► Modelo 2: Árvore de Decisão (max_depth=5)")
dt = DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=42)
dt.fit(X_train, y_train)      # Árvore NÃO precisa de normalização
y_pred_dt = dt.predict(X_test)
y_prob_dt = dt.predict_proba(X_test)[:, 1]
print(f"    Acurácia no treino: {dt.score(X_train, y_train):.4f}")
print(f"    Acurácia no teste:  {accuracy_score(y_test, y_pred_dt):.4f}")

# Visualização da Árvore de Decisão
fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(dt, feature_names=X.columns.tolist(), class_names=['Recusado','Aprovado'],
          filled=True, rounded=True, fontsize=8, ax=ax, max_depth=3)
ax.set_title("Árvore de Decisão — Aprovação de Crédito (max_depth=3 para visibilidade)", fontsize=12)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/1b_arvore_decisao.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"    Figura salva: 1b_arvore_decisao.png")

# ----------------------------------------------------------------
# MODELO 3: RANDOM FOREST (FLORESTA ALEATÓRIA)
# ----------------------------------------------------------------
# Explicação: Random Forest é um ensemble de múltiplas Árvores de
# Decisão. Cada árvore é treinada num subconjunto aleatório dos dados
# (bootstrap) e com um subconjunto aleatório das features.
# A predição final é a votação maioritária das árvores.
# n_estimators=100: 100 árvores — mais árvores = mais estável,
# mas mais lento. 100 é um bom padrão.
# Random Forest resolve o principal problema da Árvore de Decisão:
# alta variância — o conjunto é mais robusto que árvores individuais.

print("\n  ► Modelo 3: Random Forest (n_estimators=100)")
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
print(f"    Acurácia no treino: {rf.score(X_train, y_train):.4f}")
print(f"    Acurácia no teste:  {accuracy_score(y_test, y_pred_rf):.4f}")

# Importância das Features no Random Forest
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
feat_imp.plot(kind='bar', color=['#3498db','#e74c3c','#2ecc71','#f39c12','#9b59b6'], ax=ax)
ax.set_title("Importância das Features — Random Forest", fontsize=12, fontweight='bold')
ax.set_ylabel("Importância (Gini)")
ax.set_xlabel("Feature")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/1c_importancia_features_rf.png", dpi=120, bbox_inches='tight')
plt.close()
print(f"    Figura salva: 1c_importancia_features_rf.png")

# ----------------------------------------------------------------
# MODELO 4: SVM (MÁQUINA DE VECTORES DE SUPORTE)
# ----------------------------------------------------------------
# Explicação: O SVM encontra o hiperplano que maximiza a margem
# entre as classes. O kernel='rbf' (Radial Basis Function) permite
# separação não linear — equivalente ao "kernel trick" descrito no
# capítulo III: projecta os dados para espaço de maior dimensão.
# C=1.0: parâmetro de regularização — controla o trade-off entre
# margem máxima e erros de classificação.
# probability=True: necessário para calcular AUC-ROC.
# SVM EXIGE normalização — muito sensível à escala dos dados.

print("\n  ► Modelo 4: SVM (kernel=rbf, C=1.0)")
svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
svm.fit(X_train_sc, y_train)
y_pred_svm = svm.predict(X_test_sc)
y_prob_svm = svm.predict_proba(X_test_sc)[:, 1]
print(f"    Acurácia no treino: {svm.score(X_train_sc, y_train):.4f}")
print(f"    Acurácia no teste:  {accuracy_score(y_test, y_pred_svm):.4f}")

# ================================================================
# PASSO 4: AVALIAR E COMPARAR TODOS OS MODELOS
# ================================================================
# Explicação: Um bom cientista de dados não se baseia apenas
# na acurácia. Quando as classes são desequilibradas (ex: 60% recusado,
# 40% aprovado), a acurácia pode ser enganosa. Usamos:
#   - Precision: dos que previmos como positivo, quantos eram realmente?
#   - Recall: dos verdadeiros positivos, quantos detectámos?
#   - F1-Score: média harmónica de Precision e Recall
#   - AUC-ROC: área sob a curva ROC — mede capacidade discriminativa

print("\n[PASSO 4] Comparação Completa dos Modelos")
print("=" * 65)

modelos = {
    "KNN": (y_pred_knn, y_prob_knn),
    "Árvore de Decisão": (y_pred_dt, y_prob_dt),
    "Random Forest": (y_pred_rf, y_prob_rf),
    "SVM": (y_pred_svm, y_prob_svm),
}

resultados = {}
for nome, (y_pred, y_prob) in modelos.items():
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)
    resultados[nome] = {'Acurácia':acc, 'Precisão':prec, 'Recall':rec, 'F1-Score':f1, 'AUC-ROC':auc}
    print(f"\n  ▶ {nome}")
    print(f"     Acurácia: {acc:.4f}  |  Precisão: {prec:.4f}  |  Recall: {rec:.4f}")
    print(f"     F1-Score: {f1:.4f}  |  AUC-ROC:  {auc:.4f}")

df_resultados = pd.DataFrame(resultados).T.round(4)
print(f"\n{'='*65}")
print("  TABELA COMPARATIVA DE MÉTRICAS:")
print(df_resultados.to_string())
melhor = df_resultados['AUC-ROC'].idxmax()
print(f"\n  ★ MELHOR MODELO (por AUC-ROC): {melhor} ({df_resultados.loc[melhor,'AUC-ROC']:.4f})")
print(f"{'='*65}")

# ================================================================
# PASSO 5: MATRIZES DE CONFUSÃO
# ================================================================
# Explicação: A Matriz de Confusão mostra:
#   VP (Verdadeiros Positivos): correctamente classificados como 1
#   VN (Verdadeiros Negativos): correctamente classificados como 0
#   FP (Falsos Positivos): incorrecto, previu 1, era 0
#   FN (Falsos Negativos): incorrecto, previu 0, era 1
# Em crédito bancário, FN (aprovado negado) tem custo diferente
# de FP (recusado aprovado) — contexto importa!

print("\n[PASSO 5] Matrizes de Confusão")
print("-" * 50)

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Matrizes de Confusão — Todos os Modelos", fontsize=14, fontweight='bold')
for ax, (nome, (y_pred, _)) in zip(axes.flat, modelos.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Recusado','Aprovado'],
                yticklabels=['Recusado','Aprovado'],
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(f'{nome}\nAcurácia: {accuracy_score(y_test,y_pred):.3f}', fontsize=11)
    ax.set_ylabel('Real', fontsize=10)
    ax.set_xlabel('Previsto', fontsize=10)
    vp = cm[1,1]; vn = cm[0,0]; fp = cm[0,1]; fn = cm[1,0]
    print(f"  {nome}: VP={vp}, VN={vn}, FP={fp}, FN={fn}")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/1d_matrizes_confusao.png", dpi=120, bbox_inches='tight')
plt.close()
print(f"\n  Figuras salvas: 1d_matrizes_confusao.png")

# ================================================================
# PASSO 6: CURVAS ROC E AUC
# ================================================================
# Explicação: A curva ROC (Receiver Operating Characteristic)
# plota TPR vs FPR para diferentes limiares de decisão.
# O ponto ideal é (0, 1) — 0% falsos positivos, 100% verdadeiros.
# A diagonal representa um classificador aleatório (AUC=0.5).
# Quanto mais a curva se aproxima do canto superior esquerdo,
# melhor o modelo.

print("\n[PASSO 6] Curvas ROC")
print("-" * 50)

fig, ax = plt.subplots(figsize=(9, 7))
cores = {'KNN':'#3498db', 'Árvore de Decisão':'#e74c3c',
         'Random Forest':'#2ecc71', 'SVM':'#9b59b6'}
for nome, (_, y_prob) in modelos.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, color=cores[nome], lw=2,
            label=f'{nome} (AUC = {auc_val:.3f})')
    print(f"  {nome}: AUC = {auc_val:.4f}")
ax.plot([0,1],[0,1],'k--', lw=1, label='Classificador Aleatório (AUC=0.5)')
ax.fill_between([0,1],[0,1],[0,1], alpha=0.05, color='grey')
ax.set_xlabel('Taxa de Falsos Positivos (FPR)', fontsize=12)
ax.set_ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=12)
ax.set_title('Curvas ROC — Comparação de Modelos\n(Ponto ideal: canto superior esquerdo)', fontsize=12)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/1e_curvas_roc.png", dpi=120, bbox_inches='tight')
plt.close()
print(f"\n  Figura salva: 1e_curvas_roc.png")

# ================================================================
# PASSO 7: VALIDAÇÃO CRUZADA (K-FOLD)
# ================================================================
# Explicação: A validação cruzada k-fold divide os dados em K grupos.
# Em cada iteração, K-1 grupos são usados para treino e 1 para validação.
# Isto dá uma estimativa mais robusta do desempenho real do modelo,
# reduzindo o impacto da aleatoriedade na divisão treino/teste.
# Usamos StratifiedKFold para manter proporção das classes.

print("\n[PASSO 7] Validação Cruzada (K-Fold, K=5)")
print("-" * 50)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
modelos_cv = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Árvore de Decisão': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    'SVM': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
}

cv_results = {}
for nome, modelo in modelos_cv.items():
    # Para KNN e SVM, usamos dados normalizados
    X_use = X_train_sc if nome in ['KNN', 'SVM'] else X_train.values
    scores = cross_val_score(modelo, X_use, y_train, cv=skf, scoring='roc_auc')
    cv_results[nome] = scores
    print(f"  {nome}: AUC médio = {scores.mean():.4f} ± {scores.std():.4f}")

# Boxplot da validação cruzada
fig, ax = plt.subplots(figsize=(9, 5))
ax.boxplot(cv_results.values(), labels=cv_results.keys(), patch_artist=True,
           boxprops=dict(facecolor='#3498db', alpha=0.6),
           medianprops=dict(color='red', linewidth=2))
ax.set_title('Validação Cruzada (5-Fold) — Distribuição do AUC', fontsize=12)
ax.set_ylabel('AUC-ROC')
ax.set_xlabel('Modelo')
ax.grid(True, axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/1f_validacao_cruzada.png", dpi=120, bbox_inches='tight')
plt.close()
print(f"\n  Figura salva: 1f_validacao_cruzada.png")

# ================================================================
# PASSO 8: REGRESSÃO LINEAR — PREVISÃO DE PREÇO DE IMÓVEL
# ================================================================
# Explicação: Regressão Linear modela a relação entre features e
# uma variável contínua (preço). A equação:
#   y = β0 + β1*x1 + β2*x2 + ... + βn*xn
# O modelo encontra os coeficientes β que minimizam o MSE
# (erro quadrático médio) entre valores reais e previstos.
# MSE = (1/n) Σ(yi - ŷi)²
# R² mede a proporção da variância explicada pelo modelo (0 a 1).

print("\n[PASSO 8] Regressão Linear — Previsão de Preços de Imóveis")
print("-" * 50)

df_reg = pd.read_csv("../datasets/precos_imoveis.csv")
print(f"→ Dataset de Imóveis: {df_reg.shape[0]} amostras × {df_reg.shape[1]} colunas")

X_reg = df_reg.drop('Preco_USD', axis=1)
y_reg = df_reg['Preco_USD']

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

scaler_reg = StandardScaler()
X_reg_train_sc = scaler_reg.fit_transform(X_reg_train)
X_reg_test_sc  = scaler_reg.transform(X_reg_test)

lr = LinearRegression()
lr.fit(X_reg_train_sc, y_reg_train)
y_reg_pred = lr.predict(X_reg_test_sc)

mse  = mean_squared_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_reg_test, y_reg_pred)
r2   = r2_score(y_reg_test, y_reg_pred)

print(f"\n  Coeficientes do modelo:")
for feat, coef in zip(X_reg.columns, lr.coef_):
    print(f"    {feat}: {coef:.4f}")
print(f"  Intercept (β0): {lr.intercept_:.4f}")
print(f"\n  Métricas de Avaliação (Regressão):")
print(f"    MAE  (Erro Absoluto Médio):        ${mae:,.2f}")
print(f"    RMSE (Raiz do Erro Quadrático):    ${rmse:,.2f}")
print(f"    R²   (Coeficiente de Determinação): {r2:.4f}")

# Gráfico Real vs Previsto
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y_reg_test, y_reg_pred, alpha=0.6, color='#3498db', edgecolors='white', s=50)
axes[0].plot([y_reg_test.min(), y_reg_test.max()],
             [y_reg_test.min(), y_reg_test.max()], 'r--', lw=2, label='Linha perfeita')
axes[0].set_xlabel('Preço Real (USD)', fontsize=11)
axes[0].set_ylabel('Preço Previsto (USD)', fontsize=11)
axes[0].set_title(f'Regressão Linear: Real vs Previsto\nR² = {r2:.4f}', fontsize=11)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

residuos = y_reg_test - y_reg_pred
axes[1].hist(residuos, bins=25, color='#e74c3c', edgecolor='white', alpha=0.8)
axes[1].axvline(0, color='black', linestyle='--', lw=2)
axes[1].set_xlabel('Resíduo (Real − Previsto)', fontsize=11)
axes[1].set_ylabel('Frequência', fontsize=11)
axes[1].set_title('Distribuição dos Resíduos\n(Ideal: centrada em 0)', fontsize=11)
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/1g_regressao_linear.png", dpi=120, bbox_inches='tight')
plt.close()
print(f"\n  Figura salva: 1g_regressao_linear.png")

# ================================================================
# PASSO 9: RELATÓRIO FINAL
# ================================================================

print("\n" + "=" * 65)
print("  RELATÓRIO FINAL — SECÇÃO 1")
print("=" * 65)
print(f"\n  Dataset: Aprovação de Crédito Bancário")
print(f"  Amostras: {len(df)} | Features: {X.shape[1]} | Classes: 2")
print()
print(df_resultados.to_string())
print(f"\n  ★ CONCLUSÃO:")
print(f"  O modelo {melhor} apresentou o melhor desempenho global")
print(f"  (AUC-ROC = {df_resultados.loc[melhor,'AUC-ROC']:.4f}), indicando maior capacidade")
print(f"  de discriminar entre clientes aprovados e recusados.")
print(f"\n  REGRESSÃO LINEAR (Imóveis):")
print(f"  R² = {r2:.4f} — o modelo explica {r2*100:.1f}% da variância do preço.")
print(f"  RMSE = ${rmse:,.0f} — erro típico de previsão.")
print("=" * 65)
print("\n✓ Secção 1 concluída. Todas as figuras salvas em ../figuras/")
