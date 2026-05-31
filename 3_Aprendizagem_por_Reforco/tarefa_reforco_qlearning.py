"""
=============================================================
GUIÃO PRÁTICO - CAPÍTULO III: APRENDIZAGEM AUTOMÁTICA (ML)
TAREFA FINAL: APRENDIZAGEM POR REFORÇO — Q-LEARNING
Universidade Kimpa Vita - Curso de Engenharia Informática
Disciplina: Inteligência Artificial II
=============================================================

OBJECTIVO DA TAREFA:
  Implementar um agente Q-Learning que aprende a navegar
  num labirinto simples (Grid World 5×5) encontrando o
  caminho óptimo desde o ponto inicial até à recompensa,
  evitando obstáculos (penalizações).

CONCEITOS APLICADOS (referência ao Capítulo III):
  - Agente: o robô que navega no ambiente
  - Estado (s): posição actual no grid (linha, coluna)
  - Acção (a): Cima, Baixo, Esquerda, Direita
  - Recompensa (r): +100 (chegou ao destino),
                    -10 (obstáculo), -1 (passo normal)
  - Política (π): tabela Q que o agente aprende
  - Q-Learning: actualiza Q(s,a) com a equação de Bellman:
    Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]

PARÂMETROS:
  α (alpha) = 0.1   — Taxa de aprendizagem (learning rate)
  γ (gamma) = 0.95  — Factor de desconto (long-term reward)
  ε (epsilon) = 1.0 → 0.01 — Exploração vs exploração
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

print("=" * 65)
print("  TAREFA: APRENDIZAGEM POR REFORÇO — Q-LEARNING")
print("  Grid World 5×5 — Navegação com Obstáculos")
print("=" * 65)

# ================================================================
# DEFINIÇÃO DO AMBIENTE (Grid World 5×5)
# ================================================================
# Mapa do labirinto:
#   S = Start (início)    | G = Goal (destino)
#   X = Obstáculo         | . = Livre
#
#   S . . X .
#   . X . . .
#   . . . X .
#   . X . . .
#   . . . . G

GRID_ROWS, GRID_COLS = 5, 5
START  = (0, 0)   # Início
GOAL   = (4, 4)   # Destino
OBSTACLES = [(0,3),(1,1),(2,3),(3,1)]  # Obstáculos

# Acções possíveis: 0=Cima, 1=Baixo, 2=Esquerda, 3=Direita
ACTIONS = {0:(-1,0), 1:(1,0), 2:(0,-1), 3:(0,1)}
ACTION_NAMES = {0:'↑', 1:'↓', 2:'←', 3:'→'}
N_ACTIONS = 4

# Recompensas
R_GOAL     = +100   # Chegou ao destino
R_OBSTACLE = -10    # Bateu num obstáculo
R_STEP     = -1     # Passo normal (incentiva caminhos curtos)

# Hiperparâmetros Q-Learning
ALPHA   = 0.1    # Taxa de aprendizagem
GAMMA   = 0.95   # Factor de desconto
EPSILON_START = 1.0
EPSILON_END   = 0.01
EPSILON_DECAY = 0.995
N_EPISODES = 500

def get_reward(state):
    """Retorna recompensa para o estado."""
    if state == GOAL:       return R_GOAL,     True
    if state in OBSTACLES:  return R_OBSTACLE, False
    return R_STEP, False

def step(state, action):
    """Executa acção e retorna novo estado."""
    dr, dc = ACTIONS[action]
    new_r  = max(0, min(GRID_ROWS-1, state[0]+dr))
    new_c  = max(0, min(GRID_COLS-1, state[1]+dc))
    new_state = (new_r, new_c)
    if new_state in OBSTACLES:
        return state, R_OBSTACLE, False  # Fica no lugar ao bater
    reward, done = get_reward(new_state)
    return new_state, reward, done

# ================================================================
# INICIALIZAÇÃO DA TABELA Q
# ================================================================
# Explicação: A tabela Q tem dimensões [estados × acções].
# Cada estado é uma posição (linha, coluna) → linha*cols + coluna.
# Inicializamos com zeros: o agente não sabe nada no início.

N_STATES = GRID_ROWS * GRID_COLS
Q_table  = np.zeros((GRID_ROWS, GRID_COLS, N_ACTIONS))
epsilon  = EPSILON_START

print(f"\n[AMBIENTE]")
print(f"  Grid: {GRID_ROWS}×{GRID_COLS} | Estados: {N_STATES} | Acções: {N_ACTIONS}")
print(f"  Início: {START} | Destino: {GOAL}")
print(f"  Obstáculos: {OBSTACLES}")
print(f"\n[HIPERPARÂMETROS]")
print(f"  α (learning rate): {ALPHA}")
print(f"  γ (discount):      {GAMMA}")
print(f"  ε inicial:         {EPSILON_START} → {EPSILON_END} (decay={EPSILON_DECAY})")
print(f"  Episódios:         {N_EPISODES}")

# ================================================================
# TREINO DO AGENTE Q-LEARNING
# ================================================================

print(f"\n[TREINO] A iniciar {N_EPISODES} episódios...")

rewards_per_ep  = []
steps_per_ep    = []
success_per_ep  = []

for episode in range(N_EPISODES):
    state   = START
    total_r = 0
    n_steps = 0
    done    = False

    while not done and n_steps < 100:  # Máximo 100 passos por episódio
        # Política ε-greedy:
        # Com probabilidade ε: exploração (acção aleatória)
        # Com probabilidade 1-ε: exploração (melhor acção da tabela Q)
        if np.random.rand() < epsilon:
            action = np.random.randint(N_ACTIONS)
        else:
            action = np.argmax(Q_table[state[0], state[1]])

        new_state, reward, done = step(state, action)
        total_r += reward
        n_steps += 1

        # Actualização Q-Learning (Equação de Bellman):
        # Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
        best_next = np.max(Q_table[new_state[0], new_state[1]])
        Q_table[state[0], state[1], action] += ALPHA * (
            reward + GAMMA * best_next - Q_table[state[0], state[1], action]
        )
        state = new_state

    # Decay do epsilon (mais exploração → mais exploração)
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    rewards_per_ep.append(total_r)
    steps_per_ep.append(n_steps)
    success_per_ep.append(1 if state == GOAL else 0)

    if (episode+1) % 100 == 0:
        recent = success_per_ep[-100:]
        print(f"  Episódio {episode+1:4d}: "
              f"Taxa sucesso (últimos 100) = {np.mean(recent)*100:.1f}% | "
              f"ε = {epsilon:.4f} | "
              f"Recompensa média = {np.mean(rewards_per_ep[-100:]):.1f}")

# ================================================================
# VISUALIZAÇÕES
# ================================================================

FIGDIR = "./figuras"
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Treino do Agente Q-Learning — Grid World 5×5", fontsize=13, fontweight='bold')

# 1. Recompensa por episódio (média móvel)
window = 20
smoothed = np.convolve(rewards_per_ep, np.ones(window)/window, mode='valid')
axes[0].plot(smoothed, color='#3498db', lw=2)
axes[0].set_title(f'Recompensa Total por Episódio\n(Média móvel, janela={window})')
axes[0].set_xlabel('Episódio')
axes[0].set_ylabel('Recompensa Total')
axes[0].grid(True, alpha=0.3)

# 2. Taxa de sucesso cumulativa
cumul_success = np.cumsum(success_per_ep) / (np.arange(len(success_per_ep))+1) * 100
axes[1].plot(cumul_success, color='#2ecc71', lw=2)
axes[1].axhline(y=90, color='red', linestyle='--', lw=1.5, label='90% alvo')
axes[1].set_title('Taxa de Sucesso Cumulativa')
axes[1].set_xlabel('Episódio')
axes[1].set_ylabel('Sucesso (%)')
axes[1].set_ylim(0, 105)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. Passos por episódio
smooth_steps = np.convolve(steps_per_ep, np.ones(window)/window, mode='valid')
axes[2].plot(smooth_steps, color='#e74c3c', lw=2)
axes[2].set_title(f'Passos por Episódio\n(Média móvel, janela={window})')
axes[2].set_xlabel('Episódio')
axes[2].set_ylabel('Número de Passos')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{FIGDIR}/3a_treinamento_qlearning.png", dpi=120, bbox_inches='tight')
plt.close()
print(f"\n  Figura salva: 3a_treinamento_qlearning.png")

# ================================================================
# VISUALIZAÇÃO DO LABIRINTO E POLÍTICA APRENDIDA
# ================================================================

# Extrair política óptima
policy = np.argmax(Q_table, axis=2)  # Melhor acção em cada estado

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Ambiente e Política Óptima Aprendida — Q-Learning", fontsize=13, fontweight='bold')

# Labirinto
grid_display = np.zeros((GRID_ROWS, GRID_COLS))
for obs in OBSTACLES: grid_display[obs] = -1
grid_display[GOAL] = 2

colors_map = {-1:'#e74c3c', 0:'#ecf0f1', 2:'#2ecc71'}
ax = axes[0]
ax.set_xlim(0, GRID_COLS)
ax.set_ylim(0, GRID_ROWS)
for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        val = grid_display[r, c]
        color = colors_map.get(val, '#ecf0f1')
        rect = plt.Rectangle([c, GRID_ROWS-1-r], 1, 1, color=color, ec='#bdc3c7', lw=1.5)
        ax.add_patch(rect)
        if (r, c) == START:
            ax.text(c+0.5, GRID_ROWS-0.5-r, 'S', ha='center', va='center', fontsize=14, fontweight='bold', color='#2980b9')
        elif (r, c) == GOAL:
            ax.text(c+0.5, GRID_ROWS-0.5-r, 'G', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        elif (r, c) in OBSTACLES:
            ax.text(c+0.5, GRID_ROWS-0.5-r, 'X', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
ax.set_xticks(range(GRID_COLS+1))
ax.set_yticks(range(GRID_ROWS+1))
ax.set_title('Mapa do Ambiente\n(S=Início, G=Destino, X=Obstáculo)')
legend_elements = [mpatches.Patch(color='#2ecc71', label='Destino (G)'),
                   mpatches.Patch(color='#e74c3c', label='Obstáculo (X)'),
                   mpatches.Patch(color='#ecf0f1', label='Livre')]
ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

# Política aprendida (setas)
ax2 = axes[1]
ax2.set_xlim(0, GRID_COLS)
ax2.set_ylim(0, GRID_ROWS)
for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        color = colors_map.get(grid_display[r,c], '#ecf0f1')
        rect = plt.Rectangle([c, GRID_ROWS-1-r], 1, 1, color=color, ec='#bdc3c7', lw=1.5)
        ax2.add_patch(rect)
        if (r, c) not in OBSTACLES and (r, c) != GOAL:
            a = policy[r, c]
            ax2.text(c+0.5, GRID_ROWS-0.5-r, ACTION_NAMES[a],
                    ha='center', va='center', fontsize=16, color='#2c3e50', fontweight='bold')
        elif (r,c) == GOAL:
            ax2.text(c+0.5, GRID_ROWS-0.5-r, '★', ha='center', va='center', fontsize=16, color='white')
ax2.set_xticks(range(GRID_COLS+1))
ax2.set_yticks(range(GRID_ROWS+1))
ax2.set_title('Política Óptima Aprendida\n(Acção em cada estado)')
plt.tight_layout()
plt.savefig(f"{FIGDIR}/3b_labirinto_politica.png", dpi=120, bbox_inches='tight')
plt.close()
print(f"  Figura salva: 3b_labirinto_politica.png")

# ================================================================
# DEMONSTRAÇÃO: PERCURSO ÓPTIMO
# ================================================================

print(f"\n[DEMONSTRAÇÃO] Percurso Óptimo do Agente Treinado:")
state = START
path = [state]
done = False
steps_demo = 0
while not done and steps_demo < 30:
    action = np.argmax(Q_table[state[0], state[1]])
    new_state, reward, done = step(state, action)
    path.append(new_state)
    print(f"  Passo {steps_demo+1}: {state} → {ACTION_NAMES[action]} → {new_state} (recompensa={reward})")
    state = new_state
    steps_demo += 1

if state == GOAL:
    print(f"\n  ★ DESTINO ALCANÇADO em {steps_demo} passos!")
else:
    print(f"\n  → Agente não alcançou destino (continuar treinando)")

# Estatísticas finais
final_success = np.mean(success_per_ep[-100:]) * 100
print(f"\n[ESTATÍSTICAS FINAIS]")
print(f"  Taxa de sucesso (últimos 100 ep.): {final_success:.1f}%")
print(f"  Passos médios (últimos 100 ep.):   {np.mean(steps_per_ep[-100:]):.1f}")
print(f"  Recompensa média (últimos 100 ep.): {np.mean(rewards_per_ep[-100:]):.1f}")

print("\n" + "=" * 65)
print("  ★ TAREFA DE APRENDIZAGEM POR REFORÇO CONCLUÍDA")
print("  O agente aprendeu a política óptima através de")
print("  tentativa e erro — sem rótulos, sem supervisor!")
print("=" * 65)
print("\n✓ Todas as figuras salvas em ../figuras/")
