"""
Counterfactual Regret Minimization (CFR) per al joc de Kuhn.

Implementa l'algorisme de Zinkevich, Johanson, Bowling i Piccione (2007)
seguint l'esquema estandard de Neller i Lanctot (2013), amb checkpoints
a diferents nombres d'iteracions per reproduir la Taula de convergencia
de l'Annex del TFG.

Referencies:
  - Zinkevich et al. (2007), "Regret Minimization in Games with
    Incomplete Information", NeurIPS 20.
  - Neller i Lanctot (2013), "An Introduction to Counterfactual Regret
    Minimization", Model AI Assignments, EAAI-13.
"""

import numpy as np

CARDS = [0, 1, 2]  # J, Q, K
ACTIONS = ['p', 'b']  # passar/apostar a la primera decisio, retirar-se/igualar a la resposta

# Checkpoints (nombre acumulat d'iteracions) per defecte.
# NOTA: la Taula de convergencia del TFG arriba fins a 10^8 iteracions, pero
# aquesta implementacio recursiva pura en Python fa aprox. 10^4 iteracions/s,
# de manera que 10^8 iteracions trigarien unes 2-3 hores. Per defecte nomes
# arribem a 10^5 (uns 10 segons). Per reproduir la taula completa del TFG,
# crideu run_with_checkpoints(checkpoints=[10**2,...,10**8]) (veure README.md,
# temps estimat: unes 2-3 hores per arribar a 10^8).
CHECKPOINTS = [10**2, 10**3, 10**4, 10**5]

# Nodes rellevants per llegir alpha, beta, gamma (Taula de notacio del TFG)
NODE_ALPHA = "0p"    # b_B(Ap | J, Pa)
NODE_BETA = "1pb"    # b_A(Ig | Q, Ap)
NODE_GAMMA = "2"     # b_A(Ap | K)


class Node:
    """Manté el regret acumulat i l'estrategia acumulada d'un conjunt d'informacio."""

    def __init__(self):
        self.regret_sum = np.zeros(2)
        self.strategy_sum = np.zeros(2)

    def get_strategy(self, weight):
        pos = np.maximum(self.regret_sum, 0)
        s = pos.sum()
        if s > 0:
            strat = pos / s
        else:
            strat = np.array([0.5, 0.5])
        self.strategy_sum += weight * strat
        return strat

    def get_average_strategy(self):
        s = self.strategy_sum.sum()
        if s > 0:
            return self.strategy_sum / s
        else:
            return np.array([0.5, 0.5])


node_map = {}


def get_node(info_set):
    if info_set not in node_map:
        node_map[info_set] = Node()
    return node_map[info_set]


def terminal_value(cards, history, player):
    """Utilitat del jugador 'player' a una fulla del joc."""
    opponent = 1 - player
    higher = cards[player] > cards[opponent]
    if history[-1] == 'p':
        if history == "pp":
            return 1 if higher else -1
        else:
            return 1
    else:
        return 2 if higher else -2


def cfr(cards, history, p0, p1):
    """Recorregut recursiu de l'arbre, actualitzant regrets a cada info-set visitat."""
    plays = len(history)
    player = plays % 2
    if plays > 1 and (history[-1] == 'p' or history[-2:] == 'bb'):
        return terminal_value(cards, history, player)

    info_set = f"{cards[player]}{history}"
    node = get_node(info_set)
    strategy = node.get_strategy(p0 if player == 0 else p1)

    util = np.zeros(2)
    node_util = 0.0
    for i, a in enumerate(ACTIONS):
        next_history = history + a
        if player == 0:
            util[i] = -cfr(cards, next_history, p0 * strategy[i], p1)
        else:
            util[i] = -cfr(cards, next_history, p0, p1 * strategy[i])
        node_util += strategy[i] * util[i]

    for i in range(2):
        regret = util[i] - node_util
        if player == 0:
            node.regret_sum[i] += p1 * regret
        else:
            node.regret_sum[i] += p0 * regret
    return node_util


def train(iterations, cards=None):
    """Entrena 'iterations' partides addicionals (repartiment aleatori a cada una)."""
    if cards is None:
        cards = [0, 1, 2]
    for _ in range(iterations):
        np.random.shuffle(cards)
        cfr(cards, "", 1, 1)


def exact_game_value():
    """Valor exacte del joc a partir de les estrategies MITJANES ja apreses,
    recorrent els sis repartiments possibles (sense soroll de mostreig)."""

    def value(cards, history, player0):
        plays = len(history)
        player = plays % 2
        if plays > 1 and (history[-1] == 'p' or history[-2:] == 'bb'):
            v = terminal_value(cards, history, player)
            return v if player == player0 else -v

        info_set = f"{cards[player]}{history}"
        strat = get_node(info_set).get_average_strategy()
        total = 0.0
        for i, a in enumerate(ACTIONS):
            total += strat[i] * value(cards, history + a, player0)
        return total

    deals = [(a, b) for a in CARDS for b in CARDS if a != b]
    return sum(value([cA, cB], "", 0) for cA, cB in deals) / len(deals)


def read_frequencies():
    """Llegeix alpha, beta, gamma dels nodes rellevants (Taula de notacio del TFG)."""
    alpha = get_node(NODE_ALPHA).get_average_strategy()[1]
    beta = get_node(NODE_BETA).get_average_strategy()[1]
    gamma = get_node(NODE_GAMMA).get_average_strategy()[1]
    return alpha, beta, gamma


def run_with_checkpoints(checkpoints=CHECKPOINTS, seed=2026):
    """Entrena per blocs, imprimint (V_hat, alpha, beta, gamma) a cada checkpoint.
    Reprodueix la Taula de convergencia de l'Annex del TFG."""
    np.random.seed(seed)  # per reproductibilitat exacta dels resultats
    node_map.clear()

    done = 0
    results = []
    for target in checkpoints:
        train(target - done)
        done = target
        V_hat = exact_game_value()
        alpha, beta, gamma = read_frequencies()
        results.append((target, V_hat, alpha, beta, gamma))
    return results


if __name__ == "__main__":
    results = run_with_checkpoints()

    print(f"{'Iteracions':>12} {'V_hat':>10} {'alpha':>8} {'beta':>8} {'gamma':>8}")
    for target, V_hat, alpha, beta, gamma in results:
        print(f"{target:>12} {V_hat:>10.5f} {alpha:>8.4f} {beta:>8.4f} {gamma:>8.4f}")

    print()
    print(f"Valor teoric del joc: V = -1/18 = {-1/18:.5f}")
    print(f"alpha teoric: 1/3 = {1/3:.4f} (unic)")
    print("beta i gamma NO tenen valor teoric unic (punts d'indiferencia,")
    print("veure Seccio 'Per que CFR convergeix a un equilibri diferent' del TFG)")

    print()
    print("Estrategies mitjanes per a tots els conjunts d'informacio:")
    for info_set in sorted(node_map):
        print(" ", info_set, node_map[info_set].get_average_strategy())
