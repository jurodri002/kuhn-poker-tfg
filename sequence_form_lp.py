"""
Resolucio del joc de Kuhn per programacio lineal en forma de sequencia.

Construeix explicitament els conjunts de sequencies Q_A, Q_B, les matrius
de restriccio E, F que defineixen els politops Q_A^Delta, Q_B^Delta, i la
matriu de pagaments en forma de sequencia M. Resol el programa lineal (P)
amb scipy.optimize.linprog per obtenir el valor del joc i els plans de
realitzacio optims de tots dos jugadors.

Referencies:
  - Koller, Megiddo i von Stengel (1994), "Fast Algorithms for Finding
    Randomized Strategies in Game Trees".
  - von Stengel (1996), "Efficient Computation of Behavior Strategies",
    Games and Economic Behavior 14(2), 220-246.
"""

import numpy as np
from scipy.optimize import linprog

CARDS = [0, 1, 2]  # J, Q, K

# ------------------------------------------------------------------
# Sequencies
# ------------------------------------------------------------------
# Jugador A: sequencia = (carta, accions propies d'A, en ordre)
#   () -> arrel
#   (c,'p'), (c,'b') -> primera decisio amb la carta c
#   (c,'p','p'), (c,'p','b') -> segona decisio (nomes si A ha passat i B aposta)
#       'p' = retirar-se (Re), 'b' = igualar (Ig)
SEQ_A = ['()']
for c in CARDS:
    SEQ_A += [f"{c}p", f"{c}b"]
for c in CARDS:
    SEQ_A += [f"{c}pp", f"{c}pb"]
IDX_A = {s: i for i, s in enumerate(SEQ_A)}
N_A = len(SEQ_A)

# Jugador B: sequencia = (carta, accio d'A observada, accio propia de B)
#   () -> arrel
#   (c,'p','p'), (c,'p','b') -> resposta de B quan A ha passat
#   (c,'b','p'), (c,'b','b') -> resposta de B quan A ha apostat directament
SEQ_B = ['()']
for c in CARDS:
    SEQ_B += [f"{c}pp", f"{c}pb", f"{c}bp", f"{c}bb"]
IDX_B = {s: i for i, s in enumerate(SEQ_B)}
N_B = len(SEQ_B)


def build_E():
    """Matriu E i vector e tals que E pi_A = e defineixen Q_A^Delta (R1)+(R2)."""
    rows, rhs = [], []

    # (R1): pi_A(empty) = 1
    r = np.zeros(N_A)
    r[IDX_A['()']] = 1
    rows.append(r); rhs.append(1)

    # (R2) a la primera decisio de cada carta (sequencia pare = empty)
    for c in CARDS:
        r = np.zeros(N_A)
        r[IDX_A[f"{c}p"]] = 1
        r[IDX_A[f"{c}b"]] = 1
        r[IDX_A['()']] = -1
        rows.append(r); rhs.append(0)

    # (R2) a la segona decisio de cada carta (sequencia pare = (c,'p'))
    for c in CARDS:
        r = np.zeros(N_A)
        r[IDX_A[f"{c}pp"]] = 1
        r[IDX_A[f"{c}pb"]] = 1
        r[IDX_A[f"{c}p"]] = -1
        rows.append(r); rhs.append(0)

    return np.vstack(rows), np.array(rhs, dtype=float)


def build_F():
    """Matriu F i vector f tals que F pi_B = f defineixen Q_B^Delta (R1)+(R2)."""
    rows, rhs = [], []

    # (R1): pi_B(empty) = 1
    r = np.zeros(N_B)
    r[IDX_B['()']] = 1
    rows.append(r); rhs.append(1)

    # (R2) resposta de B despres que A hagi passat
    for c in CARDS:
        r = np.zeros(N_B)
        r[IDX_B[f"{c}pp"]] = 1
        r[IDX_B[f"{c}pb"]] = 1
        r[IDX_B['()']] = -1
        rows.append(r); rhs.append(0)

    # (R2) resposta de B despres que A hagi apostat directament
    for c in CARDS:
        r = np.zeros(N_B)
        r[IDX_B[f"{c}bp"]] = 1
        r[IDX_B[f"{c}bb"]] = 1
        r[IDX_B['()']] = -1
        rows.append(r); rhs.append(0)

    return np.vstack(rows), np.array(rhs, dtype=float)


def payoff_to_A(cA, cB, history):
    """Pagament a A a la fulla determinada per 'history' (des del punt de vista d'A)."""
    higher = cA > cB
    if history == "pp":
        return 1 if higher else -1
    if history == "bp":
        return 1
    if history == "bb":
        return 2 if higher else -2
    if history == "pbp":
        return -1
    if history == "pbb":
        return 2 if higher else -2
    raise ValueError(f"historia no reconeguda: {history}")


def build_M():
    """Matriu de pagaments en forma de sequencia M (files Q_A, columnes Q_B)."""
    M = np.zeros((N_A, N_B))
    deals = [(a, b) for a in CARDS for b in CARDS if a != b]
    p_deal = 1 / len(deals)  # cada un dels 6 repartiments es equiprobable

    for cA, cB in deals:
        # "pp": A passa, B passa (showdown senzill)
        qa, qb = IDX_A[f"{cA}p"], IDX_B[f"{cB}pp"]
        M[qa, qb] += p_deal * payoff_to_A(cA, cB, "pp")

        # "bp": A aposta, B es retira
        qa, qb = IDX_A[f"{cA}b"], IDX_B[f"{cB}bp"]
        M[qa, qb] += p_deal * payoff_to_A(cA, cB, "bp")

        # "bb": A aposta, B iguala (showdown doble)
        qa, qb = IDX_A[f"{cA}b"], IDX_B[f"{cB}bb"]
        M[qa, qb] += p_deal * payoff_to_A(cA, cB, "bb")

        # "pbp": A passa, B aposta, A es retira
        qa, qb = IDX_A[f"{cA}pp"], IDX_B[f"{cB}pb"]
        M[qa, qb] += p_deal * payoff_to_A(cA, cB, "pbp")

        # "pbb": A passa, B aposta, A iguala (showdown doble)
        qa, qb = IDX_A[f"{cA}pb"], IDX_B[f"{cB}pb"]
        M[qa, qb] += p_deal * payoff_to_A(cA, cB, "pbb")

    return M


def solve():
    """Resol el programa lineal (P) i retorna el valor del joc i els plans
    de realitzacio optims de tots dos jugadors."""
    E, e = build_E()
    F, f = build_F()
    M = build_M()

    n_lambda = len(f)  # una component per infoset de B, incloent l'arrel
    n_vars = N_A + n_lambda

    # Objectiu: maximitzar lambda(empty_B) == minimitzar -lambda(empty_B)
    c = np.zeros(n_vars)
    c[N_A + 0] = -1

    # Igualtat: E pi_A = e (columnes de lambda a zero)
    A_eq = np.zeros((E.shape[0], n_vars))
    A_eq[:, :N_A] = E
    b_eq = e

    # Desigualtat: F^T lambda <= M^T pi_A  <=>  -M^T pi_A + F^T lambda <= 0
    A_ub = np.zeros((N_B, n_vars))
    A_ub[:, :N_A] = -M.T
    A_ub[:, N_A:] = F.T
    b_ub = np.zeros(N_B)

    bounds = [(0, None)] * N_A + [(None, None)] * n_lambda

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs')
    if not res.success:
        raise RuntimeError(f"linprog no ha convergit: {res.message}")

    V = -res.fun
    pi_A = {s: res.x[IDX_A[s]] for s in SEQ_A}

    # Pla de realitzacio de B a partir de les variables duals de les
    # restriccions de desigualtat (marginals), amb el signe corresponent.
    pi_B = {s: -res.ineqlin.marginals[IDX_B[s]] for s in SEQ_B}

    return V, pi_A, pi_B


def realization_to_behavior(pi, parent_seq_key, actions_keys):
    """Converteix pesos d'un pla de realitzacio en probabilitats de
    comportament b_i(I)_a = pi(q.a) / pi(q), tal com defineix 
    l'Obsservació de realitzacio de comportament del TFG."""

    parent_weight = pi[parent_seq_key]

    if parent_weight <= 1e-12:
        return None

    return {
        a: pi[k] / parent_weight
        for a, k in zip(['p', 'b'], actions_keys)
    }


if __name__ == "__main__":
    V, pi_A, pi_B = solve()

    print(f"V* = {V:.6f}  (teoric -1/18 = {-1/18:.6f})")
    print()

    print("Pla de realitzacio de A (pi_A):")
    for s in SEQ_A:
        print(f"  {s:>6}: {pi_A[s]:.4f}")

    print()
    print("Pla de realitzacio de B (pi_B):")
    for s in SEQ_B:
        print(f"  {s:>6}: {pi_B[s]:.4f}")
    print()
    print("Estrategies de comportament de B:")

    for c in CARDS:
        b_after_p = realization_to_behavior( pi_B, "()", [f"{c}pp", f"{c}pb"])
        print(f"  B amb carta {c}, despres que A passi: {b_after_p}")

        b_after_b = realization_to_behavior(pi_B, "()",[f"{c}bp", f"{c}bb"])
        print( f"  B amb carta {c}, despres que A aposti: {b_after_b}")

    print()
    print("Estrategies de comportament derivades (b_i(I)_a = pi(q.a)/pi(q)):")
    for c in CARDS:
        b = realization_to_behavior(pi_A, '()', [f"{c}p", f"{c}b"])
        print(f"  A amb carta {c}, primera decisio: {b}")
    for c in CARDS:
        b = realization_to_behavior(pi_A, f"{c}p", [f"{c}pp", f"{c}pb"])
        print(f"  A amb carta {c}, segona decisio (si arriba): {b}")

    print()
    # alpha = b_B(Ap | J, Pa): normalitzem respecte la sequencia pare "()"
    alpha = pi_B["0pb"] / pi_B["()"]
    print(f"alpha = b_B(Ap | J, Pa) = {alpha:.4f}")
    beta = pi_A["1pb"]/pi_A["1p"]
    print(f"beta = b_A(Ig | Q, Ap) = {beta:-4f}")
    gamma = pi_A["2b"]
    print(f"gamma = b_A(Ap | K) = {gamma:-4f}")
