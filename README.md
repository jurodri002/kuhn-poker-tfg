# Joc de Kuhn: resolució per programació lineal i CFR

Codi font associat al Capítol 5 (*Implementació: el joc de Kuhn*) i als annexos
corresponents del TFG *[títol del TFG]*.

Es resol el joc de Kuhn per dues vies independents:

1. **Programació lineal en forma de seqüència** (`src/sequence_form_lp.py`),
   seguint Koller, Megiddo i von Stengel (1994) i von Stengel (1996).
2. **Counterfactual Regret Minimization (CFR)** (`src/cfr_kuhn.py`), seguint
   Zinkevich, Johanson, Bowling i Piccione (2007), amb l'esquema d'implementació
   de Neller i Lanctot (2013).

## Instal·lació

```bash
python3 -m venv venv
source venv/bin/activate       # a Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Requereix Python ≥ 3.9.

## Ús

### Forma de seqüència (LP)

```bash
python3 src/sequence_form_lp.py
```

Construeix explícitament les seqüències $Q_A$, $Q_B$, les matrius de
restricció $E$, $F$ i la matriu de pagaments $M$, i resol el programa lineal
$(P)$ amb `scipy.optimize.linprog`. Triga menys d'un segon i dona resultats
**exactes** (no hi ha soroll de mostreig):

```
V* = -0.055556  (teoric -1/18 = -0.055556)
...
alpha = b_B(Ap | J, Pa) = 0.3333  (teoric 1/3 = 0.3333)
```

### CFR (autoaprenentatge)

```bash
python3 src/cfr_kuhn.py
```

Per defecte entrena amb checkpoints fins a $10^5$ iteracions (uns 10 segons),
imprimint $\hat V$, $\alpha$, $\beta$, $\gamma$ a cada checkpoint.

**Important sobre el temps d'execució**: la implementació és recursiva pura en
Python (sense vectorització), i fa aproximadament $10^4$ iteracions per segon.
La Taula de convergència completa del TFG arriba fins a $10^8$ iteracions, cosa
que amb aquesta implementació trigaria **unes 2-3 hores**. Per reproduir-la
sencera:

```python
from src.cfr_kuhn import run_with_checkpoints
results = run_with_checkpoints(checkpoints=[10**2, 10**3, 10**4, 10**5, 10**6, 10**7, 10**8])
```

La llavor `np.random.seed(2026)` es fixa dins de `run_with_checkpoints` per
garantir la reproductibilitat exacta dels resultats reportats al TFG.

## Estructura del repositori

```
.
├── README.md
├── requirements.txt
├── LICENSE
└── src/
    ├── sequence_form_lp.py   # Resolució per LP en forma de seqüència
    └── cfr_kuhn.py           # Resolució per CFR (autoaprenentatge)
```

## Resultats esperats

| Mètode | $\hat V$ | $\alpha$ | $\beta$ | $\gamma$ |
|---|---|---|---|---|
| LP (exacte) | $-0{,}05556$ | $0{,}3333$ | lliure* | lliure* |
| CFR ($10^8$ it.) | $-0{,}0556$ | $0{,}333$ | oscil·la | oscil·la |

\*$\beta$ i $\gamma$ no tenen valor teòric únic, són punts d'indiferència de
l'equilibri un cop $\alpha=\tfrac13$ (veure Secció "Per què CFR convergeix a un
equilibri diferent" de l'Annex del TFG). El LP en troba un vèrtex concret del
conjunt continu d'equilibris; CFR mai s'hi estabilitza perquè no rep cap
senyal d'aprenentatge en aquests nodes.

## Referències

- Koller, D., Megiddo, N., von Stengel, B. (1994). *Fast Algorithms for
  Finding Randomized Strategies in Game Trees*. STOC '94.
- von Stengel, B. (1996). *Efficient Computation of Behavior Strategies*.
  Games and Economic Behavior, 14(2), 220-246.
- Zinkevich, M., Johanson, M., Bowling, M., Piccione, C. (2007). *Regret
  Minimization in Games with Incomplete Information*. NeurIPS 20.
- Neller, T. W., Lanctot, M. (2013). *An Introduction to Counterfactual
  Regret Minimization*. Model AI Assignments, EAAI-13.
- Kuhn, H. W. (1950). *A Simplified Two-Person Poker*. Contributions to the
  Theory of Games, 1, 97-103.

## Llicència

MIT (veure `LICENSE`).
