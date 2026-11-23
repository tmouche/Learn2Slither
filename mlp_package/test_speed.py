"""
Benchmark du package mlp : vitesse (forward, pas d'entrainement) et qualite
d'apprentissage (regression sur un reseau "professeur" fixe).

Usage, depuis mlp_package/ (pour benchmarker les sources, pas la version installee) :
    python test_speed.py                    # tout
    python test_speed.py --quick            # version rapide
    python test_speed.py --only speed       # seulement la vitesse
    python test_speed.py --only learn       # seulement l'apprentissage
    python test_speed.py --save avant.json  # sauvegarde les resultats
    python test_speed.py --compare avant.json  # compare a un run precedent
    python test_speed.py --check            # verifie que rien n'est casse (code de sortie 1 sinon)
"""
import argparse
import json
import sys
import time
from statistics import median
from typing import Callable, Dict, List, Tuple

import numpy as np

import mlp
from mlp.core.layer import Layer
from mlp.core.network import Network
from mlp.ml_tools import initialisations
from mlp.ml_tools.optimizers import ADAM, GradientDescent, RMSPropagation

# (taille, activation, initializer) ; la premiere couche est l'entree
ARCHS: Dict[str, List[Tuple[int, str | None, str | None]]] = {
    "12-32-4 (DQL)": [(12, None, None), (32, "LeakyReLu", "he_normal"), (4, "Nothing", "xavier_normal")],
    "12-16-16-4": [(12, None, None), (16, "LeakyReLu", "he_normal"),
                   (16, "LeakyReLu", "he_normal"), (4, "Nothing", "xavier_normal")],
    "12-64-64-4": [(12, None, None), (64, "LeakyReLu", "he_normal"),
                   (64, "LeakyReLu", "he_normal"), (4, "Nothing", "xavier_normal")],
}

OPTIMIZERS: Dict[str, Tuple[Callable[[Network], object], float]] = {
    # nom: (fabrique, learning rate par defaut)
    "GradientDescent": (lambda net: GradientDescent(network=net), 0.01),
    "RMSProp": (lambda net: RMSPropagation(network=net, velocity_rate=0.9), 0.001),
    "ADAM": (lambda net: ADAM(network=net, momentum_rate=0.9, velocity_rate=0.999), 0.001),
}

NP_ACTIVATIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "Nothing": lambda z: z,
    "ReLu": lambda z: np.maximum(z, 0.),
    "LeakyReLu": lambda z: np.where(z > 0., z, 0.01 * z),
    "Sigmoid": lambda z: 1. / (1. + np.exp(-np.clip(z, -500, 500))),
    "Tanh": np.tanh,
}


# --------------------------------------------------------------------------- utils

def reset_init_seed():
    """Remet le generateur des initializers a son etat initial : chaque reseau construit
    par ce script part ainsi des memes poids (comparaisons equitables, --compare fiable)."""
    if hasattr(initialisations, "generator"):
        initialisations.generator = np.random.Generator(np.random.PCG64(initialisations.SEED))
    else:
        np.random.seed(getattr(initialisations, "SEED", 42))


def build_network(arch: List[Tuple[int, str | None, str | None]], lr: float, batch_size: int,
                  reset_seed: bool = True) -> Network:
    if reset_seed:
        reset_init_seed()
    layers = [Layer(shape=s) if act is None else Layer(shape=s, activation=act, initializer=init)
              for s, act, init in arch]
    return Network(layers=layers, learning_rate=lr, batch_size=batch_size, loss_function="mean_square_error")


def numpy_forward(net: Network, arch, x: np.ndarray) -> np.ndarray:
    """Forward vectorise de reference, avec les memes poids que le reseau du package."""
    out = x
    for (_, act, _), w, b in zip(arch[1:], net.weights, net.biaises):
        out = NP_ACTIVATIONS[act](np.asarray(w) @ out + np.asarray(b))
    return out


def timeit(fn: Callable[[], None], repeats: int, inner: int) -> float:
    """Mediane, sur `repeats` mesures, du temps d'un appel (en secondes)."""
    fn()  # warmup
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        times.append((time.perf_counter() - t0) / inner)
    return median(times)


def fmt_us(seconds: float) -> str:
    us = seconds * 1e6
    return f"{us:9.1f} us" if us < 10_000 else f"{us / 1000:9.1f} ms"


def make_dataset(rng: np.random.Generator, n: int, in_dim: int, out_dim: int,
                 teacher: Tuple[np.ndarray, ...] | None = None):
    """Regression : y = professeur(x), x dans [0, 1] comme les etats du DQL."""
    if teacher is None:
        w1 = rng.normal(0, 1.5, (16, in_dim)); b1 = rng.normal(0, 0.5, 16)
        w2 = rng.normal(0, 0.8, (out_dim, 16)); b2 = rng.normal(0, 0.2, out_dim)
        teacher = (w1, b1, w2, b2)
    w1, b1, w2, b2 = teacher
    x = rng.uniform(0, 1, (n, in_dim))
    y = (np.tanh(x @ w1.T + b1) @ w2.T + b2)
    ds = [{"data": x[i], "label": y[i]} for i in range(n)]
    return ds, x, y, teacher


def mse(net: Network, x: np.ndarray, y: np.ndarray) -> float:
    pred = np.array([net.fire.full(input=xi, weights=net.weights, biaises=net.biaises) for xi in x])
    return float(np.mean((pred - y) ** 2))


# --------------------------------------------------------------------------- vitesse

def bench_speed(quick: bool, batch_size: int) -> Dict[str, Dict[str, float]]:
    repeats, inner = (3, 20) if quick else (7, 50)
    rng = np.random.default_rng(0)
    results: Dict[str, Dict[str, float]] = {}
    print(f"\n=== VITESSE (batch={batch_size}, mediane de {repeats} mesures) ===")
    print(f"{'architecture':16s} {'forward':>12s} {'forward numpy':>14s} {'ecart':>7s}  "
          + "  ".join(f"{'pas ' + o:>20s}" for o in OPTIMIZERS))
    for name, arch in ARCHS.items():
        net = build_network(arch, lr=0.001, batch_size=batch_size)
        x = rng.uniform(0, 1, arch[0][0])
        ref = numpy_forward(net, arch, x)
        got = net.fire.full(input=x, weights=net.weights, biaises=net.biaises)
        if not np.allclose(got, ref, atol=1e-6):
            print(f"  !! {name}: le forward du package differe de la reference numpy "
                  f"(ecart max {np.abs(got - ref).max():.2e})")
        t_fwd = timeit(lambda: net.fire.full(input=x, weights=net.weights, biaises=net.biaises), repeats, inner)
        t_ref = timeit(lambda: numpy_forward(net, arch, x), repeats, inner)
        res = {"forward_s": t_fwd, "forward_numpy_s": t_ref}
        batch, _, _, _ = make_dataset(rng, batch_size, arch[0][0], arch[-1][0])
        cols = []
        for opt_name, (factory, _) in OPTIMIZERS.items():
            tnet = build_network(arch, lr=1e-6, batch_size=batch_size)  # lr minuscule : on mesure, on n'apprend pas
            opt = factory(tnet)
            t_step = timeit(lambda: opt.full(batch), repeats, max(1, inner // 5))
            res[f"step_{opt_name}_s"] = t_step
            cols.append(f"{fmt_us(t_step)} ({batch_size / t_step:6.0f}/s)")
        results[name] = res
        print(f"{name:16s} {fmt_us(t_fwd):>12s} {fmt_us(t_ref):>14s} {t_fwd / t_ref:6.0f}x  " + "  ".join(f"{c:>20s}" for c in cols))
    print("  forward = 1 echantillon ; pas = forward + backward + update sur un batch (echantillons/s entre parentheses)")
    print("  ecart = combien de fois le forward du package est plus lent qu'un forward numpy vectorise")
    return results


# --------------------------------------------------------------------------- apprentissage

def bench_learning(quick: bool, batch_size: int, seed: int, lr_override: float | None) -> Dict[str, Dict]:
    epochs, n_train, n_test, eval_every = (30, 300, 100, 5) if quick else (100, 1000, 300, 10)
    arch = ARCHS["12-32-4 (DQL)"]
    rng = np.random.default_rng(seed)
    train, _, _, teacher = make_dataset(rng, n_train, arch[0][0], arch[-1][0])
    _, x_test, y_test, _ = make_dataset(rng, n_test, arch[0][0], arch[-1][0], teacher)
    baseline = float(np.mean((y_test - y_test.mean(axis=0)) ** 2))
    print(f"\n=== APPRENTISSAGE : regression sur un reseau professeur, archi {list(ARCHS)[0]} ===")
    print(f"train={n_train} test={n_test} batch={batch_size} epochs={epochs} | "
          f"MSE de reference (predire la moyenne) = {baseline:.4f}")
    results: Dict[str, Dict[str, float]] = {}
    for opt_name, (factory, default_lr) in OPTIMIZERS.items():
        lr = lr_override if lr_override is not None else default_lr
        net = build_network(arch, lr=lr, batch_size=batch_size)
        opt = factory(net)
        curve = [(0, mse(net, x_test, y_test))]
        t0 = time.perf_counter()
        diverged = False
        for e in range(1, epochs + 1):
            opt.deterministic(train)
            if e % eval_every == 0 or e == epochs:
                loss = mse(net, x_test, y_test)
                curve.append((e, loss))
                if not np.isfinite(loss):
                    diverged = True
                    break
        train_time = time.perf_counter() - t0  # inclut les evaluations
        final = curve[-1][1]
        reach = next((e for e, l in curve if l <= 0.1 * baseline), None)
        results[opt_name] = {"lr": lr, "initial_mse": curve[0][1], "final_mse": final, "r2": 1 - final / baseline,
                             "diverged": int(diverged),
                             "epochs_to_10pct": reach if reach is not None else -1, "time_s": train_time}
        status = "DIVERGE" if diverged else f"R2={1 - final / baseline:6.3f}"
        print(f"  {opt_name:16s} lr={lr:<7g} MSE finale={final:9.4f}  {status}  "
              f"MSE<10% ref a l'epoch: {reach if reach is not None else 'jamais':>6}  temps={train_time:6.1f}s")
        print("    courbe: " + "  ".join(f"e{e}={l:.4f}" for e, l in curve))
    print("  R2 = 1 - MSE/MSE_ref : 1 = parfait, 0 = pas mieux que predire la moyenne")
    return results


# --------------------------------------------------------------------------- check

R2_MIN = 0.5          # R2 minimum attendu apres le benchmark d'apprentissage complet
GRAD_TOL = 1e-4       # erreur relative max entre gradient analytique et numerique

CHECK_ARCHS: Dict[str, List[Tuple[int, str | None, str | None]]] = {
    f"cachee {act}": [(5, None, None), (4, act, "xavier_normal"), (3, "Nothing", "xavier_normal")]
    for act in ("Nothing", "LeakyReLu", "ReLu", "Sigmoid", "Tanh")
}
CHECK_ARCHS["sortie Sigmoid"] = [(5, None, None), (4, "Tanh", "xavier_normal"), (3, "Sigmoid", "xavier_normal")]
CHECK_ARCHS["2 couches cachees"] = [(5, None, None), (4, "LeakyReLu", "he_normal"),
                                    (4, "Tanh", "xavier_normal"), (3, "Nothing", "xavier_normal")]


class Checker:
    def __init__(self):
        self.failures = 0
        self.warnings = 0

    def report(self, ok: bool, name: str, detail: str = "", warn_only: bool = False):
        if ok:
            tag = "OK    "
        elif warn_only:
            tag = "AVERT "
            self.warnings += 1
        else:
            tag = "ECHEC "
            self.failures += 1
        print(f"  [{tag}] {name}" + (f" -- {detail}" if detail else ""))


def half_sse(net: Network, batch) -> float:
    """Perte que derive le backward du package : 0.5 * somme des (y - label)^2 sur le batch."""
    return sum(0.5 * float(np.sum((net.fire.full(input=d["data"], weights=net.weights, biaises=net.biaises)
                                   - d["label"]) ** 2)) for d in batch)


def gradient_check(net: Network, batch, eps: float = 1e-6) -> float:
    """Erreur relative max entre les gradients du backward et une difference finie centree."""
    net.fire._reset()
    net.fire.backward(batch, net.weights, net.biaises)
    analytic = [np.array(g, dtype=float).copy() for g in net.fire.nabla_w] + \
               [np.array(g, dtype=float).copy() for g in net.fire.nabla_b]
    net.fire._reset()
    params = list(net.weights) + list(net.biaises)
    worst = 0.
    for p, g in zip(params, analytic):
        if g.shape != np.shape(p):
            return float("inf")
        it = np.nditer(p, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            old = p[idx]
            p[idx] = old + eps; lp = half_sse(net, batch)
            p[idx] = old - eps; lm = half_sse(net, batch)
            p[idx] = old
            num = (lp - lm) / (2 * eps)
            worst = max(worst, abs(num - g[idx]) / max(1e-8, abs(num) + abs(g[idx])))
    return worst


def run_checks(batch_size: int, seed: int) -> int:
    c = Checker()
    rng = np.random.default_rng(seed)

    print("\n=== CHECK 1 : forward identique a une reference numpy ===")
    for name, arch in {**ARCHS, **CHECK_ARCHS}.items():
        net = build_network(arch, lr=0.01, batch_size=batch_size)
        xs = rng.uniform(-1, 1, (20, arch[0][0]))
        err = max(float(np.abs(net.fire.full(input=x, weights=net.weights, biaises=net.biaises)
                               - numpy_forward(net, arch, x)).max()) for x in xs)
        c.report(err < 1e-6, name, f"ecart max {err:.1e}")

    print(f"\n=== CHECK 2 : gradients du backward = gradients numeriques (tolerance {GRAD_TOL:g}) ===")
    for name, arch in CHECK_ARCHS.items():
        net = build_network(arch, lr=0.01, batch_size=3)
        batch = [{"data": rng.uniform(-1, 1, arch[0][0]), "label": rng.uniform(-1, 1, arch[-1][0])}
                 for _ in range(3)]
        err = gradient_check(net, batch)
        c.report(err < GRAD_TOL, name, f"erreur relative max {err:.1e}")

    print("\n=== CHECK 3 : Network.copy() cree un reseau independant ===")
    net = build_network(ARCHS["12-32-4 (DQL)"], lr=0.01, batch_size=batch_size)
    cp = net.copy()
    x = rng.uniform(0, 1, 12)
    same_out = np.allclose(net.fire.full(input=x, weights=net.weights, biaises=net.biaises),
                           cp.fire.full(input=x, weights=cp.weights, biaises=cp.biaises))
    c.report(same_out, "la copie donne les memes sorties que l'original")
    before = [np.array(w).copy() for w in cp.weights]
    for w in net.weights:
        w += 1.
    for b in net.biaises:
        b += 1.
    unchanged = all(np.array_equal(a, np.array(b)) for a, b in zip(before, cp.weights))
    c.report(unchanged, "modifier l'original ne modifie pas la copie")

    print("\n=== CHECK 4 : un pas de chaque optimizer fait baisser la perte du batch ===")
    arch = ARCHS["12-32-4 (DQL)"]
    batch, _, _, _ = make_dataset(rng, batch_size, arch[0][0], arch[-1][0])
    for opt_name, (factory, _) in OPTIMIZERS.items():
        net = build_network(arch, lr=1e-4, batch_size=batch_size)
        opt = factory(net)
        l0 = half_sse(net, batch)
        opt.full(batch)
        l1 = half_sse(net, batch)
        c.report(np.isfinite(l1) and l1 < l0, opt_name, f"perte {l0:.4f} -> {l1:.4f}")

    print("\n=== CHECK 5 : initialisation reproductible (necessaire pour --compare) ===")
    a = build_network(arch, lr=0.01, batch_size=batch_size)
    b = build_network(arch, lr=0.01, batch_size=batch_size)
    same = all(np.array_equal(np.array(x), np.array(y)) for x, y in zip(a.weights, b.weights))
    c.report(same, "deux reseaux construits avec la meme seed sont identiques",
             "" if same else "les resultats de --compare ne seront pas comparables", warn_only=True)

    print(f"\n=== CHECK 6 : apprentissage complet (R2 >= {R2_MIN}) ===")
    learn = bench_learning(False, batch_size, seed, None)
    for opt_name, r in learn.items():
        ok_finite = not r["diverged"] and np.isfinite(r["final_mse"])
        c.report(ok_finite, f"{opt_name} : pas de NaN / divergence")
        c.report(r["final_mse"] < r["initial_mse"], f"{opt_name} : la perte baisse",
                 f"{r['initial_mse']:.4f} -> {r['final_mse']:.4f}")
        c.report(r["r2"] >= R2_MIN, f"{opt_name} : R2 suffisant", f"R2 = {r['r2']:.3f}")

    print(f"\n=== BILAN : {c.failures} echec(s), {c.warnings} avertissement(s) ===")
    print("TOUT EST OK" if c.failures == 0 else "IL Y A DES ECHECS (voir les lignes [ECHEC] ci-dessus)")
    return 1 if c.failures else 0


# --------------------------------------------------------------------------- main

def compare(old: Dict, new: Dict):
    print("\n=== COMPARAISON (ancien -> nouveau) ===")
    for section in ("speed", "learn"):
        for key, vals in new.get(section, {}).items():
            for metric, v in vals.items():
                o = old.get(section, {}).get(key, {}).get(metric)
                if o is None or not isinstance(v, (int, float)):
                    continue
                if metric.endswith("_s") and v > 0:
                    print(f"  {section:5s} {key:16s} {metric:22s} {o:.3e} -> {v:.3e}  ({o / v:5.2f}x plus rapide)")
                else:
                    print(f"  {section:5s} {key:16s} {metric:22s} {o:.4f} -> {v:.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true",
                   help="verifie que le package n'est pas casse (OK/ECHEC, code de sortie 1 si echec)")
    p.add_argument("--quick", action="store_true", help="moins de mesures et d'epochs")
    p.add_argument("--only", choices=["speed", "learn"], help="ne lancer qu'une partie")
    p.add_argument("--batch", type=int, default=10, help="taille de batch (defaut 10, comme le DQL)")
    p.add_argument("--seed", type=int, default=0, help="seed des donnees (l'init des poids est fixee par le package)")
    p.add_argument("--lr", type=float, default=None, help="impose le meme learning rate a tous les optimizers")
    p.add_argument("--save", help="sauvegarde les resultats en JSON")
    p.add_argument("--compare", help="compare avec un JSON sauvegarde precedemment")
    args = p.parse_args()

    print(f"mlp charge depuis : {mlp.__path__[0] if hasattr(mlp, '__path__') else mlp.__file__}")
    if args.check:
        sys.exit(run_checks(args.batch, args.seed))
    results: Dict[str, Dict] = {}
    if args.only in (None, "speed"):
        results["speed"] = bench_speed(args.quick, args.batch)
    if args.only in (None, "learn"):
        results["learn"] = bench_learning(args.quick, args.batch, args.seed, args.lr)
    if args.save:
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nresultats sauvegardes dans {args.save}")
    if args.compare:
        with open(args.compare) as f:
            compare(json.load(f), results)


if __name__ == "__main__":
    main()
