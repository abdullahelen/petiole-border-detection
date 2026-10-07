"""
Automatic detection of the petiole border in plant leaves.

Reference implementation of:

    A. Elen and E. Avuçlu (2021), "Automatic detection of petiole border in
    plant leaves", Measurement and Control, 54(3-4).
    https://doi.org/10.1177/0020294020917701

Pipeline (equation numbers refer to the paper)
----------------------------------------------
Eq. (1)-(6)  : Y-axis cumulative profile h(i) + triangle (Zack) threshold -> T_opt
Eq. (7)-(9)  : petiole width range [X_left, X_right] from the X-axis profile (Fig. 7)
Eq. (10)-(11):

    C_min = min_i { i | i in {1, ..., T_opt - X_max} : s(i) = 0 } - 1   (10)
    P     = T_opt - C_min                                               (11)

computed on the cropped blade image I_c (Fig. 5c, rows X_max..T_opt) whose rows
are numbered upward from its lower edge (i = 1 is the row just above T_opt), with

    s(i) = 1  if R(i) >= 2 and [ I_c(i, X_left^(i) - 1) = 0 or I_c(i, X_right^(i) + 1) = 0 ]
           0  otherwise

R(i)                 : number of separate foreground runs in row i
X_left^(i), X_right^(i): petiole columns, starting from Eq. (7)-(9) and
                         following a slanted petiole row by row.

C_min is the depth of the basal sinus; for a leaf without a sinus C_min = 0 and
P = T_opt.

The demo uses synthetic binary leaves with a KNOWN petiole/blade junction as
ground truth. Images are oriented with the leaf tip at the top and the petiole
pointing downward, and are scaled to width 100 as in the paper (Eq. 12).

Usage
-----
    python petiole_border.py            # saves figures/petiole_demo.png
    python petiole_border.py --show     # also opens the figure window
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

W = 100  # image width after scaling (paper, Eq. 12)


# --------------------------------------------------------------------------
# Synthetic test leaves
# --------------------------------------------------------------------------
def make_leaf(kind: str, H: int = 240, petiole_len: int = 60, pw: int = 4,
              seed: int = 0) -> tuple[np.ndarray, int]:
    """Return a binary leaf image (H x W, tip up, petiole down) and the
    ground-truth row of the petiole/blade junction."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    cx = W / 2 + rng.uniform(-2, 2)
    junction = H - petiole_len                      # ground-truth border row
    top = 15
    L = junction - top                              # blade length
    t = (yy - top) / L                              # 0 at tip, 1 at base
    tc = np.clip(t, 0, 1)
    inside_t = (t >= 0) & (t <= 1)

    if kind == "ovate":
        half = 45 * np.sin(np.pi * tc ** 0.8)
    elif kind == "elliptic":
        half = 40 * np.sin(np.pi * tc)
    elif kind == "lanceolate":
        half = 22 * np.sin(np.pi * tc ** 1.3)
    elif kind == "obovate":
        half = 45 * np.sin(np.pi * tc ** 1.6)
    elif kind == "attenuate":   # base tapers gradually into the petiole
        half = 42 * np.sin(np.pi * tc ** 0.6) * (1 - tc) ** 0.5 + pw / 2
    elif kind == "cordate":     # heart-shaped: two basal lobes hang below the junction
        half = None
    else:
        raise ValueError(f"unknown leaf kind: {kind}")

    if kind != "cordate":
        blade = inside_t & (np.abs(xx - cx) <= half)
    else:
        t2 = (yy - top) / (L + 25)
        half = 45 * np.sin(np.pi * np.clip(t2, 0, 1) ** 0.75)
        blade = (t2 >= 0) & (t2 <= 1) & (np.abs(xx - cx) <= half)
        sinus = (yy > junction) & (np.abs(xx - cx) < 12 * (yy - junction) / 25 + 1)
        blade &= ~sinus

    # slightly slanted petiole
    pet = ((yy >= junction) & (yy < H - 2)
           & (np.abs(xx - cx - 0.15 * (yy - junction)) <= pw / 2))
    return (blade | pet).astype(np.uint8), junction


# --------------------------------------------------------------------------
# Method
# --------------------------------------------------------------------------
def t_opt(img: np.ndarray):
    """Eq. (1)-(6): optimum threshold T_opt on the Y-axis cumulative profile."""
    h = img.sum(axis=1).astype(float)               # h(i), pixels per row
    nz = np.nonzero(h)[0]
    xmin = int(nz[-1])                              # Eq. (1): first non-zero, right -> left
    xmax = int(np.argmax(h))                        # Eq. (2): maximum peak
    ymin, ymax = h[xmin], h[xmax]                   # Eq. (3), (4)
    a, b = ymax - ymin, -(xmax - xmin)              # Eq. (5): line d: a x + b y + c = 0
    c = -(a * xmax + b * ymax)
    r = np.arange(xmax, xmin + 1)
    dist = np.abs(a * r + b * h[r] + c) / np.hypot(a, b)   # Eq. (6)
    return int(r[np.argmax(dist)]), h, (xmin, xmax, ymin, ymax)


def petiole_range(img: np.ndarray, T: int, rows: int = 6) -> tuple[int, int]:
    """Eq. (7)-(9) (Fig. 7): petiole column range [X_left, X_right].

    Applied to the top of the cropped petiole (the first `rows` rows below
    T_opt) so that a slanted petiole does not widen the range."""
    g = img[T:T + rows, :].sum(axis=0)
    xm = int(np.argmax(g))                                    # Eq. (7)
    xr = xm + int(np.nonzero(g[xm:] == 0)[0][0]) - 1          # Eq. (8)
    xl = int(np.nonzero(g[:xm + 1] == 0)[0][-1]) + 1          # Eq. (9)
    return xl, xr


def n_runs(row: np.ndarray) -> int:
    """R(i): number of separate foreground runs in a row."""
    fg = np.nonzero(row)[0]
    return 0 if fg.size == 0 else 1 + int(np.sum(np.diff(fg) > 1))


def c_min(img: np.ndarray, T: int, xmax: int, xl: int, xr: int) -> int:
    """Eq. (10): depth of the basal sinus, C_min.

    I_c is the cropped blade image (rows X_max..T_opt) with rows numbered
    upward from its lower edge (i = 1 is the row just above T_opt). The petiole
    columns X_left^(i), X_right^(i) start from Eq. (7)-(9) and follow the
    petiole row by row."""
    Ic = img[xmax:T][::-1]                     # I_c, row i  ->  Ic[i - 1]
    w = xr - xl + 1
    n = Ic.shape[1]
    i = 1
    while i <= Ic.shape[0]:
        row = Ic[i - 1]
        lo, hi = max(xl - 1, 0), min(xr + 1, n - 1)
        fg = np.nonzero(row[lo:hi + 1])[0]
        if fg.size == 0:
            break                              # s(i) = 0
        xl = lo + int(fg[0])                   # X_left^(i)
        xr = min(xl + w - 1, n - 2)            # X_right^(i)
        s_i = n_runs(row) >= 2 and (row[xl - 1] == 0 or row[xr + 1] == 0)
        if not s_i:
            break                              # first i with s(i) = 0
        i += 1
    return i - 1


def detect_petiole_border(img: np.ndarray) -> dict:
    """Full pipeline: binary leaf image (tip up, petiole down) -> border row P."""
    T, h, (xmin, xmax, ymin, ymax) = t_opt(img)     # Eq. (1)-(6)
    xl, xr = petiole_range(img, T)                  # Eq. (7)-(9)
    C = c_min(img, T, xmax, xl, xr)                 # Eq. (10)
    P = T - C                                       # Eq. (11)
    return dict(P=P, T_opt=T, C_min=C, X_left=xl, X_right=xr, h=h,
                X_min=xmin, X_max=xmax, Y_min=ymin, Y_max=ymax)


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------
def main(show: bool = False, out: str = "figures/petiole_demo.png") -> None:
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinds = ["elliptic", "ovate", "lanceolate", "attenuate", "cordate"]
    fig, axes = plt.subplots(2, len(kinds), figsize=(15, 6.5))
    print(f"{'leaf':11s} {'truth':>5s} {'T_opt':>11s} {'X_left..X_right':>16s} "
          f"{'C_min':>5s} {'P':>11s}")
    for k, kind in enumerate(kinds):
        img, gt = make_leaf(kind, seed=k)
        r = detect_petiole_border(img)
        T, P, C, h = r["T_opt"], r["P"], r["C_min"], r["h"]

        ax = axes[0, k]
        ax.imshow(img, cmap="gray_r")
        ax.axhline(gt, color="g", lw=1.5, label="ground-truth border")
        ax.axhline(T, color="r", ls="--", lw=1.5, label="$T_{opt}$ (Eq. 6)")
        ax.axhline(P, color="c", lw=1.8, alpha=0.9,
                   label="$P = T_{opt} - C_{min}$ (Eq. 11)")
        ax.axvspan(r["X_left"], r["X_right"], color="orange", alpha=0.3,
                   label="$[X_{left}, X_{right}]$ (Eq. 7-9)")
        ax.set_title(f"{kind}\nerror: $T_{{opt}}$ {T - gt:+d} | $P$ {P - gt:+d}"
                     f"  ($C_{{min}}$={C})", fontsize=9)
        ax.axis("off")

        ax = axes[1, k]
        ax.plot(h, color="k", lw=1)
        ax.plot([r["X_max"], r["X_min"]], [r["Y_max"], r["Y_min"]], "b-", lw=1)
        ax.axvline(gt, color="g")
        ax.axvline(T, color="r", ls="--")
        ax.axvline(P, color="c")
        ax.set_xlabel("row (i)")
        ax.set_ylabel("h(i)" if k == 0 else "")

        print(f"{kind:11s} {gt:5d} {T:5d} ({T - gt:+3d}) "
              f"{r['X_left']:>9d}..{r['X_right']:<5d} {C:5d} {P:5d} ({P - gt:+3d})")

    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center",
               ncol=4, fontsize=9, frameon=False)
    fig.suptitle("Automatic detection of petiole border in plant leaves",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\nFigure saved to {out}")
    if show:
        plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--show", action="store_true", help="open the figure window")
    ap.add_argument("--out", default="figures/petiole_demo.png",
                    help="output path for the demo figure")
    a = ap.parse_args()
    main(show=a.show, out=a.out)
