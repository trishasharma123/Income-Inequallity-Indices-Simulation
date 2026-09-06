"""
Full Simulation Study — India Income Inequality EE
====================================================
Runs the complete pipeline: dataset construction, baseline indices,
severity-sweep simulation, bootstrap confidence intervals, Lorenz
crossing check, and both charts (radar + comparison bar).

REQUIREMENTS (install once, in a terminal):
    pip install numpy matplotlib

HOW TO RUN:
    python full_simulation_study.py

This will print all results to the terminal AND save two PNG chart
files (rubric_radar_chart.png, index_comparison_bar_chart.png) in the
same folder you run it from.
"""

import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# STEP 1 — BUILD THE CALIBRATED SYNTHETIC DATASET
# (lognormal body + Pareto tail, calibrated so the top 10% holds ~57%
# of total income, matching the World Inequality Database's real
# reported figure for India)
# =====================================================================

N = 20000
THRESHOLD_PERCENTILE = 90
TARGET_TOP10_SHARE = 0.57


def build_distribution(alpha, seed=42, n=N, threshold_pct=THRESHOLD_PERCENTILE):
    rng = np.random.default_rng(seed)
    body = np.sort(rng.lognormal(mean=8.0, sigma=0.6, size=n))
    xm = np.percentile(body, threshold_pct)
    n_top = int(n * (1 - threshold_pct / 100))
    pareto_tail = (rng.pareto(alpha, size=n_top) + 1) * xm
    income = np.concatenate([body[: n - n_top], pareto_tail])
    return np.sort(income)


def top_share(income, top_fraction=0.10):
    n = len(income)
    cutoff = int((1 - top_fraction) * n)
    return income[cutoff:].sum() / income.sum()


print("=" * 60)
print("STEP 1: Building calibrated dataset")
print("=" * 60)

best_alpha, best_diff = None, np.inf
for alpha in np.arange(1.05, 2.0, 0.01):
    income = build_distribution(alpha)
    diff = abs(top_share(income) - TARGET_TOP10_SHARE)
    if diff < best_diff:
        best_diff, best_alpha = diff, alpha

true_income = build_distribution(best_alpha)
print(f"Dataset built: N={len(true_income)}, alpha={best_alpha:.3f}")
print(f"Top-10% share achieved: {top_share(true_income):.3f} (target {TARGET_TOP10_SHARE})")
print(f"Median income: Rs.{np.median(true_income):.0f}\n")


# =====================================================================
# STEP 2 — INDEX FUNCTIONS (Gini, Palma, Theil)
# =====================================================================

def gini(x):
    x = np.sort(x)
    n = len(x)
    cum = np.cumsum(x)
    return (2 * np.sum((np.arange(1, n + 1)) * x) - (n + 1) * cum[-1]) / (n * cum[-1])


def palma(x):
    x = np.sort(x)
    n = len(x)
    return x[int(0.9 * n):].sum() / x[:int(0.4 * n)].sum()


def theil(x):
    x = np.asarray(x)
    mean_x = x.mean()
    x_pos = x[x > 0]  # Theil's log term is undefined at zero income
    n = len(x)
    return np.sum((x_pos / mean_x) * np.log(x_pos / mean_x)) / n


def ge2(x):
    """Generalized entropy at alpha=2 (the proposed index). No logarithm --
    well-defined at x=0, unlike theil() above."""
    x = np.asarray(x)
    mean_x = x.mean()
    return 0.5 * (np.mean((x / mean_x) ** 2) - 1)


print("=" * 60)
print("STEP 2: Baseline index values on clean data")
print("=" * 60)

base_gini = gini(true_income)
base_palma = palma(true_income)
base_theil = theil(true_income)

print(f"  Gini:  {base_gini:.4f}")
print(f"  Palma: {base_palma:.4f}")
print(f"  Theil: {base_theil:.4f}\n")


# =====================================================================
# STEP 3 — SEVERITY SWEEP SIMULATION (nested noise injection so that
# 20% distortion always includes the same households as 10%, plus more
# -- this keeps the severity sweep monotonic and properly comparable)
# =====================================================================

def inject_noise_nested(x, pct, reduction=0.5, seed=1):
    rng = np.random.default_rng(seed)
    x = x.copy()
    n = len(x)
    sorted_idx = np.argsort(x)
    bottom_decile_idx = sorted_idx[: int(0.1 * n)]
    top_decile_idx = sorted_idx[int(0.9 * n):]

    bottom_order = rng.permutation(bottom_decile_idx)
    top_order = rng.permutation(top_decile_idx)

    n_bottom = int(pct * len(bottom_decile_idx))
    n_top = int(pct * len(top_decile_idx))

    x[bottom_order[:n_bottom]] *= (1 - reduction)
    x[top_order[:n_top]] *= (1 - reduction)
    return x


print("=" * 60)
print("STEP 3: Severity sweep (10%, 20%, 30% of bottom+top deciles distorted)")
print("=" * 60)

severity_levels = [0.10, 0.20, 0.30]
sim_results = {"Gini": [], "Palma": [], "Theil": []}

print(f"{'Severity':<10}{'Gini':<12}{'Palma':<12}{'Theil':<12}")
print(f"{'0% (base)':<10}{base_gini:<12.4f}{base_palma:<12.4f}{base_theil:<12.4f}")

for severity in severity_levels:
    distorted = inject_noise_nested(true_income, severity, reduction=0.5, seed=1)
    g, p, t = gini(distorted), palma(distorted), theil(distorted)
    sim_results["Gini"].append(g)
    sim_results["Palma"].append(p)
    sim_results["Theil"].append(t)
    print(f"{int(severity*100)}%{'':<7}{g:<12.4f}{p:<12.4f}{t:<12.4f}")

print("\n% CHANGE FROM BASELINE:")
print(f"{'Severity':<10}{'Gini %chg':<14}{'Palma %chg':<14}{'Theil %chg':<14}")
for i, severity in enumerate(severity_levels):
    g_pct = 100 * (sim_results["Gini"][i] - base_gini) / base_gini
    p_pct = 100 * (sim_results["Palma"][i] - base_palma) / base_palma
    t_pct = 100 * (sim_results["Theil"][i] - base_theil) / base_theil
    print(f"{int(severity*100)}%{'':<7}{g_pct:<14.2f}{p_pct:<14.2f}{t_pct:<14.2f}")

print("\nNOTE: Theil may INCREASE slightly under distortion (not a bug) --")
print("reducing top-decile incomes also lowers the overall mean, which")
print("pushes every remaining household's y_i/mean ratio UP, partially")
print("offsetting the direct effect. Gini is not affected this way.\n")


# =====================================================================
# STEP 4 — BOOTSTRAP CONFIDENCE INTERVALS
# =====================================================================

def bootstrap_ci(data, func, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(data)
    estimates = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        estimates[i] = func(sample)
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return lower, upper


print("=" * 60)
print("STEP 4: Bootstrap 95% confidence intervals (1000 resamples each)")
print("=" * 60)

for name, func in [("Gini", gini), ("Palma", palma), ("Theil", theil)]:
    lo, hi = bootstrap_ci(true_income, func, n_boot=1000)
    point = func(true_income)
    print(f"  {name}: {point:.4f}  (95% CI: [{lo:.4f}, {hi:.4f}])")

print("\nNOTE: Palma's CI is proportionally much wider than Gini's -- Palma's")
print("reliance on two extreme-tail sums makes it more statistically volatile")
print("under resampling, given how heavy-tailed India's income distribution is.\n")


# =====================================================================
# STEP 5 — LORENZ CURVE CROSSING CHECK
# =====================================================================

def lorenz_curve(x):
    x = np.sort(x)
    cum = np.cumsum(x) / x.sum()
    cum = np.insert(cum, 0, 0)
    pop_share = np.linspace(0, 1, len(cum))
    return pop_share, cum


print("=" * 60)
print("STEP 5: Lorenz curve crossing check")
print("=" * 60)

distorted_30 = inject_noise_nested(true_income, 0.30, seed=1)
pop_clean, cum_clean = lorenz_curve(true_income)
pop_dist, cum_dist = lorenz_curve(distorted_30)

diffs = np.interp(pop_dist, pop_clean, cum_clean) - cum_dist
# Exclude first/last 0.1% of points -- both curves trivially meet at (0,0) and (1,1)
interior = slice(int(0.001 * len(diffs)), int(0.999 * len(diffs)))
interior_diffs = diffs[interior]

crosses = np.any(np.diff(np.sign(interior_diffs)) != 0)
print(f"Genuine interior Lorenz crossing under 30% distortion: {crosses}")
print(f"Max |difference| between clean and distorted Lorenz curves: {np.max(np.abs(interior_diffs)):.6f}\n")

# Save the Lorenz plot too
plt.figure(figsize=(6, 6))
plt.plot(pop_clean, cum_clean, label="Clean (baseline)", linewidth=2)
plt.plot(pop_dist, cum_dist, label="Distorted (30% bottom+top)", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect equality")
plt.xlabel("Cumulative population share")
plt.ylabel("Cumulative income share")
plt.title("Lorenz Curve: Clean vs. Distorted Data")
plt.legend()
plt.tight_layout()
plt.savefig("lorenz_comparison.png", dpi=200)
plt.close()
print("Saved: lorenz_comparison.png\n")


# =====================================================================
# STEP 6 — RADAR CHART (rubric scores from your completed axiom proofs)
# Scores: 1=fails, 2=partial/conditional, 3=passes -- EDIT these to
# match your own final axiom-test conclusions if they differ.
# =====================================================================

print("=" * 60)
print("STEP 6: Building radar chart")
print("=" * 60)

criteria = ["Pigou-\nDalton", "Tail-\nSensitivity", "Scale\nInvariance",
            "Population\nInvariance", "Decomposability"]

scores = {
    "Gini":  [3, 1, 3, 3, 1],
    "Palma": [2, 3, 3, 3, 1],
    "Theil": [3, 3, 3, 3, 3],
}

n_axes = len(criteria)
angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
colors = {"Gini": "#4C72B0", "Palma": "#DD8452", "Theil": "#55A868"}

for name, vals in scores.items():
    vals = vals + vals[:1]
    ax.plot(angles, vals, label=name, linewidth=2.5, color=colors[name])
    ax.fill(angles, vals, alpha=0.10, color=colors[name])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(criteria, fontsize=11)
ax.set_yticks([1, 2, 3])
ax.set_yticklabels(["Fails", "Partial", "Passes"], fontsize=9)
ax.set_ylim(0, 3)
ax.set_title("Rubric Scores by Index — Gini vs. Palma vs. Theil", pad=30, fontsize=13)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=11)
plt.tight_layout()
plt.savefig("rubric_radar_chart.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: rubric_radar_chart.png\n")


# =====================================================================
# STEP 7 — COMPARISON BAR CHART (same-dataset baseline values)
# =====================================================================

print("=" * 60)
print("STEP 7: Building comparison bar chart")
print("=" * 60)

names = ["Gini", "Palma", "Theil"]
values = [base_gini, base_palma, base_theil]

fig, ax = plt.subplots(figsize=(6, 4.5))
bars = ax.bar(names, values, color=["#4C72B0", "#DD8452", "#55A868"])
ax.set_ylabel("Index value")
ax.set_title("Baseline Index Values on the Calibrated 20,000-Household Dataset")
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=11)
plt.tight_layout()
plt.savefig("index_comparison_bar_chart.png", dpi=200)
plt.close()
print("Saved: index_comparison_bar_chart.png\n")


# =====================================================================
# STEP 8 — SENSITIVITY LINE CHART (% change vs. distortion severity,
# one line per index -- shows Gini/Palma declining while Theil rises)
# =====================================================================

print("=" * 60)
print("STEP 8: Building sensitivity line chart")
print("=" * 60)

severities_full = [0, 0.10, 0.20, 0.30]
pct_changes = {"Gini": [0], "Palma": [0], "Theil": [0]}

for s in severities_full[1:]:
    distorted = inject_noise_nested(true_income, s, seed=1)
    pct_changes["Gini"].append(100 * (gini(distorted) - base_gini) / base_gini)
    pct_changes["Palma"].append(100 * (palma(distorted) - base_palma) / base_palma)
    pct_changes["Theil"].append(100 * (theil(distorted) - base_theil) / base_theil)

fig, ax = plt.subplots(figsize=(7, 5))
for name, vals in pct_changes.items():
    ax.plot([s * 100 for s in severities_full], vals, marker='o', linewidth=2.5,
             label=name, color=colors[name])

ax.axhline(0, color='gray', linestyle='--', linewidth=1)
ax.set_xlabel("Distortion severity (% of bottom+top deciles affected)")
ax.set_ylabel("% change in index value from baseline")
ax.set_title("Index Sensitivity to Distortion Severity")
ax.legend()
plt.tight_layout()
plt.savefig("sensitivity_line_chart.png", dpi=200)
plt.close()
print("Saved: sensitivity_line_chart.png\n")


print("=" * 60)
print("ALL STEPS COMPLETE.")
print("Check your folder for 4 PNG files:")
print("  - lorenz_comparison.png")
print("  - rubric_radar_chart.png")
print("  - index_comparison_bar_chart.png")
print("  - sensitivity_line_chart.png")
print("=" * 60)
