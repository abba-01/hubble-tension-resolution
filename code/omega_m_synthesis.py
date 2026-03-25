"""
Ω_m Synthesis Figure — All Probes on One Axis
==============================================
Author : Eric D. Martin
Framework: UHA / N/U Algebra
Date   : 2026-03-24

PURPOSE
-------
Publication-quality figure showing the implied Ω_m from every independent
cosmological probe used in the UHA audit, plotted on a single axis with
error bars. The convergence of BAO, RSD, and WL around Ω_m ≈ 0.295 vs
Planck ΛCDM 0.3153 is the "smoking gun" figure for the paper.

Also produces a companion σ₈ figure.

All numbers sourced from manuscript_data.json (the SSOT).
"""

import numpy as np
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

# ─────────────────────────────────────────────────────────────────────────────
# DATA (from manuscript_data.json SSOT)
# ─────────────────────────────────────────────────────────────────────────────

OM_PLANCK     = 0.3153
OM_PLANCK_ERR = 0.0073
SIG8_PLANCK   = 0.8111
SIG8_ERR_PLANCK = 0.0060

# Ω_m measurements — each probe's direct or implied value
OM_PROBES = [
    # label,                     Om,     err,    color,       marker, note
    ("BAO eBOSS DR16\n(Alam+2021)",
                                  0.295,  0.010, "goldenrod",  "D",  "direct"),
    ("RSD fσ₈ (BOSS DR12)\n[Ω_m residual from ~0% ξ-relief]",
                                  0.295,  0.020, "steelblue",  "s",  "implied"),
    ("KiDS-1000 WL\n(Asgari+2021)",
                                  0.2627, 0.0166,"orchid",     "o",  "implied"),
    ("DES Y3 WL\n(Abbott+2022)",
                                  0.2746, 0.0120,"tomato",     "o",  "implied"),
    ("HSC Y3 WL\n(Dalal+2023)",
                                  0.2697, 0.0217,"salmon",     "o",  "implied"),
    ("KiDS+BOSS+2dFLenS\n(Heymans+2021)",
                                  0.2676, 0.0140,"mediumorchid","o", "implied"),
    ("WL Combined\n(IVW)",
                                  0.2696, 0.0075,"purple",     "^",  "combined"),
    ("BAO + WL Joint\n(this work)",
                                  0.2787, 0.0060,"darkgreen",  "*",  "joint"),
    ("Planck CMB ΛCDM\n(Planck2018)",
                                  0.3153, 0.0073,"red",        "P",  "reference"),
    ("ACT DR6 lensing\n(Madhavacheril+2024)",
                                  0.307,  0.020, "seagreen",   "v",  "CMB-lensing"),
]

# σ₈ measurements
SIG8_PROBES = [
    # label,                      sig8,   err,    color,      marker
    ("KiDS-1000 (Ω_m-pinned)\n0.295",
                                  0.7654, 0.0275, "orchid",    "o"),
    ("DES Y3 (Ω_m-pinned)\n0.295",
                                  0.7825, 0.0217, "tomato",    "o"),
    ("HSC Y3 (Ω_m-pinned)\n0.295",
                                  0.7755, 0.0339, "salmon",    "o"),
    ("KiDS+BOSS+2dFLenS\n(Ω_m-pinned)",
                                  0.7725, 0.0240, "mediumorchid","o"),
    ("WL Combined σ₈\n(IVW, Ω_m-pinned)",
                                  0.7749, 0.0129, "purple",    "^"),
    ("Planck CMB ΛCDM\n(Planck2018)",
                                  0.8111, 0.0060, "red",       "P"),
]


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Ω_m synthesis
# ─────────────────────────────────────────────────────────────────────────────

def make_Om_figure():
    fig, ax = plt.subplots(figsize=(10, 8))

    n = len(OM_PROBES)
    y_pos = np.arange(n)

    for i, (label, Om, err, color, marker, kind) in enumerate(OM_PROBES):
        lw     = 2.5 if kind in ("combined", "joint", "reference") else 1.5
        ms     = 14 if kind == "joint" else 10 if kind in ("combined","reference") else 8
        alpha  = 1.0 if kind != "implied" else 0.85
        zorder = 5 if kind == "joint" else 4 if kind in ("combined","reference") else 3

        ax.errorbar(Om, i, xerr=err,
                    fmt=marker, color=color, capsize=5,
                    linewidth=lw, markersize=ms, alpha=alpha, zorder=zorder,
                    capthick=lw)

    # Planck reference band
    ax.axvline(OM_PLANCK, color="red", linestyle="--", linewidth=1.5,
               alpha=0.8, zorder=2, label=f"Planck Ω_m = {OM_PLANCK}")
    ax.axvspan(OM_PLANCK - OM_PLANCK_ERR, OM_PLANCK + OM_PLANCK_ERR,
               alpha=0.08, color="red", zorder=1)

    # BAO+WL joint line
    bao_wl_Om = 0.2787
    ax.axvline(bao_wl_Om, color="darkgreen", linestyle=":", linewidth=2,
               alpha=0.7, zorder=2, label=f"BAO+WL joint Ω_m = {bao_wl_Om}")

    # Shade the "concordance zone"
    ax.axvspan(0.280, 0.310, alpha=0.04, color="goldenrod", zorder=0,
               label="BAO/WL concordance zone")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([p[0] for p in OM_PROBES], fontsize=9)
    ax.set_xlabel("Ω_m", fontsize=13)
    ax.set_title(
        "Cross-Probe Ω_m Synthesis\n"
        "BAO, RSD, and WL converge below Planck ΛCDM\n"
        "UHA ξ-Test | Alam+2017, Abbott+2022, Asgari+2021, Alam+2021",
        fontsize=11, pad=12
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.2, axis="x")
    ax.set_xlim(0.22, 0.36)

    # Annotation: the key tension
    delta = OM_PLANCK - bao_wl_Om
    sigma_val = delta / np.sqrt(OM_PLANCK_ERR**2 + 0.006**2)
    ax.annotate(
        f"ΔΩ_m = {delta:.4f}\n({sigma_val:.1f}σ below Planck)",
        xy=(bao_wl_Om + 0.001, 7.5),
        xytext=(bao_wl_Om + 0.020, 6.5),
        arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.5),
        fontsize=9, color="darkgreen",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="darkgreen", alpha=0.9)
    )

    # Watermark
    ax.text(0.99, 0.01, "UHA Framework | Eric D. Martin 2026",
            transform=ax.transAxes, fontsize=7, color="gray",
            ha="right", va="bottom", alpha=0.6)

    plt.tight_layout()
    outpath = "/scratch/repos/hubble-tension-resolution/figures/omega_m_synthesis.png"
    plt.savefig(outpath, dpi=180, bbox_inches="tight")
    print(f"  Ω_m synthesis figure → {outpath}")
    plt.close()
    return outpath


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: σ₈ synthesis
# ─────────────────────────────────────────────────────────────────────────────

def make_sig8_figure():
    fig, ax = plt.subplots(figsize=(9, 5))

    n = len(SIG8_PROBES)
    y_pos = np.arange(n)

    for i, (label, sig8, err, color, marker) in enumerate(SIG8_PROBES):
        is_planck = "Planck" in label
        lw = 2.5 if is_planck else 1.5
        ms = 12 if is_planck else 9
        ax.errorbar(sig8, i, xerr=err,
                    fmt=marker, color=color, capsize=5,
                    linewidth=lw, markersize=ms, zorder=4)

    ax.axvline(SIG8_PLANCK, color="red", linestyle="--", linewidth=1.5,
               alpha=0.8, label=f"Planck σ₈ = {SIG8_PLANCK}")
    ax.axvspan(SIG8_PLANCK - SIG8_ERR_PLANCK, SIG8_PLANCK + SIG8_ERR_PLANCK,
               alpha=0.10, color="red")

    sig8_wl = 0.7749
    ax.axvline(sig8_wl, color="purple", linestyle=":", linewidth=2,
               alpha=0.7, label=f"WL combined σ₈ = {sig8_wl}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([p[0] for p in SIG8_PROBES], fontsize=9)
    ax.set_xlabel("σ₈", fontsize=13)
    ax.set_title(
        "σ₈ Tension — WL Surveys vs Planck\n"
        "(after Ω_m anchoring to BAO: Ω_m = 0.295)\n"
        "UHA ξ-Test | Eric D. Martin 2026",
        fontsize=11, pad=10
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.2, axis="x")
    ax.set_xlim(0.70, 0.87)

    delta_sig8 = SIG8_PLANCK - sig8_wl
    sigma_sig8 = delta_sig8 / np.sqrt(SIG8_ERR_PLANCK**2 + 0.013**2)
    ax.annotate(
        f"Δσ₈ = {delta_sig8:.4f}\n({sigma_sig8:.1f}σ)",
        xy=(sig8_wl, 4.5),
        xytext=(sig8_wl + 0.01, 3.5),
        arrowprops=dict(arrowstyle="->", color="purple", lw=1.5),
        fontsize=9, color="purple",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="purple", alpha=0.9)
    )

    ax.text(0.99, 0.01, "UHA Framework | Eric D. Martin 2026",
            transform=ax.transAxes, fontsize=7, color="gray",
            ha="right", va="bottom", alpha=0.6)

    plt.tight_layout()
    outpath = "/scratch/repos/hubble-tension-resolution/figures/sig8_synthesis.png"
    plt.savefig(outpath, dpi=180, bbox_inches="tight")
    print(f"  σ₈ synthesis figure   → {outpath}")
    plt.close()
    return outpath


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: Four-panel audit summary
# ─────────────────────────────────────────────────────────────────────────────

def make_audit_summary():
    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel A: H₀ tension ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    labels = ["SH0ES\n(Riess+2022)", "Planck\n(CMB 2018)", "UHA Resolved\n(frame-mixing)"]
    vals   = [73.2, 67.36, 67.56]
    errs   = [1.0,  0.54,  0.60]
    colors = ["steelblue", "tomato", "darkgreen"]
    x = np.arange(len(labels))
    bars = ax1.bar(x, vals, yerr=errs, color=colors, alpha=0.7, capsize=5, width=0.5)
    ax1.axhline(67.36, color="tomato", linestyle="--", alpha=0.5, linewidth=1)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("H₀ (km/s/Mpc)", fontsize=10)
    ax1.set_title("H₀ Tension\n93% coordinate artifact", fontsize=10)
    ax1.set_ylim(64, 76)
    ax1.grid(True, alpha=0.2, axis="y")

    # ── Panel B: RSD fσ₈ ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    z_rsd     = [0.38, 0.51, 0.61]
    fsig8_pub = [0.497, 0.458, 0.436]
    fsig8_pl  = [0.4763, 0.4741, 0.4687]
    fsig8_corr= [0.4963, 0.4572, 0.4351]
    ax2.errorbar(z_rsd, fsig8_pub,  yerr=[0.045, 0.038, 0.034],
                 fmt="o", color="steelblue", capsize=4, label="BOSS DR12 published", markersize=8)
    ax2.errorbar([z+0.005 for z in z_rsd], fsig8_corr,
                 yerr=[0.045, 0.038, 0.034],
                 fmt="s", color="tomato", capsize=4, label="ξ-corrected", markersize=6,
                 markerfacecolor="none")
    ax2.plot(z_rsd, fsig8_pl, "g^--", markersize=10, label="Planck ΛCDM")
    ax2.set_xlabel("z_eff", fontsize=10); ax2.set_ylabel("fσ₈(z)", fontsize=10)
    ax2.set_title("RSD fσ₈\n0% ξ-relief → physical Ω_m", fontsize=10)
    ax2.legend(fontsize=7); ax2.grid(True, alpha=0.2)

    # ── Panel C: S₈ ──────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    s8_surveys = ["KiDS-1000", "DES Y3", "HSC Y3", "KiDS+B+2dF", "ACT DR6"]
    s8_vals    = [0.759, 0.776, 0.769, 0.766, 0.840]
    s8_errs    = [0.024, 0.017, 0.031, 0.020, 0.028]
    s8_colors  = ["orchid", "tomato", "salmon", "mediumorchid", "seagreen"]
    y = np.arange(len(s8_surveys))
    ax3.barh(y, s8_vals, xerr=s8_errs, color=s8_colors, alpha=0.7, height=0.5, capsize=4)
    ax3.axvline(0.832, color="red", linestyle="--", linewidth=1.5,
                label="Planck S₈=0.832")
    ax3.axvspan(0.832-0.013, 0.832+0.013, alpha=0.1, color="red")
    ax3.set_yticks(y); ax3.set_yticklabels(s8_surveys, fontsize=8)
    ax3.set_xlabel("S₈", fontsize=10)
    ax3.set_title("WL S₈ Tension\n0% ξ-relief → physical Ω_m + σ₈", fontsize=10)
    ax3.legend(fontsize=7); ax3.grid(True, alpha=0.2, axis="x")

    # ── Panel D: Ω_m cross-probe ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    om_labels = ["BAO eBOSS", "WL Combined", "BAO+WL Joint", "Planck CMB"]
    om_vals   = [0.295, 0.2696, 0.2787, 0.3153]
    om_errs   = [0.010, 0.0075, 0.0060, 0.0073]
    om_colors = ["goldenrod", "purple", "darkgreen", "red"]
    om_markers= ["D", "^", "*", "P"]
    y4 = np.arange(len(om_labels))
    for i in range(len(om_labels)):
        ms = 14 if om_labels[i] in ("BAO+WL Joint","Planck CMB") else 10
        ax4.errorbar(om_vals[i], i, xerr=om_errs[i],
                     fmt=om_markers[i], color=om_colors[i], capsize=5,
                     linewidth=2, markersize=ms)
    ax4.axvline(0.3153, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    ax4.axvline(0.2787, color="darkgreen", linestyle=":", linewidth=2, alpha=0.7)
    ax4.set_yticks(y4); ax4.set_yticklabels(om_labels, fontsize=9)
    ax4.set_xlabel("Ω_m", fontsize=10)
    ax4.set_title("Ω_m Cross-Probe Synthesis\nAll probes converge below Planck", fontsize=10)
    ax4.grid(True, alpha=0.2, axis="x")
    ax4.set_xlim(0.24, 0.34)

    fig.suptitle(
        "UHA Cosmological Audit — Four-Probe Summary\n"
        "Eric D. Martin 2026 | Framework: UHA / N/U Algebra",
        fontsize=12, y=1.01
    )

    outpath = "/scratch/repos/hubble-tension-resolution/figures/audit_summary_4panel.png"
    plt.savefig(outpath, dpi=180, bbox_inches="tight")
    print(f"  4-panel audit summary → {outpath}")
    plt.close()
    return outpath


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Ω_m / σ₈ Synthesis Figures — UHA Audit")
    print("  Eric D. Martin | 2026-03-24")
    print("=" * 60)

    p1 = make_Om_figure()
    p2 = make_sig8_figure()
    p3 = make_audit_summary()

    print("\n  All figures generated:")
    print(f"    {p1}")
    print(f"    {p2}")
    print(f"    {p3}")
    print("\n  Task D complete.")
