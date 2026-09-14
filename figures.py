def generate_figures(df: pd.DataFrame) -> None:
    if not HAS_MPL:
        return
    figs_dir = ensure_fig_dir()

    import os
    import numpy as np
    import matplotlib.pyplot as plt

    figs_dir = ensure_fig_dir()

    if "ab_kind" not in df.columns:
        print("Validation figs skipped: no 'ab_kind' column in df")
        return

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    def viewpoint_of(bottleneck: str) -> str:
        # Example placeholder — replace with your actual mapping
        return bottleneck_viewpoint(bottleneck)

    # --- MAIN PLOT SECTION ---
    KEY = ["group", "bottleneck", "field"]

    intended = df[df["ab_kind"] == "INTENDED"].copy()
    integrated = df[df["ab_kind"] == "INTEGRATED"].copy()

    intended_mism = intended[intended["result"].str.lower() == "mismatch"]

    if intended_mism.empty:
        print("No INTENDED mismatches to validate.")
    else:
        mm = intended_mism.merge(
            integrated[KEY + ["result"]],
            on=KEY,
            how="left",
            suffixes=("_INT", "_INTG")
        )

        mm["_aligned"] = (mm["result_INTG"].str.lower() == "mismatch")

        # --- COLORS ---
        COLORS = {
            True: "#4daf4a",  # aligned → green
            False: "#e41a1c"  # incorrect → red
        }

        sns.set_theme(style="white")  # clean background, no grid
        plt.rcParams.update({
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "figure.dpi": 300,
            "savefig.dpi": 300
        })

        # --- Figure 1: Overall ---
        counts_overall = mm["_aligned"].value_counts().reindex([True, False], fill_value=0)
        labels = ["True Detection", "False Detection"]
        values = [int(counts_overall.get(True, 0)), int(counts_overall.get(False, 0))]
        colors = [COLORS[True], COLORS[False]]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.6)

        # Add value labels on top
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + max(values) * 0.02,
                    f"{int(height)}", ha="center", va="bottom", fontsize=12, weight="bold")

        ax.set_ylabel("Count", fontsize=13, weight="bold")
        ax.set_xlabel("")
        ax.set_title("Validation Against Realized Integrations", fontsize=15, weight="bold", pad=10)
        sns.despine(left=True, bottom=True)
        ax.grid(False)

        fig.tight_layout()
        fig.savefig(os.path.join(figs_dir, "validation_overall.png"), bbox_inches="tight", dpi=300)
        plt.close(fig)

        # --- Figure 2: By RM-ODP Viewpoint ---
        mm["_viewpoint"] = mm["bottleneck"].apply(viewpoint_of)
        piv = (mm.groupby("_viewpoint")["_aligned"]
               .value_counts()
               .unstack(fill_value=0)
               .reindex(columns=[True, False], fill_value=0))

        if not piv.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            idx = np.arange(len(piv.index))

            # Bar colors
            ax.bar(idx, piv[True].values, color=COLORS[True], label="True Detection", width=0.6, edgecolor="white")
            ax.bar(idx, piv[False].values, bottom=piv[True].values,
                   color=COLORS[False], label="False Detection", width=0.6, edgecolor="white")

            # Axis setup
            ax.set_xticks(idx)
            ax.set_xticklabels(piv.index, rotation=16, ha="right", fontsize=12)
            ax.set_ylabel("Number of Detected Mismatches (by Detector)", fontsize=12, weight="bold")
            ax.set_xlabel("")
            ax.set_title("", fontsize=15, weight="bold", pad=10)
            ax.legend(fontsize=16, frameon=False, loc="upper right")

            # Total value annotations
            for i, (a, b) in enumerate(zip(piv[True], piv[False])):
                total = a + b
                ax.text(i, total + max(piv.sum(axis=1)) * 0.02, f"{total}",
                        ha="center", va="bottom", fontsize=11)

            # ---- Clean but visible axis lines ----
            sns.despine(left=False, bottom=False)  # keep both x and y visible
            ax.spines["bottom"].set_color("black")
            ax.spines["left"].set_color("black")
            ax.spines["bottom"].set_linewidth(1.2)
            ax.spines["left"].set_linewidth(1.2)

            # Remove top/right borders
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # No gridlines
            ax.grid(False)

            fig.tight_layout()
            fig.savefig(os.path.join(figs_dir, "validation_by_viewpoint.png"),
                        bbox_inches="tight", dpi=300)
            plt.close(fig)

        print("✅ Saved validation figures in figs/:")
        print(" - validation_overall.png")
        print(" - validation_by_viewpoint.png")

    # --- define remap/merge/remove rules once, available everywhere ---
    def remap_field(f: str) -> Optional[str]:
        if f in {"file_formats", "(A or B).input vs AB.input", "A.input vs AB.input"}:
            return "Input mismatch"
        if f == "communication_mechanism":
            return None  # remove
        if f == "model_version":
            return None  # remove
        if f == "distribution_version":
            return None  # remove
        if f.startswith("availability_of_source_code"):
            return "Availability of source code"
        if f.startswith("landing_page"):
            return "Landing page"
        if f in {"B.output vs AB.output", "(A or B).output vs AB.output"}:
            return "Output"
        if f == "Direction (any)":
            return None  # remove
        return f  # keep others unchanged

    # ---------- canonical colors & orders ----------
    # softer, print-friendly palette
    COLORS = {
        "Match": "#4CAF50",  # professional green
        "Mismatch": "#E57373",  # soft red
        "Metadata Gap": "#FFB74D"  # orange
    }
    # softer categorical palette for arbitrary categories (patterns, fields, bottlenecks, etc.)
    SOFT_CMAP = [
        "#A6CEE3", "#B2DF8A", "#FB9A99", "#FDBF6F", "#CAB2D6",
        "#FFFF99", "#1F78B4", "#33A02C", "#E31A1C", "#FF7F00",
        "#6A3D9A", "#B15928", "#CCEBC5", "#FFED6F", "#B3CDE3",
        "#FBB4AE", "#DECBE4", "#FED9A6", "#FFFFCC", "#E5D8BD",
        "#FDDAEC", "#F2F2F2", "#80B1D3", "#B3DE69", "#FCCDE5",
        "#D9D9D9", "#BC80BD", "#CCEBC5", "#FFED6F", "#8DD3C7",
        "#FFFFB3", "#BEBADA", "#FB8072", "#80B1D3", "#FDB462",
        "#B3DE69", "#FCCDE5", "#D9D9D9", "#BC80BD", "#CCEBC5",
        "#FFED6F", "#FFB3A2", "#A0CBE8", "#FFD92F", "#E5C494",
        "#B3B3B3"
    ]
    import matplotlib
    plt.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=SOFT_CMAP)
    ORDER2 = ["Mismatch", "Metadata Gap"]
    ORDER3 = ["Match", "Mismatch", "Metadata Gap"]
    VP_ORDER = ["Domain", "Information", "Computational", "Engineering", "Technology"]
    #PATTERNS = ["One-Way", "Loose", "Shared", "Integrated", "Embedded", "(unspecified)"]
    PATTERNS = ["One-Way", "Loose", "Shared", "Integrated", "Embedded"]


    # ---------- helpers ----------
    def pie_percent(series: pd.Series, title: str, outpath: str, colors: Optional[List[str]] = None):
        if series.empty or series.sum() == 0:
            return
        fig = plt.figure()
        ax = fig.add_subplot(111)
        vals = series.values
        labels = series.index.tolist()
        if colors is None:
            ax.pie(vals, labels=labels, autopct=lambda p: f"{p:.1f}%")
        else:
            # truncate/align colors with labels length
            ax.pie(vals, labels=labels, autopct=lambda p: f"{p:.1f}%", colors=colors[:len(vals)])
        ax.set_title(title)
        fig.subplots_adjust(wspace=0.25, hspace=0.35)

        fig.tight_layout()
        fig.savefig(outpath, dpi=200)
        plt.close(fig)

    def stacked_bars(ax, piv: pd.DataFrame, order: List[str]):
        """Draw stacked bars in fixed color order."""
        idx = np.arange(len(piv.index))
        bottoms = np.zeros(len(idx))
        for col in order:
            vals = piv[col].values if col in piv.columns else np.zeros(len(idx))
            ax.bar(idx, vals, bottom=bottoms, label=col, color=COLORS.get(col))
            bottoms += vals
        ax.set_xticks(idx)
        ax.set_xticklabels(piv.index, rotation=0)

    def nice_ceil(x: float, step: int = 5) -> int:
        if x <= 0:
            return 1
        return int(np.ceil(x / step) * step)

    # ---------- derived labels ----------
    df["_viewpoint"] = df["bottleneck"].apply(bottleneck_viewpoint)
    df["_pattern"] = df["pattern"].fillna("(unspecified)").replace("", "(unspecified)")
    df["_is_mismatch"] = df["result"].str.lower().eq("mismatch")
    df["_is_gap"] = df["result"].str.lower().eq("missing")
    df["_is_problem"] = df["_is_mismatch"] | df["_is_gap"]
    df["_group"] = df.apply(
        lambda r: "Mismatch" if r["_is_mismatch"] else ("Metadata Gap" if r["_is_gap"] else "OK"), axis=1)
    df["_plot_group3"] = df["_group"].replace({"OK": "Match"})

    # ---------- 1) Overall detection rate (with 95% Wilson CI) ----------
    grp_any = df.groupby("group")["_is_problem"].any()
    n_cfg = int(len(grp_any)) if len(grp_any) else 0
    k_cfg = int(grp_any.sum()) if len(grp_any) else 0
    p_hat = (k_cfg / n_cfg) if n_cfg else 0.0
    lo, hi = wilson_ci(k_cfg, n_cfg) if n_cfg else (0.0, 0.0)
    err_low, err_high = p_hat - lo, hi - p_hat

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar([0], [p_hat], color="#88AADD")  # neutral color
    ax.errorbar([0], [p_hat], yerr=[[err_low], [err_high]], fmt="o", capsize=5)
    ax.set_ylim(0, 1)
    ax.set_xticks([0])
    ax.set_xticklabels(["Detection rate"])
    ax.set_ylabel("Share of configurations")
    ax.set_title(f"Overall detection rate (n={n_cfg}, 95% Wilson CI)")
    fig.tight_layout()
    fig.savefig(os.path.join(figs_dir, "overall_detection_rate.png"), dpi=200)
    plt.close(fig)

    # ---------- 2) Viewpoint × group (problems only; stacked) ----------
    pivot_counts = (df[df["_is_problem"]]
                    .groupby(["_viewpoint", "_group"])
                    .size()
                    .unstack(fill_value=0)
                    .reindex(index=VP_ORDER, fill_value=0)
                    .reindex(columns=ORDER2, fill_value=0))
    if not pivot_counts.empty:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        stacked_bars(ax, pivot_counts, ORDER2)
        ax.set_ylabel("Problem count")
        ax.set_title("Bottlenecks by RM-ODP viewpoint × group")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(figs_dir, "viewpoint_by_group_stacked.png"), dpi=200)
        plt.close(fig)

    # ---------- 3) Pattern sensitivity heatmap (pattern × group) ----------
    heat = (df[df["_is_problem"]]
            .groupby(["_pattern", "_group"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=PATTERNS, fill_value=0)
            .reindex(columns=ORDER2, fill_value=0))
    if not heat.empty:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(heat.values, aspect="auto")
        ax.set_xticks(np.arange(heat.shape[1]))
        ax.set_xticklabels(list(heat.columns))
        ax.set_yticks(np.arange(heat.shape[0]))
        ax.set_yticklabels(list(heat.index))
        ax.set_title("Pattern sensitivity: problems by pattern × group")
        for i in range(heat.shape[0]):
            for j in range(heat.shape[1]):
                ax.text(j, i, str(int(heat.values[i, j])), ha="center", va="center")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(figs_dir, "pattern_sensitivity_heatmap.png"), dpi=200)
        plt.close(fig)

    # ---------- 4) Field-level impact (Pareto on problems) ----------
    field_counts = (df[df["_is_problem"]].groupby("field").size().sort_values(ascending=False))
    if not field_counts.empty:
        top_k = min(15, len(field_counts))
        fc = field_counts.iloc[:top_k]
        cum = (fc.cumsum() / fc.sum()) if fc.sum() else fc * 0
        fig = plt.figure()
        ax = fig.add_subplot(111)
        idx = np.arange(len(fc.index))
        ax.bar(idx, fc.values, color="#88AADD")
        ax.set_xticks(idx)
        ax.set_xticklabels(fc.index, rotation=90)
        ax.set_ylabel("Problem count")
        ax.set_title("Field-level impact (Pareto: top-k fields)")
        ax2 = ax.twinx()
        ax2.plot(idx, cum.values, marker="o")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Cumulative share")
        fig.tight_layout()
        fig.savefig(os.path.join(figs_dir, "field_level_pareto.png"), dpi=200)
        plt.close(fig)

    # ---------- 5) Pies: mismatch/gap distributions ----------
    def counts_of(df_sub: pd.DataFrame, col: str) -> pd.Series:
        if df_sub.empty:
            return pd.Series(dtype=int)
        return df_sub.groupby(col).size().sort_values(ascending=False)

    mism = df[df["_is_mismatch"]]
    gaps = df[df["_is_gap"]]

    pie_percent(counts_of(mism, "_viewpoint"), "Mismatch by Viewpoint (%)",
                os.path.join(figs_dir, "pie_mismatch_by_viewpoint.png"),colors=SOFT_CMAP)
    pie_percent(counts_of(gaps, "_viewpoint"), "Metadata Gap by Viewpoint (%)",
                os.path.join(figs_dir, "pie_gap_by_viewpoint.png"),colors=SOFT_CMAP)

    pie_percent(counts_of(mism, "bottleneck"), "Mismatch by Bottleneck (%)",
                os.path.join(figs_dir, "pie_mismatch_by_bottleneck.png"),colors=SOFT_CMAP)
    pie_percent(counts_of(gaps, "bottleneck"), "Metadata Gap by Bottleneck (%)",
                os.path.join(figs_dir, "pie_gap_by_bottleneck.png"),colors=SOFT_CMAP)

    pie_percent(counts_of(mism, "_pattern"), "Mismatch by Integration Pattern (%)",
                os.path.join(figs_dir, "pie_mismatch_by_pattern.png"),colors=SOFT_CMAP)
    pie_percent(counts_of(gaps, "_pattern"), "Metadata Gap by Integration Pattern (%)",
                os.path.join(figs_dir, "pie_gap_by_pattern.png"),colors=SOFT_CMAP)

    pie_percent(counts_of(mism, "field"), "Mismatch by Field (%)",
                os.path.join(figs_dir, "pie_mismatch_by_field.png"),colors=SOFT_CMAP)
    pie_percent(counts_of(gaps, "field"), "Metadata Gap by Field (%)",
                os.path.join(figs_dir, "pie_gap_by_field.png"),colors=SOFT_CMAP)

    # ---------- 6) Facet: viewpoint × group per pattern (UNIFIED Y) ----------
    # pre-compute a global y-limit across panels
    y_max = 0
    pivots = {}
    for pat in PATTERNS:
        sub = df[(df["_pattern"] == pat) & (df["_is_problem"])]
        if sub.empty:
            continue
        piv = (sub.groupby(["_viewpoint", "_group"])
               .size()
               .unstack(fill_value=0)
               .reindex(index=VP_ORDER, fill_value=0)
               .reindex(columns=ORDER2, fill_value=0))
        pivots[pat] = piv
        y_max = max(y_max, int(piv.sum(axis=1).max()) if not piv.empty else 0)
    y_max = nice_ceil(y_max)

    n = len(PATTERNS)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.0, rows * 4.0), squeeze=False, sharey=True)
    any_plotted = False
    for i, pat in enumerate(PATTERNS):
        ax = axes[i // cols][i % cols]
        piv = pivots.get(pat)
        if piv is None or piv.empty:
            ax.axis("off")
            ax.set_title(f"{pat} (no problems)")
            continue
        stacked_bars(ax, piv, ORDER2)
        ax.set_ylabel("Count")
        ax.set_title(f"{pat} (n={int(piv.values.sum())})")
        ax.set_ylim(0, max(1, y_max) * 1.10)
        any_plotted = True
    for j in range(len(PATTERNS), rows * cols):
        axes[j // cols][j % cols].axis("off")
    if any_plotted:
        fig.legend(ORDER2, loc="upper center", ncol=2)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(os.path.join(figs_dir, "viewpoint_by_group_stacked_by_pattern.png"), dpi=200)
        plt.close(fig)

    # ---------- 7) Facet: per pattern, per-viewpoint Match/Mismatch/Gap (UNIFIED Y) ----------
    y_max3 = 0
    piv3 = {}
    for pat in PATTERNS:
        sub = df[df["_pattern"] == pat]
        if sub.empty:
            continue
        piv = (sub.groupby(["_viewpoint", "_plot_group3"])
               .size()
               .unstack(fill_value=0)
               .reindex(index=VP_ORDER, fill_value=0)
               .reindex(columns=ORDER3, fill_value=0))
        piv3[pat] = piv
        y_max3 = max(y_max3, int(piv.sum(axis=1).max()) if not piv.empty else 0)
    y_max3 = nice_ceil(y_max3)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 4.2), squeeze=False, sharey=True)
    any_plotted = False
    for i, pat in enumerate(PATTERNS):
        ax = axes[i // cols][i % cols]
        piv = piv3.get(pat)
        if piv is None or piv.empty:
            ax.axis("off")
            ax.set_title(f"{pat} (no rows)")
            continue
        idx = np.arange(len(piv.index))
        bottoms = np.zeros(len(idx))
        for col in ORDER3:
            ax.bar(idx, piv[col].values, bottom=bottoms, label=col, color=COLORS[col])
            bottoms += piv[col].values
        ax.set_xticks(idx)
        ax.set_xticklabels(piv.index, rotation=45, ha="right")
        ax.set_ylabel("Count")
        ax.set_title(f"{pat} (n={int(piv.values.sum())})")
        ax.set_ylim(0, max(1, y_max3) * 1.10)
        any_plotted = True
    for j in range(len(PATTERNS), rows * cols):
        axes[j // cols][j % cols].axis("off")
    if any_plotted:
        fig.legend(ORDER3, loc="upper center", ncol=3)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(os.path.join(figs_dir, "viewpoint_match_mismatch_gap_by_pattern.png"), dpi=200)
        plt.close(fig)

    # ---------- 8) Single chart: per-viewpoint Match/Mismatch/Gap ----------
    piv_vp = (df.groupby(["_viewpoint", "_plot_group3"])
              .size()
              .unstack(fill_value=0)
              .reindex(index=VP_ORDER, fill_value=0)
              .reindex(columns=ORDER3, fill_value=0))
    if not piv_vp.empty:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        idx = np.arange(len(piv_vp.index))
        bottoms = np.zeros(len(idx))
        for col in ORDER3:
            ax.bar(idx, piv_vp[col].values, bottom=bottoms, label=col, color=COLORS[col])
            bottoms += piv_vp[col].values
        ax.set_xticks(idx)
        ax.set_xticklabels(piv_vp.index, rotation=0)
        ax.set_ylabel("Count")
        ax.set_title("Per-viewpoint: Match vs Mismatch vs Metadata Gap")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(figs_dir, "viewpoint_match_mismatch_gap.png"), dpi=200)
        plt.close(fig)

    # ---------- 9) For each pattern: % Match / % Mismatch / % Metadata Gap (pie) ----------
    n = len(PATTERNS)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.0, rows * 4.5), squeeze=False)

    for i, pat in enumerate(PATTERNS):
        ax = axes[i // cols][i % cols]
        sub = df[df["_pattern"] == pat]
        if sub.empty:
            ax.axis("off")
            ax.set_title(f"{pat} (no rows)")
            continue

        counts = (sub["_plot_group3"].value_counts().reindex(ORDER3, fill_value=0))
        total = counts.sum()
        if total == 0:
            ax.axis("off")
            ax.set_title(f"{pat} (no rows)")
            continue

        ax.pie(counts.values,
               labels=counts.index,
               autopct=lambda p: f"{p:.1f}%",
               colors=[COLORS[k] for k in ORDER3])
        ax.set_title(f"{pat} (n={int(total)})")

    for j in range(len(PATTERNS), rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(figs_dir, "pie_match_mismatch_gap_by_pattern.png"), dpi=200)
    plt.close(fig)
    # --- NEW: mismatch-by-bottleneck pie with <4% grouped into "Other" ---
    def pie_percent_collapse_other(series: pd.Series, title: str, outpath: str, min_pct: float = 0.04):
        """Collapse categories whose share < min_pct into a single 'Other' slice."""
        if series.empty or series.sum() == 0:
            return
        series = series.sort_values(ascending=False)
        total = float(series.sum())
        pct = series / total
        small_mask = pct < min_pct
        other_count = series[small_mask].sum()
        series2 = series[~small_mask].copy()
        if other_count > 0:
            series2.loc["Other"] = other_count
            # keep 'Other' as the last slice for readability
            order = [i for i in series2.index if i != "Other"] + ["Other"]
            series2 = series2.reindex(order)
        # Reuse the existing pie helper
        pie_percent(series2, title, outpath)

    pie_percent_collapse_other(
        counts_of(mism, "bottleneck"),
        "Mismatch by Bottleneck (%) — small slices (<4%) grouped as Other",
        os.path.join(figs_dir, "pie_mismatch_by_bottleneck_other.png"),
        min_pct=0.04
    )

    def pie_percent_collapse_other(series: pd.Series, title: str, outpath: str, min_pct: float = 0.04):
        """
        Creates a publication-quality horizontal bar chart of category percentages,
        collapsing small categories (<min_pct) into 'Other'.

        Parameters
        ----------
        series : pd.Series
            Counts by category.
        title : str
            Chart title.
        outpath : str
            Path to save PNG/PDF figure.
        min_pct : float
            Minimum share threshold for retaining a category (default=0.04 → 4%).
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        # --- Safety check ---
        if series.empty or series.sum() == 0:
            print("⚠️ Warning: Empty or zero-sum data, no chart created.")
            return

        # --- Collapse small categories into "Other" ---
        series = series.sort_values(ascending=False)
        total = float(series.sum())
        pct = series / total
        small_mask = pct < min_pct
        other_count = series[small_mask].sum()

        series2 = series[~small_mask].copy()
        if other_count > 0:
            series2.loc["Other"] = other_count
            order = [i for i in series2.index if i != "Other"] + ["Other"]
            series2 = series2.reindex(order)

        # --- Normalize to percentages ---
        pct_series = (series2 / series2.sum()) * 100

        # --- Professional color palette (ColorBrewer Set2 – colorblind safe) ---
        COLORS_BOTTLENECK = [
            "#66c2a5",  # soft green
            "#fc8d62",  # coral
            "#8da0cb",  # soft blue
            "#e78ac3",  # pink/magenta
            "#a6d854",  # lime green
            "#ffd92f",  # amber/yellow
            "#e5c494",  # beige
            "#b3b3b3"  # gray for 'Other'
        ]
        colors = COLORS_BOTTLENECK[:len(pct_series)]

        # --- Plot setup ---
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.barh(
            pct_series.index[::-1],
            pct_series.values[::-1],
            color=colors[::-1],
            edgecolor="white"
        )

        # --- Add percentage labels on bars ---
        for bar, value in zip(bars, pct_series.values[::-1]):
            ax.text(
                value + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%",
                va="center",
                fontsize=12
            )

        # --- Axis labels and formatting ---
        ax.set_xlabel("Percentage (%)", fontsize=13, weight="bold")
        ax.set_ylabel("")
        ax.set_xlim(0, pct_series.max() * 1.25)
        ax.set_title(title, fontsize=15, weight="bold", pad=10)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="x", linestyle="--", alpha=0.6)
        sns.despine(left=True, bottom=True)

        # --- Final layout and save ---
        plt.tight_layout()
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ Saved bar chart to: {outpath}")

    # ----------------------------------------------------------------------
    # ✅ Call the function (same as before)
    # ----------------------------------------------------------------------
    pie_percent_collapse_other(
        counts_of(mism, "bottleneck"),
        "Mismatch by Bottleneck (%) — small slices (<4%) grouped as Other",
        os.path.join(figs_dir, "pie_mismatch_by_bottleneck_other2.png"),
        min_pct=0.04
    )

    import string
    import matplotlib.patches as mpatches

    # ---------- Pattern pies WITHOUT labels ----------
    # One pie per integration pattern; slices = [Match, Mismatch, Metadata Gap]; only % shown.
    n = len(PATTERNS)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.0, rows * 5.5), squeeze=False)

    for i, pat in enumerate(PATTERNS):
        ax = axes[i // cols][i % cols]
        sub = df[df["_pattern"] == pat]

        if sub.empty:
            ax.axis("off")
            continue

        counts = sub["_plot_group3"].value_counts().reindex(ORDER3, fill_value=0)
        total = counts.sum()

        if total == 0:
            ax.axis("off")
            continue

        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=None,  # no slice labels
            autopct=lambda p: f"{p:.1f}%",  # show only %
            colors=[COLORS[k] for k in ORDER3],
            textprops={"fontsize": 16, "weight": "bold"}
        )

        # Format percentages
        for t in autotexts:
            t.set_fontsize(16)
            t.set_weight("bold")

        # Title BELOW each pie (close distance)
        ax.text(
            0.5, -0.07, f"{string.ascii_lowercase[i]}) {pat}",
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=16, weight="bold"
        )

    # Create legend
    handles = [
        mpatches.Patch(color=COLORS["Match"], label="Match"),
        mpatches.Patch(color=COLORS["Mismatch"], label="Mismatch"),
        mpatches.Patch(color=COLORS["Metadata Gap"], label="Metadata Gap"),
    ]

    # Use empty slot for legend if available
    if len(PATTERNS) < rows * cols:
        ax_legend = axes[len(PATTERNS) // cols][len(PATTERNS) % cols]
        ax_legend.axis("off")
        ax_legend.legend(
            handles=handles,
            loc="center",
            frameon=True,
            fancybox=True,
            fontsize=16,
            title="Categories",
            title_fontsize=20
        )

    # Turn off unused subplot cells
    for j in range(len(PATTERNS) + 1, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.tight_layout()
    fig.savefig(
        os.path.join(figs_dir, "pie_match_mismatch_gap_by_pattern_nolabels.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)



    #new:

    def pie_percent_collapse_other(series: pd.Series, title: str, outpath: str, min_pct: float = 0.04):
        """
        Creates a publication-quality pie chart of category percentages,
        collapsing small categories (<min_pct) into 'Other'.

        Parameters
        ----------
        series : pd.Series
            Counts by category.
        title : str
            Chart title.
        outpath : str
            Path to save PNG/PDF figure.
        min_pct : float
            Minimum share threshold for retaining a category (default=0.04 → 4%).
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        # --- Safety check ---
        if series.empty or series.sum() == 0:
            print("⚠️ Warning: Empty or zero-sum data, no chart created.")
            return

        # --- Collapse small categories into "Other" ---
        series = series.sort_values(ascending=False)
        total = float(series.sum())
        pct = series / total
        small_mask = pct < min_pct
        other_count = series[small_mask].sum()

        series2 = series[~small_mask].copy()
        if other_count > 0:
            series2.loc["Other"] = other_count
            # Keep 'Other' as the last slice
            order = [i for i in series2.index if i != "Other"] + ["Other"]
            series2 = series2.reindex(order)

        # --- Professional color palette (ColorBrewer Set2 – colorblind-safe) ---
        COLORS_BOTTLENECK = [
            "#66c2a5",  # soft green
            "#fc8d62",  # coral
            "#8da0cb",  # soft blue
            "#e78ac3",  # pink/magenta
            "#a6d854",  # lime green
            "#ffd92f",  # amber/yellow
            "#e5c494",  # beige
            "#b3b3b3"  # gray for 'Other'
        ]
        colors = COLORS_BOTTLENECK[:len(series2)]

        # --- Compute percentages for labels ---
        pct_series = (series2 / series2.sum()) * 100

        # --- Plot setup ---
        sns.set_style("white")
        fig, ax = plt.subplots(figsize=(6, 6))

        wedges, texts, autotexts = ax.pie(
            pct_series,
            labels=series2.index,
            autopct=lambda p: f"{p:.1f}%",
            startangle=90,
            colors=colors,
            wedgeprops=dict(edgecolor="white", linewidth=1.5),
            textprops={"fontsize": 12, "color": "black"},
            pctdistance=0.8,
        )

        # --- Styling and title ---
        plt.setp(autotexts, size=12, weight="bold", color="black")
        ax.set_title(title, fontsize=15, weight="bold", pad=20)
        ax.axis("equal")  # equal aspect ratio for perfect circle
        plt.tight_layout()

        # --- Save high-quality figure ---
        fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"✅ Saved improved pie chart to: {outpath}")

    pie_percent_collapse_other(
        counts_of(mism, "bottleneck"),
        "Mismatch by Bottleneck (%) — small slices (<4%) grouped as Other",
        os.path.join(figs_dir, "pie_mismatch_by_bottleneck_other3.png"),
        min_pct=0.04
    )

    # ---------- NEW: Global color guidance (single legend image) ----------
    import matplotlib.patches as mpatches
    fig = plt.figure(figsize=(5.2, 1.3))
    ax = fig.add_subplot(111)
    ax.axis("off")
    handles = [
        mpatches.Patch(color=COLORS["Match"], label="Match"),
        mpatches.Patch(color=COLORS["Mismatch"], label="Mismatch"),
        mpatches.Patch(color=COLORS["Metadata Gap"], label="Metadata Gap"),
    ]
    # Centered, no frame, one row
    leg = ax.legend(handles=handles, loc="center", ncol=3, frameon=False, title="Color guide")
    fig.tight_layout()
    fig.savefig(os.path.join(figs_dir, "color_guide_match_mismatch_gap.png"), dpi=200)
    plt.close(fig)
    # ---------- NEW: For each viewpoint, normalized (by n_vp) stacked bars across patterns ----------
    # Each panel = one viewpoint; x = patterns; bars = [Match, Mismatch, Metadata Gap] / n_vp  ∈ [0,1]
    views = VP_ORDER  # ["Domain","Information","Computational","Engineering","Technology"]
    n_v = len(views)
    cols = 2
    rows = int(np.ceil(n_v / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.0, rows * 4.6), squeeze=False, sharey=True)

    for i, vp in enumerate(views):
        ax = axes[i // cols][i % cols]
        sub = df[df["_viewpoint"] == vp]
        n_total = int(sub.shape[0])

        if n_total == 0:
            ax.axis("off")
            ax.set_title(f"{vp} (no rows)")
            continue

        piv = (sub.groupby(["_pattern", "_plot_group3"])
                   .size()
                   .unstack(fill_value=0)
                   .reindex(index=PATTERNS, fill_value=0)      # ["One-Way","Loose","Shared","Integrated","Embedded","(unspecified)"]
                   .reindex(columns=ORDER3, fill_value=0))     # ["Match","Mismatch","Metadata Gap"]

        piv_norm = piv / float(n_total)  # <-- normalize by viewpoint total (e.g., 44) to get shares in [0,1]

        idx = np.arange(len(piv_norm.index))
        bottoms = np.zeros(len(idx))
        for col in ORDER3:
            ax.bar(idx, piv_norm[col].values, bottom=bottoms, color=COLORS[col])
            bottoms += piv_norm[col].values

        ax.set_xticks(idx)
        ax.set_xticklabels(piv_norm.index, rotation=45, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Share of viewpoint rows")
        ax.set_title(f"{vp} (n={n_total})")

    # turn off any extra empty subplots
    for j in range(n_v, rows * cols):
        axes[j // cols][j % cols].axis("off")

    # single legend for the whole grid (colors already established globally)
    fig.legend(ORDER3, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(figs_dir, "viewpoint_by_pattern_normalized.png"), dpi=200)
    plt.close(fig)
    import string
    import matplotlib.patches as mpatches

    # ---------- Per-viewpoint distribution within each pattern (shares sum to 1) ----------
    # For each viewpoint, show across patterns the share of Match/Mismatch/Metadata Gap **within** each pattern.
    views = VP_ORDER  # ["Domain","Information","Computational","Engineering","Technology"]
    n_v = len(views)
    cols = 2
    rows = int(np.ceil(n_v / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7.0, rows * 5.5), squeeze=False, sharey=True)

    for i, vp in enumerate(views):
        ax = axes[i // cols][i % cols]
        sub = df[df["_viewpoint"] == vp]

        if sub.empty:
            ax.axis("off")
            continue

        # counts per (pattern, result-class)
        piv = (
            sub.groupby(["_pattern", "_plot_group3"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=PATTERNS, fill_value=0)  # keep consistent pattern order
            .reindex(columns=ORDER3, fill_value=0)  # ["Match","Mismatch","Metadata Gap"]
        )

        # normalize **within each pattern** so each row sums to 1
        totals = piv.sum(axis=1).replace(0, np.nan)  # avoid div-by-zero
        piv_norm = piv.div(totals, axis=0).fillna(0.0)

        idx = np.arange(len(piv_norm.index))
        bottoms = np.zeros(len(idx))

        for col in ORDER3:  # fixed color mapping, no black border
            ax.bar(idx, piv_norm[col].values, bottom=bottoms, color=COLORS[col])
            bottoms += piv_norm[col].values

        ax.set_xticks(idx)
        ax.set_xticklabels(piv_norm.index, rotation=45, ha="right", fontsize=20)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Proportion of Match/ Mismatch/ Gap", fontsize=20, weight="bold")
        ax.tick_params(axis="y", labelsize=22)

        # Title BELOW each subplot with alphabetic label — slightly more distance
        ax.text(
            0.5, -0.18, f"{string.ascii_lowercase[i]}) {vp}",
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=25, weight="bold"
        )

    # Legend setup
    handles = [
        mpatches.Patch(color=COLORS["Match"], label="Match"),
        mpatches.Patch(color=COLORS["Mismatch"], label="Mismatch"),
        mpatches.Patch(color=COLORS["Metadata Gap"], label="Metadata Gap"),
    ]

    # If there’s an empty slot, use it for the legend
    if n_v < rows * cols:
        ax_legend = axes[n_v // cols][n_v % cols]
        ax_legend.axis("off")
        ax_legend.legend(
            handles=handles,
            loc="center",
            frameon=True,
            fancybox=True,
            fontsize=25,
            title="Categories",
            title_fontsize=25
        )

    # Turn off any other extra axes beyond the legend slot
    for j in range(n_v + 1, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(figs_dir, "viewpoint_by_pattern_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    import matplotlib.pyplot as plt
    import numpy as np
    import string
    import matplotlib.patches as mpatches
    import os

    # ---------- Per-viewpoint distribution within each pattern (shares sum to 1) ----------
    views = VP_ORDER  # ["Domain","Information","Computational","Engineering","Technology"]
    n_v = len(views)
    cols = 2
    rows = int(np.ceil(n_v / cols))

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * 7.0, rows * 5.8),
        squeeze=False,
        sharey=True
    )

    for i, vp in enumerate(views):
        ax = axes[i // cols][i % cols]
        sub = df[df["_viewpoint"] == vp]

        if sub.empty:
            ax.axis("off")
            continue

        # counts per (pattern, result-class)
        piv = (
            sub.groupby(["_pattern", "_plot_group3"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=PATTERNS, fill_value=0)
            .reindex(columns=ORDER3, fill_value=0)
        )

        # normalize within each pattern
        totals = piv.sum(axis=1).replace(0, np.nan)
        piv_norm = piv.div(totals, axis=0).fillna(0.0)

        idx = np.arange(len(piv_norm.index))
        bottoms = np.zeros(len(idx))

        for col in ORDER3:
            ax.bar(idx, piv_norm[col].values, bottom=bottoms, color=COLORS[col])
            bottoms += piv_norm[col].values

        ax.set_xticks(idx)
        ax.set_xticklabels(piv_norm.index, rotation=45, ha="right", fontsize=18)
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis="y", labelsize=18)

        # Hide y-axis label for all except the left column
        if i % cols != 0:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Proportion of Match/ Mismatch/ Gap", fontsize=18, weight="bold")

        # Title BELOW each subplot — moved farther down for spacing
        ax.text(
            0.5, -0.45, f"{string.ascii_lowercase[i]}) {vp}",
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=22, weight="bold"
        )

    # Legend setup
    handles = [
        mpatches.Patch(color=COLORS["Match"], label="Match"),
        mpatches.Patch(color=COLORS["Mismatch"], label="Mismatch"),
        mpatches.Patch(color=COLORS["Metadata Gap"], label="Metadata Gap"),
    ]

    # Use empty slot for legend if available
    if n_v < rows * cols:
        ax_legend = axes[n_v // cols][n_v % cols]
        ax_legend.axis("off")
        ax_legend.legend(
            handles=handles,
            loc="center",
            frameon=True,
            fancybox=True,
            fontsize=22,
            title="Categories",
            title_fontsize=22
        )

    # Turn off any remaining unused subplots
    for j in range(n_v + 1, rows * cols):
        axes[j // cols][j % cols].axis("off")

    # ---- Layout & Spacing Fixes ----
    fig.tight_layout(h_pad=2.0, w_pad=1.2)
    plt.subplots_adjust(left=0.08, bottom=0.32)  # left for shared ylabel, bottom for titles

    # Add a single shared y-label for the whole figure
    fig.text(
        0.02, 0.5,
        "Proportion of Match / Mismatch / Gap",
        va="center", rotation="vertical",
        fontsize=20, weight="bold"
    )

    # Save figure
    fig.savefig(
        os.path.join(figs_dir, "viewpoint_by_pattern_distribution.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    import string
    import matplotlib.patches as mpatches

    # ---------- Per-viewpoint distribution within each pattern (shares sum to 1) ----------
    # For each viewpoint, show across patterns the share of Match/Mismatch/Metadata Gap **within** each pattern.
    views = VP_ORDER  # ["Domain","Information","Computational","Engineering","Technology"]
    n_v = len(views)
    cols = 2
    rows = int(np.ceil(n_v / cols))

    # ↓ Decrease vertical size (y-axis height) by about half
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7.0, rows * 3.0), squeeze=False, sharey=True)

    for i, vp in enumerate(views):
        ax = axes[i // cols][i % cols]
        sub = df[df["_viewpoint"] == vp]

        if sub.empty:
            ax.axis("off")
            continue

        # counts per (pattern, result-class)
        piv = (
            sub.groupby(["_pattern", "_plot_group3"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=PATTERNS, fill_value=0)  # keep consistent pattern order
            .reindex(columns=ORDER3, fill_value=0)  # ["Match","Mismatch","Metadata Gap"]
        )

        # normalize **within each pattern** so each row sums to 1
        totals = piv.sum(axis=1).replace(0, np.nan)  # avoid div-by-zero
        piv_norm = piv.div(totals, axis=0).fillna(0.0)

        idx = np.arange(len(piv_norm.index))
        bottoms = np.zeros(len(idx))

        for col in ORDER3:  # fixed color mapping, no black border
            ax.bar(idx, piv_norm[col].values, bottom=bottoms, color=COLORS[col])
            bottoms += piv_norm[col].values

        ax.set_xticks(idx)
        ax.set_xticklabels(piv_norm.index, rotation=45, ha="right", fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Proportion of Match/Mismatch/Gap", fontsize=13, weight="bold")
        ax.tick_params(axis="y", labelsize=12)

        # ↓ Title BELOW each subplot with slightly smaller gap (closer to chart)
        ax.text(
            0.5, -0.12, f"{string.ascii_lowercase[i]}) {vp}",
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=14, weight="bold"
        )

    # ---------- Legend setup ----------
    handles = [
        mpatches.Patch(color=COLORS["Match"], label="Match"),
        mpatches.Patch(color=COLORS["Mismatch"], label="Mismatch"),
        mpatches.Patch(color=COLORS["Metadata Gap"], label="Metadata Gap"),
    ]

    # If there’s an empty slot, use it for the legend
    if n_v < rows * cols:
        ax_legend = axes[n_v // cols][n_v % cols]
        ax_legend.axis("off")
        ax_legend.legend(
            handles=handles,
            loc="center",
            frameon=True,
            fancybox=True,
            fontsize=14,
            title="Categories",
            title_fontsize=15
        )

    # Turn off any other extra axes beyond the legend slot
    for j in range(n_v + 1, rows * cols):
        axes[j // cols][j % cols].axis("off")

    # ↓ Reduce padding for tighter, publication-style layout
    fig.tight_layout(pad=1.0, rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(figs_dir, "viewpoint_by_pattern_distribution2.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------- NEW: For each integration pattern, pie of bottlenecks (top-5 + Other) ----------
    def pie_topk_with_other(counts: pd.Series, k: int = 5) -> pd.Series:
        """Keep top-k categories; collapse the rest into 'Other'."""
        if counts.empty:
            return counts
        counts = counts.sort_values(ascending=False)
        if len(counts) <= k:
            return counts
        top = counts.iloc[:k].copy()
        other = counts.iloc[k:].sum()
        if other > 0:
            top.loc["Other"] = other
        # put "Other" at the end
        order = [c for c in top.index if c != "Other"] + (["Other"] if "Other" in top.index else [])
        return top.reindex(order)

    # Use “bottlenecks” to mean problems (Mismatch + Metadata Gap)
    n = len(PATTERNS)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.4, rows * 4.6), squeeze=False)

    for i, pat in enumerate(PATTERNS):
        ax = axes[i // cols][i % cols]
        sub = df[(df["_pattern"] == pat) & (df["_is_problem"])]  # problems only
        if sub.empty:
            ax.axis("off")
            ax.set_title(f"{pat} (no bottlenecks)")
            continue

        counts = sub["bottleneck"].value_counts()
        series5 = pie_topk_with_other(counts, k=5)
        ax.pie(series5.values,
               labels=series5.index.tolist(),
               autopct=lambda p: f"{p:.1f}%")
        ax.set_title(f"{pat}")

    # turn off any extra cells
    for j in range(len(PATTERNS), rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(figs_dir, "pie_bottleneck_distribution_by_pattern.png"), dpi=200)
    plt.close(fig)

    # ---------- NEW: Per-viewpoint pies of Match / Mismatch / Metadata Gap (percentages) ----------
    n = len(VP_ORDER)  # ["Domain","Information","Computational","Engineering","Technology"]
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.0, rows * 4.2), squeeze=False)

    for i, vp in enumerate(VP_ORDER):
        ax = axes[i // cols][i % cols]
        sub = df[df["_viewpoint"] == vp]
        if sub.empty:
            ax.axis("off")
            ax.set_title(f"{vp} (no rows)")
            continue

        counts = (sub["_plot_group3"]            # values are "Match", "Mismatch", "Metadata Gap"
                    .value_counts()
                    .reindex(ORDER3, fill_value=0))  # keep fixed order/colors

        total = counts.sum()
        if total == 0:
            ax.axis("off")
            ax.set_title(f"{vp} (no rows)")
            continue

        ax.pie(
            counts.values,
            labels=None,  # keep pies clean; rely on the global color guide
            autopct=lambda p: f"{p:.1f}%",
            colors=[COLORS[k] for k in ORDER3],
            textprops={"fontsize": 9}
        )
        ax.set_title(vp)

    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color=COLORS["Match"], label="Match"),
        mpatches.Patch(color=COLORS["Mismatch"], label="Mismatch"),
        mpatches.Patch(color=COLORS["Metadata Gap"], label="Metadata Gap"),
    ]

    # If there’s an empty slot, put the legend there
    if n < rows * cols:
        ax_legend = axes[n // cols][n % cols]
        ax_legend.axis("off")
        ax_legend.legend(handles=handles, loc="center", frameon=True, fancybox=True)

    # Turn off any remaining spare cells (after the legend one)
    for j in range(n + 1, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(figs_dir, "pie_match_mismatch_gap_by_viewpoint.png"), dpi=200)
    plt.close(fig)

    # ---------- 4) Field-level impact (Pareto on problems, with merged/removals) ----------
    field_counts = (
        df[df["_is_problem"]]
        .groupby("field")
        .size()
        .sort_values(ascending=False)
    )

    if not field_counts.empty:
        # Initialize new mapping container
        merged_counts = {}

        for field, count in field_counts.items():
            # 1. Merge into "Input mismatch"
            if field in {"file_formats", "(A or B).input vs AB.input", "A.input vs AB.input"}:
                merged_counts["Input"] = merged_counts.get("Input mismatch", 0) + count

            # 2. Remove communication_mechanism
            elif field == "communication_mechanism":
                continue

            # 3. Remove model_version
            elif field == "model_version":
                continue

            # 4. Remove distribution_version
            elif field == "distribution_version":
                continue

            # 5. Merge availability_of_source_code A & B
            elif field in {"availability_of_source_code (A)", "availability_of_source_code (B)"}:
                merged_counts["Availability of source code"] = merged_counts.get("Availability of source code", 0) + count

            # 6. Merge landing_page A & B
            elif field in {"landing_page (A)", "landing_page (B)"}:
                merged_counts["Landing page"] = merged_counts.get("Landing page", 0) + count

            # 7. Merge outputs into "Output"
            elif field in {"B.output vs AB.output", "(A or B).output vs AB.output"}:
                merged_counts["Output"] = merged_counts.get("Output", 0) + count

            # 8. Remove Direction (any)
            elif field == "Direction (any)":
                continue

            # Otherwise keep field as is
            else:
                merged_counts[field] = merged_counts.get(field, 0) + count

        # Convert back to Series sorted descending
        field_counts2 = pd.Series(merged_counts).sort_values(ascending=False)

        # --- Pareto chart (top-k after merge/remove) ---
        top_k = min(15, len(field_counts2))
        fc = field_counts2.iloc[:top_k]
        cum = (fc.cumsum() / fc.sum()) if fc.sum() else fc * 0

        fig = plt.figure()
        ax = fig.add_subplot(111)
        idx = np.arange(len(fc.index))
        ax.bar(idx, fc.values, color="#88AADD")
        ax.set_xticks(idx)
        ax.set_xticklabels(fc.index, rotation=90)
        ax.set_ylabel("Problem count")
        ax.set_title("Field-level impact (Pareto: top-k fields)")
        ax2 = ax.twinx()
        ax2.plot(idx, cum.values, marker="o")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Cumulative share")
        fig.tight_layout()
        fig.savefig(os.path.join(figs_dir, "field_level_pareto.png"), dpi=200)
        plt.close(fig)

        # --- Horizontal bar chart (all fields merged/remapped) ---
        fig = plt.figure(figsize=(10, max(4, 0.35 * len(field_counts2))))
        ax = fig.add_subplot(111)
        y_pos = np.arange(len(field_counts2))
        ax.barh(y_pos, field_counts2.values, color="#88AADD")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(field_counts2.index)
        ax.invert_yaxis()  # largest on top
        ax.set_xlabel("Problem count")
        ax.set_ylabel("Field")
        ax.set_title("Problems by Field (merged categories)")
        fig.tight_layout()
        fig.savefig(os.path.join(figs_dir, "field_level_all_fields_horizontal.png"), dpi=200)
        plt.close(fig)

        # --- Mismatch-only (theta/pie chart) ---
        mismatch_counts = (
            df[df["_is_mismatch"]]
            .groupby("field")
            .size()
            .sort_values(ascending=False)
        )

        if not mismatch_counts.empty:
            # Apply the same remap/merge/remove rules
            mapped_mism = {}
            for k, v in mismatch_counts.items():
                newk = remap_field(k)
                if newk is None:
                    continue
                mapped_mism[newk] = mapped_mism.get(newk, 0) + v

            mismatch_counts2 = pd.Series(mapped_mism).sort_values(ascending=False)

            # --- Polar theta-style pie (mismatch only) ---

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="polar")
            values = mismatch_counts2.values
            labels = mismatch_counts2.index
            total = values.sum()
            angles = np.linspace(0, 2 * np.pi, len(values) + 1)

            ax.set_theta_direction(-1)  # clockwise
            ax.set_theta_offset(np.pi / 2.0)  # start at top

            bars = ax.bar(
                angles[:-1],
                values,
                width=(2 * np.pi / len(values)),
                bottom=0,
                align="edge",
                color="#F28E8C",
                edgecolor="white"
            )

            # Annotate labels
            for angle, val, lab in zip(angles[:-1], values, labels):
                ax.text(angle + (np.pi / len(values)) / 2,
                        val + max(values) * 0.05,
                        f"{lab}\n({val})",
                        ha="center", va="center", fontsize=8)

            ax.set_title("Mismatch by Field (polar view)", va="bottom")
            ax.set_yticklabels([])  # hide radial ticks

            fig.tight_layout()
            fig.savefig(os.path.join(figs_dir, "theta_mismatch_by_field.png"), dpi=200)
            plt.close(fig)




            # --- Mismatch-only horizontal bar chart ---
            # ---------- Improved: Mismatches by Field (Horizontal Bar Chart, Final Polished) ----------
            mismatch_counts = (
                df[df["_is_mismatch"]]
                .groupby("field")
                .size()
                .sort_values(ascending=False)
            )

            if not mismatch_counts.empty:
                # Apply the same remap/merge/remove rules
                mapped_mism = {}
                for k, v in mismatch_counts.items():
                    newk = remap_field(k)
                    if newk is None:
                        continue
                    mapped_mism[newk] = mapped_mism.get(newk, 0) + v

                mismatch_counts2 = pd.Series(mapped_mism).sort_values(ascending=False)

                import seaborn as sns
                sns.set_style("whitegrid")

                fig_height = max(4, 0.45 * len(mismatch_counts2))
                fig, ax = plt.subplots(figsize=(7.5, fig_height))

                # Consistent palette
                palette = sns.color_palette("Set2", n_colors=len(mismatch_counts2))
                y_pos = np.arange(len(mismatch_counts2))

                bars = ax.barh(y_pos, mismatch_counts2.values, color=palette, edgecolor="white")

                # Add value labels
                for bar, value in zip(bars, mismatch_counts2.values):
                    ax.text(
                        value + 0.5,
                        bar.get_y() + bar.get_height() / 2,
                        f"{value}",
                        va="center",
                        ha="left",
                        fontsize=11,
                        weight="bold",
                        color="black"
                    )

                # Axis formatting
                ax.set_yticks(y_pos)
                ax.set_yticklabels(mismatch_counts2.index, fontsize=11)
                ax.invert_yaxis()
                ax.set_xlabel("Mismatch Count", fontsize=12, weight="bold")
                ax.set_ylabel("")
                ax.set_xlim(0, mismatch_counts2.max() * 1.15)
                ax.xaxis.grid(True, linestyle="--", alpha=0.6)
                sns.despine(left=True, bottom=True)

                # Title
                ax.set_title("Mismatches by Metadata Field", fontsize=13, weight="bold", pad=10)

                # Save high-quality output
                fig.tight_layout(pad=0.5)
                fig.savefig(os.path.join(figs_dir, "bar_mismatch_by_field.png"),
                            dpi=300, bbox_inches="tight", pad_inches=0.02)
                plt.close(fig)

                print(f"✅ Saved improved figure (no warnings): {os.path.join(figs_dir, 'bar_mismatch_by_field.png')}")
