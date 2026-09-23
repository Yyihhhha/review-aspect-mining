# -*- coding: utf-8 -*-
"""
天池化妆品电商评论数据 —— 属性维度分析脚本 eda_category.py
==========================================================
输入（清洗后）：cleaned_reviews.csv、cleaned_labels.csv、review_wide.csv
输出：charts/05~08 四张 PNG（300 DPI）
      + category_polarity_table.csv（图7原始数据）
      + eda_category_summary.md（核心数据汇总）

全局配色约定（与图4视觉语言一致）：正面=绿、负面=红、中性=灰。
"""

import os
from itertools import combinations
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# 全局风格与中文字体
# ---------------------------------------------------------------------------
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="SimHei")
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 12

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(BASE_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)

# 极性配色：正面=绿、负面=红、中性=灰
POL_ORDER = ["正面", "负面", "中性"]
POL_COLOR = {"正面": "#5CB85C", "负面": "#D9534F", "中性": "#9E9E9E"}
IMPLICIT_TOKEN = "_"          # AspectTerms 用 "_" 表示隐式属性
GLOBAL_NEG_LINE = None        # 占位

# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
reviews = pd.read_csv(os.path.join(BASE_DIR, "cleaned_reviews.csv"), encoding="utf-8")
labels = pd.read_csv(os.path.join(BASE_DIR, "cleaned_labels.csv"), encoding="utf-8")
wide = pd.read_csv(os.path.join(BASE_DIR, "review_wide.csv"), encoding="utf-8")

labels["is_implicit"] = labels["AspectTerms"] == IMPLICIT_TOKEN
TOTAL_LABELS = len(labels)
GLOBAL_IMPLICIT_RATE = labels["is_implicit"].mean() * 100  # 约 71.4%


def announce(name, *frames):
    print(f"\n[{name}] 用到的数据形状：")
    for df, cols in frames:
        sub = df[cols] if cols else df
        print(f"    - {df.shape} 使用列: {list(sub.columns)}  行数={len(df)}")


# ---------------------------------------------------------------------------
# 图表 5：隐式属性 vs 显式属性的极性分布对比（分组柱状图）
# ---------------------------------------------------------------------------
def chart5():
    announce("图5 隐式vs显式极性分布", (labels, ["AspectTerms", "Polarities", "is_implicit"]))

    groups = {"隐式属性（AspectTerms=\"_\"）": labels[labels["is_implicit"]],
              "显式属性（点名属性）": labels[~labels["is_implicit"]]}

    stats = {}
    for gname, df in groups.items():
        total = len(df)
        vc = df["Polarities"].value_counts()
        stats[gname] = {p: int(vc.get(p, 0)) for p in POL_ORDER}
        stats[gname]["_total"] = total

    x = np.arange(len(groups))          # 两组
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 7))

    for k, pol in enumerate(POL_ORDER):
        offsets = x + (k - 1) * width
        counts = [stats[g][pol] for g in groups]
        pcts = [stats[g][pol] / stats[g]["_total"] * 100 for g in groups]
        bars = ax.bar(offsets, pcts, width, label=pol, color=POL_COLOR[pol],
                      edgecolor="white")
        for rect, pct, cnt, g in zip(bars, pcts, counts, groups):
            ax.text(rect.get_x() + rect.get_width() / 2, pct + 1.0,
                    f"{pct:.1f}%\n({cnt}/{stats[g]['_total']})",
                    ha="center", va="bottom", fontsize=9.5)

    ax.set_xticks(x)
    ax.set_xticklabels(list(groups.keys()), fontsize=12)
    ax.set_ylabel("占比（%）")
    ax.set_ylim(0, 100)
    ax.set_title("隐式属性 vs 显式属性的极性分布对比（正面=绿 / 负面=红 / 中性=灰）")
    ax.legend(title="极性", loc="upper right")

    # 图注：两组负面率对比
    neg_imp = stats[list(groups)[0]]["负面"] / stats[list(groups)[0]]["_total"] * 100
    neg_exp = stats[list(groups)[1]]["负面"] / stats[list(groups)[1]]["_total"] * 100
    ratio = neg_imp / neg_exp if neg_exp else float("inf")
    note = (f"隐式属性负面率 {neg_imp:.1f}%  vs  显式属性负面率 {neg_exp:.1f}%，"
            f"前者是后者的 {ratio:.1f} 倍")
    ax.text(0.5, -0.14, note, transform=ax.transAxes, ha="center",
            fontsize=11.5, color="#D9534F")

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out = os.path.join(CHART_DIR, "05_implicit_vs_explicit_polarity.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"    [OK] 已保存 {out}")
    return stats, neg_imp, neg_exp, ratio


# ---------------------------------------------------------------------------
# 图表 6：属性共现热力图
# ---------------------------------------------------------------------------
def chart6():
    announce("图6 属性共现热力图", (labels, ["ID", "Categories"]))
    MIN_TOTAL = 30      # 只保留总提及数 >= 30 的属性
    WEAK = 10           # 共现次数 < 10 视为弱共现

    cat_total = labels["Categories"].value_counts()
    kept = [c for c in cat_total.index if cat_total[c] >= MIN_TOTAL]
    kept = sorted(kept, key=lambda c: cat_total[c], reverse=True)

    # 每条评论涉及的（去重后）属性集合，仅保留 kept 内属性
    review_cats = (labels[labels["Categories"].isin(kept)]
                   .groupby("ID")["Categories"].apply(lambda s: sorted(set(s))))

    pair_cnt = defaultdict(int)
    for cats in review_cats:
        for a, b in combinations(cats, 2):
            pair_cnt[(a, b)] += 1

    n = len(kept)
    mat = pd.DataFrame(np.zeros((n, n)), index=kept, columns=kept)
    for (a, b), c in pair_cnt.items():
        mat.loc[a, b] = c
        mat.loc[b, a] = c
    np.fill_diagonal(mat.values, 0)

    vmax = mat.values.max() if mat.size else 0
    # 掩膜：对角线 + 弱共现(<WEAK) 显示为空白（白底）
    mask = (mat < WEAK)
    np.fill_diagonal(mask.values, True)

    fig, ax = plt.subplots(figsize=(max(9, n * 0.85), max(7.5, n * 0.8)))
    sns.heatmap(mat, mask=mask, cmap="Blues", annot=False, fmt=".0f",
                linewidths=0.6, linecolor="white", cbar_kws={"label": "共现次数"},
                vmax=vmax, ax=ax)

    # 手动标注所有格子
    for i in range(n):
        for j in range(n):
            v = int(mat.values[i, j])
            if i == j:
                ax.text(j + 0.5, i + 0.5, "—", ha="center", va="center",
                        color="#BBBBBB", fontsize=10)
            elif mask.values[i, j]:          # 弱共现：浅灰字
                ax.text(j + 0.5, i + 0.5, f"{v}", ha="center", va="center",
                        color="#AAAAAA", fontsize=9)
            else:                            # 强共现：深底白字 / 浅底深字
                txt_color = "white" if v > vmax * 0.6 else "#1F4E9C"
                ax.text(j + 0.5, i + 0.5, f"{v}", ha="center", va="center",
                        color=txt_color, fontsize=10, fontweight="bold")

    ax.set_title(f"属性共现热力图（同条评论共同出现次数，提及≥{MIN_TOTAL}）\n空白格 = 共现<{WEAK} 的弱共现或自共现", fontsize=13)
    ax.set_xlabel("属性")
    ax.set_ylabel("属性")
    fig.tight_layout()
    out = os.path.join(CHART_DIR, "06_category_cooccurrence_heatmap.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"    [OK] 已保存 {out}")
    top_pairs = sorted(pair_cnt.items(), key=lambda kv: kv[1], reverse=True)
    return top_pairs, kept, mat


# ---------------------------------------------------------------------------
# 图表 7：属性 × 极性交叉分析热力图（按行归一化配色）
# ---------------------------------------------------------------------------
def chart7():
    announce("图7 属性×极性交叉热力图", (labels, ["Categories", "Polarities"]))

    cross = (labels.pivot_table(index="Categories", columns="Polarities",
                                values="ID", aggfunc="count", fill_value=0)
             .reindex(columns=POL_ORDER, fill_value=0))
    cross["总提及数"] = cross.sum(axis=1)
    cross = cross.sort_values("总提及数", ascending=False)

    counts = cross[POL_ORDER].values.astype(float)
    row_total = counts.sum(axis=1, keepdims=True)
    prop = counts / row_total * 100            # 行归一化占比（用于配色）

    row_labels = cross.index.tolist()
    fig, ax = plt.subplots(figsize=(9, max(7, len(row_labels) * 0.55)))
    sns.heatmap(prop, cmap="GnBu", annot=False, fmt=".1f", linewidths=0.6,
                linecolor="white", cbar_kws={"label": "行内占比（%）"},
                xticklabels=POL_ORDER, yticklabels=row_labels, vmin=0, vmax=100, ax=ax)

    # 格子标注：数量 + 行内百分比
    for i in range(len(row_labels)):
        for j in range(len(POL_ORDER)):
            cnt = int(counts[i, j])
            pc = prop[i, j]
            txt_color = "white" if pc > 60 else "#333333"
            ax.text(j + 0.5, i + 0.5, f"{cnt}\n({pc:.1f}%)",
                    ha="center", va="center", fontsize=9.5, color=txt_color)

    ax.set_title("属性 × 极性交叉分析热力图（颜色按行归一化，展示各属性的极性结构）")
    ax.set_xlabel("极性")
    ax.set_ylabel("属性（按总提及数降序）")
    fig.tight_layout()
    out = os.path.join(CHART_DIR, "07_category_polarity_heatmap.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"    [OK] 已保存 {out}")

    # 输出原始数据 CSV（数量 + 行归一化占比）
    tbl = cross.copy()
    for p in POL_ORDER:
        tbl[f"{p}占比"] = (cross[p] / cross["总提及数"] * 100).round(1)
    csv_path = os.path.join(BASE_DIR, "category_polarity_table.csv")
    tbl.to_csv(csv_path, encoding="utf-8", index_label="Categories")
    print(f"    [OK] 交叉表已写入 {csv_path}")
    return cross


# ---------------------------------------------------------------------------
# 图表 8：各属性的隐式占比排序（横向条形图）
# ---------------------------------------------------------------------------
def chart8():
    announce("图8 各属性隐式占比", (labels, ["Categories", "AspectTerms", "is_implicit"]))

    grp = labels.groupby("Categories")["is_implicit"]
    stat = pd.DataFrame({
        "total": grp.size(),
        "implicit": grp.sum(),
    })
    stat["imp_rate"] = stat["implicit"] / stat["total"] * 100
    stat = stat.sort_values("imp_rate", ascending=True)   # barh 升序=顶部最高

    colors = ["#E8853B" if r > GLOBAL_IMPLICIT_RATE else "#1F6FB2"
              for r in stat["imp_rate"]]                   # 高于=橙，低于=蓝

    fig, ax = plt.subplots(figsize=(10, max(6.5, len(stat) * 0.5)))
    y = np.arange(len(stat))
    ax.barh(y, stat["imp_rate"].values, color=colors, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(stat.index.tolist())
    ax.set_xlim(0, 100)
    ax.set_xlabel("该属性中隐式（AspectTerms=\"_\"）占比（%）")
    ax.set_ylabel("属性种类")
    ax.set_title("各属性的隐式表达占比排序（橙=高于全局均值，蓝=低于）")

    for i, (rate, imp, tot) in enumerate(
            zip(stat["imp_rate"], stat["implicit"], stat["total"])):
        ax.text(rate + 1.0, i, f"{rate:.1f}%（{int(imp)}/{int(tot)}）",
                va="center", fontsize=10)

    ax.axvline(GLOBAL_IMPLICIT_RATE, color="#555555", linestyle="--", linewidth=1.6)
    ax.text(GLOBAL_IMPLICIT_RATE, len(stat) - 0.35,
            f" 全局隐式占比：{GLOBAL_IMPLICIT_RATE:.1f}%",
            color="#555555", fontsize=10.5, va="bottom")

    fig.tight_layout()
    out = os.path.join(CHART_DIR, "08_implicit_aspect_ratio.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"    [OK] 已保存 {out}")
    return stat.sort_values("imp_rate", ascending=False)


# ---------------------------------------------------------------------------
# 汇总报告
# ---------------------------------------------------------------------------
def write_summary(stats, neg_imp, neg_exp, ratio, top_pairs, cross, imp_stat):
    lines = []
    lines.append("# 属性维度分析总结（eda_category）\n")
    lines.append(f"> 数据源：清洗后化妆品电商评论数据集 | 总四元组 {TOTAL_LABELS} 个 | "
                 f"全局隐式占比 {GLOBAL_IMPLICIT_RATE:.1f}% | 全局平均负面率 "
                 f"{(labels['Polarities']=='负面').mean()*100:.1f}%\n")

    # 一、隐式 vs 显式
    lines.append("## 一、隐式 vs 显式属性的极性分布对比\n")
    lines.append("| 组别 | 总数 | 正面 | 负面 | 中性 | 负面率 |")
    lines.append("|------|------|------|------|------|--------|")
    for g, s in stats.items():
        neg_rate = s["负面"] / s["_total"] * 100
        lines.append(f"| {g} | {s['_total']} | {s['正面']} | {s['负面']} | "
                     f"{s['中性']} | {neg_rate:.1f}% |")
    lines.append("")
    lines.append(f"> **隐式属性负面率 {neg_imp:.1f}% vs 显式属性负面率 {neg_exp:.1f}%，"
                 f"前者是后者的 {ratio:.1f} 倍。**\n")

    # 二、共现 Top10
    lines.append("## 二、属性共现 Top10 组合（过滤共现<10 的弱组合）\n")
    strong = [(ab, c) for ab, c in top_pairs if c >= 10][:10]
    if strong:
        lines.append("| 排名 | 属性对 | 共现次数 |")
        lines.append("|------|--------|---------|")
        for rank, ((a, b), c) in enumerate(strong, 1):
            lines.append(f"| {rank} | {a} + {b} | {c} |")
    else:
        lines.append("_无共现次数 ≥ 10 的属性对。_")
    lines.append("")

    # 三、属性 × 极性交叉表
    lines.append("## 三、属性 × 极性交叉表（数量 + 行归一化占比）\n")
    lines.append("| 属性 | 总提及数 | 正面 | 负面 | 中性 | 正面占比 | 负面占比 | 中性占比 |")
    lines.append("|------|---------|------|------|------|---------|---------|---------|")
    for cat, row in cross.iterrows():
        tot = int(row["总提及数"])
        fp = row["正面"] / tot * 100
        fn = row["负面"] / tot * 100
        fz = row["中性"] / tot * 100
        lines.append(f"| {cat} | {tot} | {int(row['正面'])} | {int(row['负面'])} | "
                     f"{int(row['中性'])} | {fp:.1f}% | {fn:.1f}% | {fz:.1f}% |")
    lines.append("\n> 原始数据另存于 `category_polarity_table.csv`。\n")

    # 四、各属性隐式占比排序（全量）
    lines.append("## 四、各属性隐式占比排序（全量）\n")
    lines.append("| 排名 | 属性 | 隐式数 | 总提及数 | 隐式占比 | 相对全局 |")
    lines.append("|------|------|--------|---------|---------|---------|")
    for rank, (cat, row) in enumerate(imp_stat.iterrows(), 1):
        flag = "↑ 高于均值" if row["imp_rate"] > GLOBAL_IMPLICIT_RATE else "↓ 低于均值"
        lines.append(f"| {rank} | {cat} | {int(row['implicit'])} | {int(row['total'])} | "
                     f"{row['imp_rate']:.1f}% | {flag} |")
    lines.append("")

    # 五、一句话结论
    top3_imp = list(imp_stat.index[:3])
    top3_imp_str = "、".join(f"{c}({imp_stat.loc[c,'imp_rate']:.1f}%)" for c in top3_imp)
    top3_pair_str = "、".join(f"{a}+{b}({c})" for (a, b), c in top_pairs[:3])
    lines.append("## 五、一句话结论\n")
    lines.append(
        f"- **隐式占比最高的 3 个属性**：{top3_imp_str}；\n"
        f"- **共现最强的 3 个属性对**：{top3_pair_str}；\n"
        f"- **隐式 vs 显式负面率差异**：隐式 {neg_imp:.1f}% vs 显式 {neg_exp:.1f}%"
        f"（隐式为显式的 {ratio:.1f} 倍）——"
        f"{'用户不点名属性时吐槽更狠，隐式表达承载了更多负面情绪。' if neg_imp>neg_exp else '点名属性时反而更集中表达负面。'}"
    )
    lines.append("")

    md = "\n".join(lines)
    md_path = os.path.join(BASE_DIR, "eda_category_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\n[OK] 汇总报告已写入 {md_path}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    stats, neg_imp, neg_exp, ratio = chart5()
    top_pairs, kept, co_mat = chart6()
    cross = chart7()
    imp_stat = chart8()
    write_summary(stats, neg_imp, neg_exp, ratio, top_pairs, cross, imp_stat)
    print("\n" + "=" * 60)
    print(f"[完成] 图表 5~8 已生成于 {CHART_DIR}")


if __name__ == "__main__":
    main()
