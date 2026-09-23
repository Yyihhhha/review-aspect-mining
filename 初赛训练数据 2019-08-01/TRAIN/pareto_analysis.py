# -*- coding: utf-8 -*-
"""
天池化妆品电商评论数据 —— 差评帕累托分析脚本 pareto_analysis.py
================================================================
目标：找出“解决了就能消除大部分差评”的核心问题点（帕累托 80/20）。

数据源：
  - bad_review_subcategories.csv（具体子类问题点计数）
  - cleaned_labels.csv（属性维度负面总数；并复算“其他”高频词）
  - opinion_word_freq.csv（辅助，未直接用于筛选）

问题点单位 = 属性 + 子类；排除“其他/隐式负面/通用差评”三类聚合类；
但把各属性“其他”中出现 >= 3 次的具体词提升为独立问题点。

输出：charts/16、charts/17（300 DPI）+ pareto_analysis.md，均无 BOM UTF-8。
"""

import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import bad_review_subcategory as brs   # 复用归类规则，保证口径一致

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

CORE_COLOR = "#C0392B"   # 80% 以内的核心问题点（红）
TAIL_COLOR = "#BFBFBF"   # 长尾问题点（灰）
LINE_COLOR = "#1F4E9C"   # 累计占比折线（蓝）

EXCLUDE = {"其他", "隐式负面", "通用差评"}
MIN_OTHER_PROMOTE = 3    # “其他”里词频 >= 3 提升为独立问题点


# ---------------------------------------------------------------------------
# 构建问题点清单
# ---------------------------------------------------------------------------
def build_problem_points():
    csv = pd.read_csv(os.path.join(BASE_DIR, "bad_review_subcategories.csv"), encoding="utf-8")
    lab = pd.read_csv(os.path.join(BASE_DIR, "cleaned_labels.csv"), encoding="utf-8")
    neg = lab[lab["Polarities"] == "负面"]

    points = []   # (标签, 属性, 子类/词, 负面数, 来源)
    # 1) 具体子类
    spec = csv[~csv["子类"].isin(EXCLUDE)]
    for _, r in spec.iterrows():
        points.append({"label": f"{r['属性']}-{r['子类']}", "属性": r["属性"],
                       "子类": r["子类"], "count": int(r["负面数"]), "来源": "具体子类"})

    # 2) 各属性“其他”里 >= MIN_OTHER_PROMOTE 的高频词
    for attr in brs.ATTR_ORDER:
        rules = brs.RULES[attr] + [(brs.GENERIC, brs.GENERIC_KWS)]
        cnt = Counter()
        for x in neg[neg["Categories"] == attr]["OpinionTerms"]:
            sc, _ = brs.classify(x, rules)
            if sc == brs.OTHER:
                cnt[str(x).strip()] += 1
        for w, c in cnt.items():
            if c >= MIN_OTHER_PROMOTE:
                points.append({"label": f"{attr}-{w}", "属性": attr,
                               "子类": w, "count": c, "来源": "其他高频词提升"})

    df = pd.DataFrame(points).sort_values("count", ascending=False).reset_index(drop=True)
    total = df["count"].sum()
    df["占比"] = df["count"] / total * 100
    df["累计占比"] = df["占比"].cumsum()
    return df, total, neg


# ---------------------------------------------------------------------------
# 图表 16：问题点维度帕累托图
# ---------------------------------------------------------------------------
def chart16(df):
    n80 = int((df["累计占比"] < 80).sum()) + 1     # 达到 80% 所需前 N 个
    colors = [CORE_COLOR if i < n80 else TAIL_COLOR for i in range(len(df))]
    x = np.arange(len(df))

    fig, ax1 = plt.subplots(figsize=(13, 7.5))
    ax1.bar(x, df["count"], color=colors, edgecolor="white")
    ax1.set_ylabel("负面数（个）")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["label"], rotation=40, ha="right", fontsize=10)
    ax1.set_ylim(0, df["count"].max() * 1.18)
    for i, (c, pct) in enumerate(zip(df["count"], df["占比"])):
        ax1.text(i, c + 0.6, f"{c}\n{pct:.1f}%", ha="center", va="bottom", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(x, df["累计占比"], color=LINE_COLOR, marker="o", linewidth=2)
    for i, cum in enumerate(df["累计占比"]):
        ax2.annotate(f"{cum:.0f}%", (i, cum), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=9, color=LINE_COLOR)
    ax2.axhline(80, color="#E67E22", linestyle="--", linewidth=1.6)
    ax2.text(len(df) - 0.4, 81, "80%", color="#E67E22", fontsize=11, ha="right")
    ax2.set_ylabel("累计占比（%）")
    ax2.set_ylim(0, 108)

    ax1.set_title(f"问题点维度帕累托图（共 {len(df)} 个问题点，合计 {int(df['count'].sum())} 个可归类差评）\n"
                  f"红色=前 {n80} 个核心问题点（累计达 {df['累计占比'].iloc[n80-1]:.1f}% ≥ 80%）")
    fig.tight_layout()
    out = os.path.join(CHART_DIR, "16_pareto_problem_points.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[OK] 图16已保存 {out}")
    return n80


# ---------------------------------------------------------------------------
# 图表 17：属性维度帕累托图（全部属性，基于 cleaned_labels 负面总数）
# ---------------------------------------------------------------------------
def chart17(neg):
    cat = neg["Categories"].value_counts()
    total = cat.sum()
    pct = cat / total * 100
    cum = pct.cumsum()
    n80 = int((cum < 80).sum()) + 1
    x = np.arange(len(cat))
    colors = [CORE_COLOR if i < n80 else TAIL_COLOR for i in range(len(cat))]

    fig, ax1 = plt.subplots(figsize=(13, 7.5))
    ax1.bar(x, cat.values, color=colors, edgecolor="white")
    ax1.set_ylabel("负面数（个）")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cat.index, rotation=40, ha="right", fontsize=10)
    ax1.set_ylim(0, cat.max() * 1.18)
    for i, (c, p) in enumerate(zip(cat.values, pct.values)):
        ax1.text(i, c + 1.5, f"{c}\n{p:.1f}%", ha="center", va="bottom", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(x, cum.values, color=LINE_COLOR, marker="o", linewidth=2)
    for i, cc in enumerate(cum.values):
        ax2.annotate(f"{cc:.0f}%", (i, cc), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=9, color=LINE_COLOR)
    ax2.axhline(80, color="#E67E22", linestyle="--", linewidth=1.6)
    ax2.text(len(cat) - 0.4, 81, "80%", color="#E67E22", fontsize=11, ha="right")
    ax2.set_ylabel("累计占比（%）")
    ax2.set_ylim(0, 108)

    ax1.set_title(f"属性维度帕累托图（全部 {len(cat)} 类属性，合计 {total} 个负面四元组）\n"
                  f"红色=前 {n80} 个属性（累计达 {cum.values[n80-1]:.1f}% ≥ 80%）")
    fig.tight_layout()
    out = os.path.join(CHART_DIR, "17_pareto_categories.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[OK] 图17已保存 {out}")
    return n80, cat


# ---------------------------------------------------------------------------
# P0 业务改进方向（基于关键词推断）
# ---------------------------------------------------------------------------
BIZ = {
    "使用体验-质地问题": "优化配方质地与保湿力，按干皮/油皮分型出清爽/滋润版本；细化粉体粒径，改善卡粉、浮粉、拔干。",
    "包装-包装简陋/随意": "升级外包装盒与缓冲防护材料，规范塑封/封口，杜绝“简陋、随便、旧”的开箱观感。",
    "气味-香味过重/刺鼻": "下调香精浓度并提供无香/淡香版本，改用更温和香型，避免刺鼻、酒精味过重。",
    "使用体验-效果问题": "强化功效实证与成分安全测试，明确标注适用人群与致敏成分，减少“没效果/过敏/长痘”。",
}


def write_report(df, total, n80, n80_cat, cat_series, neg_total_all):
    L = []
    L.append("# 差评帕累托分析（pareto_analysis）\n")
    L.append(f"> 问题点单位=属性+子类；排除“其他/隐式负面/通用差评”聚合类，但将“其他”中"
             f"出现≥{MIN_OTHER_PROMOTE}次的词提升为独立问题点。"
             f"共 **{len(df)} 个问题点**，合计 **{total} 个可归类差评**（占全部 {neg_total_all} 个负面的 {total/neg_total_all*100:.1f}%）。\n")

    # 一、核心结论
    L.append("## 一、核心结论\n")
    top_share = df["占比"].iloc[0]
    n80_share = n80 / len(df) * 100
    L.append(f"- 前 **{n80}** 个问题点（占全部问题点的 **{n80_share:.1f}%**）贡献了 **80%** 的可归类差评；")
    L.append(f"- 但头部 1 个问题点“{df['label'].iloc[0]}”独占 **{top_share:.1f}%**，集中度极高；")
    verdict = "符合" if n80_share <= 30 else "不完全符合（头部有单一超集中问题，但整体较分散）"
    L.append(f"- 判定：{'典型' if n80_share<=25 else ''}帕累托 80/20 —— {verdict}。\n")

    # 二、优先级清单
    L.append("## 二、优先解决问题清单（按 P0/P1/P2）\n")
    p0 = df[df["占比"] >= 10]
    p1 = df[(df["占比"] >= 5) & (df["占比"] < 10)]
    p2 = df[df["占比"] < 5]
    L.append("### P0（解决 1 个即可消除 10%+ 差评）\n")
    L.append("| 优先级 | 问题点 | 负面数 | 占可归类差评 |")
    L.append("|--------|--------|--------|-------------|")
    for _, r in p0.iterrows():
        L.append(f"| P0 | {r['label']} | {r['count']} | {r['占比']:.1f}% |")
    L.append("")
    L.append("### P1（解决 1 个可消除 5%~10% 差评）\n")
    for _, r in p1.iterrows():
        L.append(f"- {r['label']}：{r['count']} 个（{r['占比']:.1f}%）")
    L.append("")
    L.append(f"### P2（长尾问题点）\n\n共 **{len(p2)} 个**长尾问题点，单个占比均 < 5%，合计 "
             f"{p2['占比'].sum():.1f}%（{int(p2['count'].sum())} 个差评），暂不单独投入资源。\n")

    # 三、P0 业务改进方向
    L.append("## 三、P0 问题点的业务改进方向\n")
    for _, r in p0.iterrows():
        tip = BIZ.get(r["label"], "结合该问题点的高频观点词，针对性优化产品/包装/服务流程。")
        L.append(f"- **{r['label']}**（{r['count']} 个，{r['占比']:.1f}%）：{tip}")
    L.append("")

    # 四、属性维度
    L.append("## 四、属性维度帕累托（补充视角）\n")
    L.append(f"- 按属性看，前 **{n80_cat}** 个属性（{('、'.join(cat_series.index[:n80_cat]))}）"
             f"合计贡献了 80% 的全部差评（{neg_total_all} 个）；")
    L.append(f"- 说明差评高度集中在“使用体验”这一属性上（单属性 {int(cat_series.iloc[0])} 个，"
             f"占全部差评 {cat_series.iloc[0]/neg_total_all*100:.1f}%）。\n")

    # 五、一句话结论
    L.append("## 五、一句话结论\n")
    L.append(f"> **解决前 {n80} 个问题点，就能消除 {df['累计占比'].iloc[n80-1]:.1f}% 的可归类差评**；"
             f"其中优先攻克“{df['label'].iloc[0]}”单点即可覆盖 {top_share:.1f}%。")
    L.append("")

    md = "\n".join(L)
    md_path = os.path.join(BASE_DIR, "pareto_analysis.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[OK] 报告已写入 {md_path}")


def main():
    df, total, neg = build_problem_points()
    print(f"[数据] 问题点 {len(df)} 个，合计可归类差评 {total} 个")
    n80 = chart16(df)
    neg_total_all = int((neg["Polarities"] == "负面").sum()) if "Polarities" in neg else len(neg)
    n80_cat, cat_series = chart17(neg)
    write_report(df, total, n80, n80_cat, cat_series, len(neg))


if __name__ == "__main__":
    main()
