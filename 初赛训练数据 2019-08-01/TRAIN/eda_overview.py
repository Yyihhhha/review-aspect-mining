# -*- coding: utf-8 -*-
"""
天池化妆品电商评论数据 EDA 可视化脚本 eda_overview.py
====================================================
输入（清洗后）：cleaned_reviews.csv、cleaned_labels.csv、review_wide.csv
输出：charts/01~04 四张 PNG（300 DPI）+ eda_overview_summary.md
统一：matplotlib + seaborn，白色网格底、SimHei 中文字体、数据标注齐全。
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

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

# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
reviews = pd.read_csv(os.path.join(BASE_DIR, "cleaned_reviews.csv"), encoding="utf-8")
labels = pd.read_csv(os.path.join(BASE_DIR, "cleaned_labels.csv"), encoding="utf-8")
wide = pd.read_csv(os.path.join(BASE_DIR, "review_wide.csv"), encoding="utf-8")

TOTAL_LABELS = len(labels)


def announce(name, *frames):
    print(f"\n[{name}] 用到的数据形状：")
    for df, cols in frames:
        sub = df[cols] if cols else df
        print(f"    - {df.shape} 使用列: {list(sub.columns)}  行数={len(df)}")


# ---------------------------------------------------------------------------
# 图表 1：评论长度分布（直方图 + KDE）
# ---------------------------------------------------------------------------
def chart1():
    lengths = wide["评论字数"].dropna().values
    announce("图1 评论长度分布", (wide, ["评论字数"]))

    mean_len = np.mean(lengths)
    median_len = np.median(lengths)
    max_len, min_len = np.max(lengths), np.min(lengths)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    bins = np.arange(0, 201, 20)  # 0-200，20 字分桶

    sns.histplot(lengths, bins=bins, kde=False, color="#4C8EDE",
                 edgecolor="white", ax=ax1)
    ax1.set_xlabel("评论字数（字）")
    ax1.set_ylabel("评论数量（条）")
    ax1.set_xlim(0, 200)
    ax1.set_xticks(bins)

    # 右侧 KDE 密度曲线
    ax2 = ax1.twinx()
    sns.kdeplot(lengths, color="#E8663C", linewidth=2.2, ax=ax2)
    ax2.set_ylabel("密度（KDE）", color="#E8663C")
    ax2.tick_params(axis="y", labelcolor="#E8663C")

    # 均值 / 中位数虚线
    ax1.axvline(mean_len, color="#D62728", linestyle="--", linewidth=1.6,
                label=f"平均长度 {mean_len:.1f}")
    ax1.axvline(median_len, color="#2CA02C", linestyle="--", linewidth=1.6,
                label=f"中位数 {median_len:.0f}")

    # 角落文本框
    stat_text = (f"均值: {mean_len:.1f} 字\n"
                 f"中位数: {median_len:.0f} 字\n"
                 f"最大值: {max_len:.0f} 字\n"
                 f"最小值: {min_len:.0f} 字\n"
                 f"样本数: {len(lengths)}")
    ax1.text(0.985, 0.95, stat_text, transform=ax1.transAxes,
             va="top", ha="right", fontsize=10.5,
             bbox=dict(boxstyle="round", facecolor="#F5F7FA",
                       edgecolor="#B0BEC5", alpha=0.9))

    ax1.legend(loc="upper left", frameon=True)
    ax1.set_title("评论长度分布（直方图 + KDE 密度）")
    fig.tight_layout()
    out = os.path.join(CHART_DIR, "01_review_length_dist.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"    [OK] 已保存 {out}")
    return dict(mean=mean_len, median=median_len, max=max_len,
                min=min_len, n=len(lengths))


# ---------------------------------------------------------------------------
# 图表 2：每条评论的四元组数量分布（柱状图，渐变蓝）
# ---------------------------------------------------------------------------
def chart2():
    counts = wide["四元组数量"].dropna()
    counts = counts[counts >= 1]  # 至少含 1 个四元组的评论
    announce("图2 每条评论四元组数量分布", (wide, ["四元组数量"]))

    max_q = int(counts.max())
    total = len(counts)
    buckets = [("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4", 4, 4)]
    if max_q >= 5:
        buckets.append(("5及以上", 5, max_q))

    labels_x, values = [], []
    for name, lo, hi in buckets:
        cnt = int(((counts >= lo) & (counts <= hi)).sum())
        labels_x.append(name)
        values.append(cnt)

    # 渐变蓝：数量越多颜色越深
    cmap = plt.get_cmap("Blues")
    vmax = max(values)
    colors = [cmap(0.35 + 0.55 * v / vmax) for v in values]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels_x, values, color=colors, edgecolor="white")
    for b, v in zip(bars, values):
        pct = v / total * 100
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=10.5)

    title_note = ""
    if max_q >= 5:
        title_note = f"（5及以上：实际最大 {max_q} 个）"
    ax.set_title(f"每条评论的四元组数量分布 {title_note}")
    ax.set_xlabel("四元组数量")
    ax.set_ylabel("评论条数")
    ax.set_ylim(0, vmax * 1.18)
    fig.tight_layout()
    out = os.path.join(CHART_DIR, "02_quadruple_count_dist.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"    [OK] 已保存 {out}")
    return dict(zip(labels_x, values)), total, max_q


# ---------------------------------------------------------------------------
# 图表 3：属性种类分布（横向条形图，动态分层配色）
# ---------------------------------------------------------------------------
def chart3():
    announce("图3 属性种类分布", (labels, ["Categories"]))
    cat_counts = labels["Categories"].value_counts()
    total = cat_counts.sum()
    pct = cat_counts / total * 100

    # 按占比动态分层（不硬编码前 3 名）
    def color_of(p):
        if p > 5:
            return "#1F4E9C"   # 头部 深蓝
        elif p >= 1:
            return "#6FA8DC"   # 腰部 中蓝
        else:
            return "#C9CDD4"   # 长尾 浅灰

    colors = [color_of(p) for p in pct]

    fig, ax = plt.subplots(figsize=(10, 7))
    order = cat_counts.index.tolist()  # value_counts 已降序
    y = np.arange(len(order))[::-1]
    bars = ax.barh(y, cat_counts.values, color=colors, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel("提及次数（个）")
    ax.set_ylabel("属性种类")
    ax.set_xlim(0, cat_counts.max() * 1.18)

    for yi, (cnt, p) in enumerate(zip(cat_counts.values, pct.values)):
        ax.text(cnt + cat_counts.max() * 0.01, y[yi],
                f"{cnt}（{p:.1f}%）", va="center", fontsize=10)

    legend_handles = [
        Patch(facecolor="#1F4E9C", label="头部（占比>5%）"),
        Patch(facecolor="#6FA8DC", label="腰部（占比1%~5%）"),
        Patch(facecolor="#C9CDD4", label="长尾（占比<1%）"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", title="占比分层",
              frameon=True)
    ax.set_title("属性种类分布（按提及次数排序）")
    fig.tight_layout()
    out = os.path.join(CHART_DIR, "03_category_distribution.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"    [OK] 已保存 {out}")
    return cat_counts, pct


# ---------------------------------------------------------------------------
# 图表 4：各属性的负面评价占比排序（横向条形图）
# ---------------------------------------------------------------------------
def chart4():
    announce("图4 各属性负面占比", (labels, ["Categories", "Polarities"]))
    MIN_TOTAL = 80  # 主图仅展示总提及数 >= 80 的属性（服务 86 纳入主图）
    grp = labels.groupby("Categories")["Polarities"]
    stat = pd.DataFrame({
        "total": grp.size(),
        "neg": grp.apply(lambda s: (s == "负面").sum()),
    })
    stat["neg_rate"] = stat["neg"] / stat["total"] * 100

    # 全局平均负面率
    global_neg_rate = (labels["Polarities"] == "负面").sum() / TOTAL_LABELS * 100

    # 主图（>= MIN_TOTAL）与小样本（< MIN_TOTAL）拆分
    main_stat = stat[stat["total"] >= MIN_TOTAL].sort_values("neg_rate", ascending=True)
    small_stat = stat[stat["total"] < MIN_TOTAL].sort_values("neg_rate", ascending=False)

    colors = ["#D9534F" if r > global_neg_rate else "#5CB85C"
              for r in main_stat["neg_rate"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(main_stat))
    ax.barh(y, main_stat["neg_rate"].values, color=colors, edgecolor="white")
    ax.set_yticks(y)
    names = main_stat.index.tolist()
    # 有效重灾区 Top3 加 ★（main_stat 升序，末尾 3 个负面率最高；用★避免中文字体缺字）
    top3_idx = set(range(len(names) - 3, len(names)))
    ax.set_yticklabels([f"{n} ★" if i in top3_idx else n
                        for i, n in enumerate(names)])

    xmax = main_stat["neg_rate"].max()
    ax.set_xlim(0, xmax * 1.22)
    for i, (rate, neg, tot) in enumerate(
            zip(main_stat["neg_rate"], main_stat["neg"], main_stat["total"])):
        ax.text(rate + xmax * 0.01, i, f"{rate:.1f}%（{neg}/{tot}）",
                va="center", fontsize=10)

    ax.axvline(global_neg_rate, color="#555555", linestyle="--", linewidth=1.6)
    ax.text(global_neg_rate, len(names) - 0.35,
            f" 全局平均负面率：{global_neg_rate:.1f}%",
            color="#555555", fontsize=10.5, va="bottom")

    ax.set_xlabel("负面评价占比（负面数 / 该属性总提及数）")
    ax.set_ylabel("属性种类")
    ax.set_title(f"各属性负面评价占比排序（仅含总提及数≥{MIN_TOTAL}；红=高于全局均值，绿=低于）")

    # 图下方文本：小样本属性（< MIN_TOTAL）单独列出
    small_str = "、".join(
        f"{cat} {row['neg_rate']:.1f}%({int(row['neg'])}/{int(row['total'])})"
        for cat, row in small_stat.iterrows()
    ) or "无"
    fig.text(0.5, 0.015,
             f"小样本属性（提及数<{MIN_TOTAL}，仅供参考）：{small_str}",
             ha="center", fontsize=10.5, color="#555555")

    fig.tight_layout(rect=[0, 0.045, 1, 1])  # 底部留出行给小样本文本
    out = os.path.join(CHART_DIR, "04_category_negative_rate.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"    [OK] 已保存 {out}")
    # 返回：主图排序(降序)、小样本排序(降序)、全局均值、阈值
    return (main_stat.sort_values("neg_rate", ascending=False),
            small_stat, global_neg_rate, MIN_TOTAL)


# ---------------------------------------------------------------------------
# 主流程 + 汇总报告
# ---------------------------------------------------------------------------
def main():
    len_stats = chart1()
    bucket_counts, bucket_total, max_q = chart2()
    cat_counts, cat_pct = chart3()
    main_stat, small_stat, global_neg_rate, min_total = chart4()
    neg_stat = pd.concat([main_stat, small_stat]).sort_values("neg_rate", ascending=False)

    # 极性明细
    pol = labels["Polarities"].value_counts()
    pol_pct = labels["Polarities"].value_counts(normalize=True) * 100

    # 长尾属性（<1%）
    long_tail = [c for c in cat_pct.index if cat_pct[c] < 1.0]

    # 有效重灾区 Top3（仅取主图 >= min_total 的属性）相对全局倍数
    top3 = main_stat.head(3)
    top3_lines = []
    for cat, row in top3.iterrows():
        mult = row["neg_rate"] / global_neg_rate if global_neg_rate else 0
        top3_lines.append(f"{cat}({row['neg_rate']:.1f}%)")
    # 小样本文本行（< min_total）
    small_md_str = "、".join(
        f"{cat} {row['neg_rate']:.1f}%({int(row['neg'])}/{int(row['total'])})"
        for cat, row in small_stat.iterrows()
    ) or "无"

    # 四元组分桶占比
    bucket_md = []
    for k in ["1", "2", "3", "4", "5及以上"]:
        if k in bucket_counts:
            cnt = bucket_counts[k]
            bucket_md.append(f"| {k} | {cnt} | {cnt / bucket_total * 100:.1f}% |")

    md = f"""# EDA 数据概览总结

> 数据源：清洗后化妆品电商评论数据集
> 总评论 {len(wide)} 条 / 总四元组 {TOTAL_LABELS} 个

## 一、评论长度统计（字）

| 指标 | 值 |
|------|----|
| 均值 | {len_stats['mean']:.1f} |
| 中位数 | {len_stats['median']:.0f} |
| 最大值 | {len_stats['max']:.0f} |
| 最小值 | {len_stats['min']:.0f} |
| 样本数 | {len_stats['n']} |

评论普遍简短（多数在 20 字左右），无超长文本，适合句级观点抽取。

## 二、每条评论的四元组数量分布

| 四元组数量 | 评论条数 | 占比 |
|-----------|---------|------|
{chr(10).join(bucket_md)}

> 实际最大值为 {max_q} 个，已将 5 及以上合并为一档。多数评论仅含 1~2 个四元组，多观点评论占少数。

## 三、属性种类分布

**Top5 属性：**

| 排名 | 属性 | 提及次数 | 占比 |
|------|------|---------|------|
""" + "\n".join(
        f"| {i+1} | {c} | {cat_counts[c]} | {cat_pct[c]:.1f}% |"
        for i, c in enumerate(cat_counts.index[:5])
    ) + f"""

**长尾属性（占比 < 1%）：** {'、'.join(long_tail) if long_tail else '无'}

属性高度集中于头部，“整体”单类占比最高；长尾属性样本稀少，建模需做类别加权或样本增强。

## 四、各属性负面评价占比排序（全量）

| 属性 | 负面数 | 总提及数 | 负面占比 | 是否入主图(≥{min_total}) |
|------|--------|---------|---------|------------------|
""" + "\n".join(
        f"| {cat} | {int(row['neg'])} | {int(row['total'])} | {row['neg_rate']:.1f}% | {'✅' if row['total'] >= min_total else '—'} |"
        for cat, row in neg_stat.iterrows()
    ) + f"""

> 全局平均负面率：**{global_neg_rate:.1f}%**（= 全量负面 {int(pol.get('负面', 0))} / 总四元组 {TOTAL_LABELS}）。
> 图 4 主图仅展示总提及数 ≥{min_total} 的属性；以下为小样本属性（< {min_total}，仅供参考，负面率置信度低）：
> {small_md_str}

## 五、整体极性分布明细（对照）

| 极性 | 数量 | 占比 |
|------|------|------|
""" + "\n".join(
        f"| {p} | {pol[p]} | {pol_pct[p]:.1f}% |"
        for p in ["正面", "负面", "中性"] if p in pol.index
    ) + f"""

正面占压倒性多数，单独看无洞察价值，因此图 4 改为“按属性看负面占比”以定位差评重灾区。

## 六、一句话结论

剔除小样本属性后，负面率最高的有效重灾区是：**{('、'.join(top3_lines)) if top3_lines else '无'}**，
是运营与质量改进应优先关注的属性。

> 补充洞察：占大头的“整体”类评价（{cat_pct.get('整体', 0):.1f}%）负面率仅 {neg_stat.loc['整体','neg_rate']:.1f}%，低于全局平均；
> 用户在表扬时倾向泛泛而谈，在吐槽时才点名属性，因此点名属性的负面率更能反映真实痛点。
"""

    md_path = os.path.join(BASE_DIR, "eda_overview_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print("\n" + "=" * 60)
    print(f"[完成] 4 张图表已生成于 {CHART_DIR}")
    print(f"[完成] 汇总报告已写入 {md_path}")


if __name__ == "__main__":
    main()
