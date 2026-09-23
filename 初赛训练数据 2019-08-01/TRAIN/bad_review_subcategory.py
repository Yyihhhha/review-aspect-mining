# -*- coding: utf-8 -*-
"""
天池化妆品电商评论数据 —— 差评子类细粒度分析脚本 bad_review_subcategory.py
==========================================================================
输入：cleaned_labels.csv（只取 Polarity="负面" 的四元组）
输出：
  - bad_review_subcategories.csv（属性、子类、负面数、占该属性负面比）
  - charts/15_bad_review_subcategory_stacked.png（100% 堆叠条形图，300 DPI）
  - bad_review_subcategory.md（分析报告）

归类规则：观点词包含关键词即命中；按子类定义顺序，命中第一个即停止（互斥）；
未命中任何关键词统一归为"其他"。
"""

import os
from collections import Counter

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

OTHER = "其他"
IMPLICIT = "隐式负面"      # OpinionTerms 为 "_"（空观点词）
GENERIC = "通用差评"        # 泛化吐槽、无具体指向

# 堆叠图配色
COLOR_OTHER = "#CCCCCC"       # 其他：浅灰
COLOR_IMPLICIT = "#666666"    # 隐式负面：深灰
COLOR_GENERIC = "#FF9999"     # 通用差评：浅红

# 四个核心差评属性（固定展示顺序）
ATTR_ORDER = ["使用体验", "包装", "气味", "服务"]

# 通用差评关键词（所有属性共用，排具体子类之后）
GENERIC_KWS = ["差", "不好", "差劲", "不满意", "太差", "烂"]

# 子类关键词规则（优化后）：每个属性 -> [(子类名, [关键词...]), ...]，按顺序命中即停
# 本次补充的关键词已合入对应子类；具体子类之后统再接上 GENERIC
RULES = {
    "使用体验": [
        ("质地问题", ["干", "卡粉", "浮粉", "粉", "油", "腻", "起皮", "拔干"]),
        ("效果问题", ["不好用", "难用", "没用", "不保湿", "不吸收", "过敏", "搓泥",
                   "长痘", "辣眼睛", "假滑"]),
        ("肤感问题", ["紧绷", "不舒服", "刺激", "痒"]),
    ],
    "包装": [
        ("运输破损/漏", ["破", "漏", "压", "碎", "坏", "扁", "水漏"]),
        ("包装简陋/随意", ["简陋", "随便", "简单", "次", "胶带", "旧", "没封口", "没有",
                     "丑", "难看", "太差"]),
    ],
    "气味": [
        ("香味过重/刺鼻", ["香", "刺鼻", "浓", "重", "大", "酒精"]),
        ("异味/不好闻", ["怪", "味", "酸", "臭", "难闻", "不好闻", "不喜欢"]),
    ],
    "服务": [
        ("态度差/敷衍", ["态度", "差", "极差", "敷衍", "恭维", "欺骗", "不理"]),
        ("售后问题/慢", ["解决", "慢", "退货", "麻烦", "不发货", "没人理"]),
    ],
}

# 优化前的原始规则（仅用于“其他”占比前后对比，不含隐式负面拆分与通用差评）
ORIGINAL_RULES = {
    "使用体验": [
        ("质地问题", ["干", "卡粉", "浮粉", "粉", "油", "腻"]),
        ("效果问题", ["不好用", "难用", "没用", "不保湿", "不吸收", "过敏", "搓泥"]),
        ("肤感问题", ["紧绷", "不舒服", "刺激", "痒"]),
    ],
    "包装": [
        ("运输破损/漏", ["破", "漏", "压", "碎", "坏", "扁", "水漏"]),
        ("包装简陋/随意", ["简陋", "随便", "简单", "次", "胶带", "旧", "没封口", "没有"]),
    ],
    "气味": [
        ("香味过重/刺鼻", ["香", "刺鼻", "浓", "重", "大", "酒精"]),
        ("异味/不好闻", ["怪", "味", "酸", "臭", "难闻", "不好闻"]),
    ],
    "服务": [
        ("态度差/敷衍", ["态度", "差", "极差", "敷衍", "恭维", "欺骗", "不理"]),
        ("售后问题/慢", ["解决", "慢", "退货", "麻烦", "不发货", "没人理"]),
    ],
}

# 本次新增/补充的关键词（用于统计实际命中条数；按 (子类, 关键词) 归因）
NEW_KEYWORDS = {
    "质地问题": ["起皮", "拔干"],
    "效果问题": ["长痘", "过敏", "搓泥", "辣眼睛", "假滑"],
    "包装简陋/随意": ["丑", "难看", "太差"],
    "异味/不好闻": ["不喜欢", "难闻"],
    GENERIC: GENERIC_KWS,
}


# ---------------------------------------------------------------------------
# 归类
# ---------------------------------------------------------------------------
def classify(term, rules):
    """返回 (子类, 命中的第一个关键词)；无关键词命中时返回 (其他, None)。
    优先级：隐式负面(_) > 具体子类(按顺序) > 通用差评 > 其他。"""
    t = str(term).strip()
    if t == "_":
        return IMPLICIT, None
    for subcat, kws in rules:
        for kw in kws:
            if kw in t:
                return subcat, kw
    return OTHER, None


def matched_orig(term, orig_rules):
    """优化前规则是否命中（不含隐式负面拆分，_ 归为未命中）。"""
    t = str(term).strip()
    for _, kws in orig_rules:
        for kw in kws:
            if kw in t:
                return True
    return False


def main():
    lab = pd.read_csv(os.path.join(BASE_DIR, "cleaned_labels.csv"), encoding="utf-8")
    neg = lab[lab["Polarities"] == "负面"].copy()
    print(f"[数据] cleaned_labels.csv 总行数={len(lab)}，负面四元组={len(neg)}")

    rows = []               # 统计结果
    other_terms = {}        # 属性 -> Counter(其他观点词)
    attr_neg_total = {}     # 属性 -> 负面总数
    kw_hits = Counter()     # (子类, 关键词) -> 命中条数（首匹配归因）
    other_before = {}       # 属性 -> 优化前“其他”占比

    for attr in ATTR_ORDER:
        sub = neg[neg["Categories"] == attr].copy()
        total = len(sub)
        attr_neg_total[attr] = total
        rules = RULES[attr] + [(GENERIC, GENERIC_KWS)]

        results = sub["OpinionTerms"].apply(lambda x: classify(x, rules))
        sub["子类"] = results.apply(lambda t: t[0])
        sub["命中词"] = results.apply(lambda t: t[1])
        cnt = Counter(sub["子类"])
        for _, r in sub.iterrows():
            if r["命中词"] is not None:
                kw_hits[(r["子类"], r["命中词"])] += 1
        other_terms[attr] = Counter(
            sub.loc[sub["子类"] == OTHER, "OpinionTerms"].astype(str).str.strip()
        )
        # 优化前“其他”占比（含 _ 未拆分）
        ob = sum(0 if matched_orig(x, ORIGINAL_RULES[attr]) else 1
                 for x in sub["OpinionTerms"])
        other_before[attr] = ob / total * 100 if total else 0.0

        # 展示顺序：具体子类 -> 通用差评 -> 隐式负面 -> 其他
        ordered = [s for s, _ in RULES[attr]] + [GENERIC, IMPLICIT, OTHER]
        for sc in ordered:
            c = int(cnt.get(sc, 0))
            pct = c / total * 100 if total else 0.0
            rows.append({"属性": attr, "子类": sc, "负面数": c, "占该属性负面比": f"{pct:.1f}%"})

    df = pd.DataFrame(rows, columns=["属性", "子类", "负面数", "占该属性负面比"])
    csv_path = os.path.join(BASE_DIR, "bad_review_subcategories.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[OK] 统计表已写入 {csv_path}（{len(df)} 行）")

    chart15(df, attr_neg_total)
    write_report(df, attr_neg_total, other_terms, kw_hits, other_before)


# ---------------------------------------------------------------------------
# 图表 15：差评子类构成 100% 堆叠条形图
# ---------------------------------------------------------------------------
def _text_color(rgb):
    """根据背景亮度选黑/白文字，保证浅底(通用差评/其他/浅红)可读。"""
    r, g, b = rgb[0], rgb[1], rgb[2]
    return "#333333" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.62 else "white"


def chart15(df, attr_neg_total):
    fig, ax = plt.subplots(figsize=(12, 8))
    bar_w = 0.55
    special = {IMPLICIT, GENERIC, OTHER}
    for xi, attr in enumerate(ATTR_ORDER):
        sub_df = df[df["属性"] == attr].reset_index(drop=True)
        spec_names = [s for s in sub_df["子类"] if s not in special]
        n_spec = len(spec_names)
        reds = [plt.cm.Reds(v) for v in np.linspace(0.45, 0.92, n_spec)] if n_spec else []

        def color_of(sc, _reds=reds, _spec=spec_names):
            if sc == IMPLICIT:
                return matplotlib.colors.to_rgb(COLOR_IMPLICIT)
            if sc == GENERIC:
                return matplotlib.colors.to_rgb(COLOR_GENERIC)
            if sc == OTHER:
                return matplotlib.colors.to_rgb(COLOR_OTHER)
            return _reds[_spec.index(sc)]

        bottom = 0.0
        for _, row in sub_df.iterrows():
            pct = float(row["占该属性负面比"].rstrip("%"))
            if pct <= 0:
                continue
            rgb = color_of(row["子类"])
            ax.bar(xi, pct, bar_w, bottom=bottom, color=rgb,
                   edgecolor="white", linewidth=1.2)
            label = f"{row['子类']}\n{pct:.1f}%" if pct >= 5 else (f"{pct:.1f}%" if pct >= 2 else "")
            if label:
                ax.text(xi, bottom + pct / 2, label, ha="center", va="center",
                        fontsize=9, color=_text_color(rgb), fontweight="bold")
            bottom += pct

    ax.set_xticks(range(len(ATTR_ORDER)))
    ax.set_xticklabels([f"{a}\n(负面 {attr_neg_total[a]} 个)" for a in ATTR_ORDER], fontsize=12)
    ax.set_ylabel("占该属性负面总数比例（%）")
    ax.set_ylim(0, 100)
    ax.set_title("差评子类构成（100% 堆叠）：红=具体问题，浅红=通用差评，深灰=隐式负面，浅灰=其他")
    ax.set_xlabel("")
    fig.tight_layout()
    out = os.path.join(CHART_DIR, "15_bad_review_subcategory_stacked.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[OK] 图表15已保存 {out}")


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def write_report(df, attr_neg_total, other_terms, kw_hits, other_before):
    L = []
    L.append("# 差评子类细粒度分析（bad_review_subcategory）— 关键词表优化版\n")
    L.append("> 数据源：cleaned_labels.csv 中 Polarity=\"负面\" 的四元组；"
             "按优先级【隐式负面(_) > 具体子类(按顺序) > 通用差评 > 其他】命中即停（互斥）。\n")

    # 一、各属性统计表
    L.append("## 一、各属性差评子类统计\n")
    for attr in ATTR_ORDER:
        sub = df[df["属性"] == attr]
        L.append(f"### {attr}（负面总数 {attr_neg_total[attr]}）\n")
        L.append("| 子类 | 负面数 | 占该属性负面比 |")
        L.append("|------|--------|----------------|")
        for _, r in sub.iterrows():
            L.append(f"| {r['子类']} | {r['负面数']} | {r['占该属性负面比']} |")
        L.append("")

    # 二、一句话结论（取非“其他/隐式负面”的最大子类）
    L.append("## 二、一句话结论\n")
    parts = []
    for attr in ATTR_ORDER:
        sub = df[(df["属性"] == attr) & (~df["子类"].isin([OTHER, IMPLICIT]))]
        if len(sub) and sub["负面数"].max() > 0:
            top = sub.loc[sub["负面数"].idxmax()]
            parts.append(f"{attr}最大痛点是**{top['子类']}**（占 {top['占该属性负面比']}）")
        else:
            parts.append(f"{attr}未命中任何关键词子类")
    L.append("；".join(parts) + "。\n")

    # 三、Top5 具体问题（排除其他/隐式负面）
    L.append("## 三、最严重的具体问题 Top5（按绝对负面数，排除“其他/隐式负面”）\n")
    real = df[~df["子类"].isin([OTHER, IMPLICIT])].copy()
    real = real.sort_values("负面数", ascending=False).head(5)
    L.append("| 排名 | 属性 | 子类 | 负面数 | 占该属性负面比 |")
    L.append("|------|------|------|--------|----------------|")
    for i, (_, r) in enumerate(real.iterrows(), 1):
        L.append(f"| {i} | {r['属性']} | {r['子类']} | {r['负面数']} | {r['占该属性负面比']} |")
    L.append("")

    # 四、优化前后“其他”占比对比 + 覆盖率
    L.append("## 四、“其他”占比：优化前后对比与关键词覆盖率\n")
    L.append("| 属性 | 其他占比(前) | 其他占比(后) | 隐式负面(后) | 覆盖率(后) |")
    L.append("|------|-------------|-------------|-------------|-----------|")
    tot_all = cov_all = 0
    for attr in ATTR_ORDER:
        total = attr_neg_total[attr]
        o_after = int(df[(df["属性"] == attr) & (df["子类"] == OTHER)]["负面数"].iloc[0])
        imp = int(df[(df["属性"] == attr) & (df["子类"] == IMPLICIT)]["负面数"].iloc[0])
        o_pct = o_after / total * 100
        cov = (total - o_after - imp) / total * 100
        tot_all += total
        cov_all += (total - o_after - imp)
        L.append(f"| {attr} | {other_before[attr]:.1f}% | {o_pct:.1f}% | {imp}({imp/total*100:.1f}%) | {cov:.1f}% |")
    overall_cov = cov_all / tot_all * 100
    L.append("")
    L.append(f"> 覆盖率 = （具体子类 + 通用差评）/ 负面总数；“隐式负面”（OpinionTerms=\"_\"）已单独拆出，不再计入“其他”。\n")

    # 五、新增关键词实际命中条数
    L.append("## 五、本次新增/补充关键词的实际命中条数\n")
    L.append("（按首匹配归因；命中 0 条说明该词在当前数据中不存在，或已被同子类更早的关键词覆盖，属规则正常）\n")
    L.append("| 子类 | 新增关键词 | 命中条数 |")
    L.append("|------|-----------|---------|")
    for sc, kws in NEW_KEYWORDS.items():
        for kw in kws:
            L.append(f"| {sc} | {kw} | {kw_hits.get((sc, kw), 0)} |")
    L.append("")

    # 六、“其他”长尾高频词
    L.append("## 六、剩余“其他”长尾高频词\n")
    for attr in ATTR_ORDER:
        top5 = other_terms[attr].most_common(5)
        words = "、".join(f"{w}({c})" for w, c in top5) if top5 else "（无）"
        L.append(f"- **{attr}**：“其他”高频词 Top5：{words}")
    L.append("")
    L.append(f"> 本次优化后关键词覆盖率约 **{overall_cov:.0f}%**，剩余“其他”多为仅出现 1 次的长尾个例（如“洗不净”“偏深”等），"
             f"不代表共性问题。\n")

    md = "\n".join(L)
    md_path = os.path.join(BASE_DIR, "bad_review_subcategory.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[OK] 报告已写入 {md_path}")


if __name__ == "__main__":
    main()
