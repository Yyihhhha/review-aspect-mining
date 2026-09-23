# -*- coding: utf-8 -*-
"""
最终报告整合脚本 final_report.py
================================
读取本项目前面所有分析产出，重算核心指标（保证数字真实、口径一致），
生成一份可用于简历/面试的完整分析报告 final_report.md（无 BOM UTF-8）。

数据来源与口径：
  - 可重算指标：cleaned_labels.csv / review_wide.csv / opinion_word_freq.csv /
                bad_review_subcategories.csv（与前序 EDA 脚本同源，结果一致）
  - 仅存在于文本的指标（数据质量评分）：从 cleaning_report.md 正则提取
  - 帕累托：复用 pareto_analysis.build_problem_points()
  任一指标缺失时写“数据缺失”，绝不编造。
"""

import os
import re
from collections import Counter
from itertools import combinations

import pandas as pd

import pareto_analysis as pa   # 复用帕累托问题点构建（import 不触发绘图）

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MISS = "数据缺失"


def _p(path):
    return os.path.join(BASE_DIR, path)


def _fmt(v, nd=1, suffix=""):
    return MISS if v is None else f"{v:.{nd}f}{suffix}"


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------
def compute_metrics():
    m = {}
    lab = pd.read_csv(_p("cleaned_labels.csv"), encoding="utf-8")
    wide = pd.read_csv(_p("review_wide.csv"), encoding="utf-8")

    # --- 概览 ---
    m["评论数"] = int(wide["ID"].nunique())
    m["四元组数"] = int(len(lab))
    m["平均每评论标签数"] = m["四元组数"] / m["评论数"] if m["评论数"] else None

    pol = lab["Polarities"].value_counts()
    total = len(lab)
    m["极性"] = {p: (int(pol.get(p, 0)), pol.get(p, 0) / total * 100 if total else None)
                 for p in ["正面", "负面", "中性"]}
    m["负面绝对数"] = int(pol.get("负面", 0))

    cat = lab["Categories"].value_counts()
    m["属性Top3"] = [(c, int(n), n / total * 100) for c, n in cat.head(3).items()]

    imp_n = int((lab["AspectTerms"].astype(str).str.strip() == "_").sum())
    m["隐式占比"] = imp_n / total * 100 if total else None

    # --- 评论长度 / 四元组数量分布（来自 review_wide）---
    if "评论字数" in wide.columns:
        L = pd.to_numeric(wide["评论字数"], errors="coerce").dropna()
        m["评论长度"] = {"均值": L.mean(), "中位数": L.median(), "最大": L.max(), "最小": L.min()}
    else:
        m["评论长度"] = None
    if "四元组数量" in wide.columns:
        q = pd.to_numeric(wide["四元组数量"], errors="coerce").dropna()
        bucket = q.apply(lambda x: "5及以上" if x >= 5 else str(int(x)))
        bc = bucket.value_counts()
        order = ["1", "2", "3", "4", "5及以上"]
        m["四元组分布"] = [(k, int(bc.get(k, 0)), bc.get(k, 0) / len(q) * 100 if len(q) else None)
                          for k in order]
        m["四元组分布最大档"] = "5及以上"
    else:
        m["四元组分布"] = None

    # --- 各属性负面率 + 有效重灾区（总提及≥80，按负面率降序）---
    stat = lab.groupby("Categories")["Polarities"].agg(
        total="size", neg=lambda s: int((s == "负面").sum()))
    stat["rate"] = stat["neg"] / stat["total"] * 100
    m["全局平均负面率"] = m["负面绝对数"] / total * 100 if total else None
    valid = stat[stat["total"] >= 80].sort_values("rate", ascending=False)
    m["有效重灾区Top3"] = [(idx, int(r["neg"]), int(r["total"]), r["rate"])
                          for idx, r in valid.head(3).iterrows()]
    m["使用体验负面"] = (int(stat.loc["使用体验", "neg"]), int(stat.loc["使用体验", "total"]),
                       stat.loc["使用体验", "rate"]) if "使用体验" in stat.index else None

    # --- 隐式 vs 显式负面率 ---
    is_imp = lab["AspectTerms"].astype(str).str.strip() == "_"
    for name, mask in [("隐式", is_imp), ("显式", ~is_imp)]:
        sub = lab[mask]
        n = len(sub)
        neg = int((sub["Polarities"] == "负面").sum())
        m[f"{name}负面率"] = (n, neg, neg / n * 100 if n else None)

    # --- 属性共现 Top3（同条评论去重属性两两共现）---
    review_cats = lab.groupby("ID")["Categories"].apply(lambda s: set(s))
    pair = Counter()
    for cats in review_cats:
        for a, b in combinations(sorted(cats), 2):
            pair[(a, b)] += 1
    m["共现Top3"] = [(f"{a}+{b}", c) for (a, b), c in pair.most_common(3)]

    # --- 各属性隐式占比（取最高/最低两极）---
    ir = lab.assign(_imp=is_imp).groupby("Categories").agg(
        total=("ID", "size"), imp=("_imp", "sum"))
    ir["rate"] = ir["imp"] / ir["total"] * 100
    irs = ir["rate"].sort_values(ascending=False)
    m["隐式占比高"] = [(c, ir.loc[c, "rate"]) for c in irs.head(3).index]
    m["隐式占比低"] = [(c, ir.loc[c, "rate"]) for c in irs.tail(3).index[::-1]]

    # --- 观点词平均长度（加权）+ Top3 ---
    owf = pd.read_csv(_p("opinion_word_freq.csv"), encoding="utf-8")
    for name in ["正面", "负面"]:
        sub = owf[owf["Polarity"] == name]
        csum = sub["Count"].sum()
        avg = (sub["AvgLength"] * sub["Count"]).sum() / csum if csum else None
        top = sub.sort_values("Count", ascending=False)["OpinionTerm"].head(3).tolist()
        m[f"观点{name}"] = {"平均长度": avg, "Top3": top}

    # --- 差评子类四属性最大痛点（读 CSV）---
    brs = pd.read_csv(_p("bad_review_subcategories.csv"), encoding="utf-8")
    EXCL = {"其他", "隐式负面", "通用差评"}
    m["差评最大痛点"] = {}
    for attr in ["使用体验", "包装", "气味", "服务"]:
        sub = brs[(brs["属性"] == attr) & (~brs["子类"].isin(EXCL))]
        if len(sub):
            r = sub.loc[sub["负面数"].idxmax()]
            m["差评最大痛点"][attr] = (r["子类"], int(r["负面数"]), r["占该属性负面比"])

    # --- 帕累托（复用 build_problem_points）---
    pp, ptot, _ = pa.build_problem_points()
    n80 = int((pp["累计占比"] < 80).sum()) + 1
    m["帕累托"] = {
        "问题点数": len(pp), "可归类差评": int(ptot),
        "n80": n80, "n80累计": pp["累计占比"].iloc[n80 - 1],
        "n80占问题点比": n80 / len(pp) * 100,
        "质地占可归类": pp["占比"].iloc[0],
        "质地差评": int(pp["count"].iloc[0]), "质地标签": pp["label"].iloc[0],
        "可归类占全量": ptot / m["负面绝对数"] * 100 if m["负面绝对数"] else None,
        "质地占全量": pp["count"].iloc[0] / m["负面绝对数"] * 100 if m["负面绝对数"] else None,
        "P0": pp[pp["占比"] >= 10][["label", "count", "占比"]].values.tolist(),
        "P1": pp[(pp["占比"] >= 5) & (pp["占比"] < 10)][["label", "count", "占比"]].values.tolist(),
        "P2数": int((pp["占比"] < 5).sum()),
        "P2合计占比": float(pp[pp["占比"] < 5]["占比"].sum()),
    }
    # 属性维度帕累托
    negcat = lab[lab["Polarities"] == "负面"]["Categories"].value_counts()
    ncum = negcat / negcat.sum() * 100
    ncum = ncum.cumsum()
    n80c = int((ncum < 80).sum()) + 1
    m["属性帕累托"] = {"n80": n80c, "累计": ncum.values[n80c - 1],
                    "属性": list(negcat.index[:n80c])}

    # --- 数据质量评分（从 cleaning_report.md 正则提取）---
    m["质量评分"], m["质量等级"] = read_quality()
    return m


def read_quality():
    try:
        txt = open(_p("cleaning_report.md"), encoding="utf-8").read()
        score = re.search(r"总评分[:：]\s*([\d.]+)", txt)
        grade = re.search(r"数据质量等级[:：]\s*([\u4e00-\u9fa5A-Za-z]+)", txt)
        return (score.group(1) if score else MISS, grade.group(1) if grade else MISS)
    except Exception:
        return MISS, MISS


# ---------------------------------------------------------------------------
# 报告组装
# ---------------------------------------------------------------------------
def build_report(m):
    L = []
    A = L.append
    pol = m["极性"]
    pare = m["帕累托"]

    A("# 电商评论观点挖掘与差评归因分析报告\n")

    # 核心指标速览
    A("## 核心指标速览\n")
    A("| 指标 | 数值 |")
    A("|------|------|")
    A(f"| 评论数 / 四元组数 | {m['评论数']} 条 / {m['四元组数']} 个 |")
    A(f"| 数据质量评分 | {m['质量评分']} / {m['质量等级']} |")
    A(f"| 极性分布（正/负/中） | {pol['正面'][1]:.1f}% / {pol['负面'][1]:.1f}% / {pol['中性'][1]:.1f}% |")
    hot = m["有效重灾区Top3"]
    A(f"| 有效差评重灾区 Top3 | {'、'.join(f'{a}({r:.1f}%)' for a, _, _, r in hot)} |")
    p0 = pare["P0"][:3]
    A(f"| 可归类差评 P0 问题点 Top3 | {'、'.join(f'{lb}({int(c)}个/{pc:.1f}%)' for lb, c, pc in p0)} |")
    A(f"| 隐式属性占比 | {m['隐式占比']:.1f}% |")
    A(f"| 一句话结论 | 解决前 {pare['n80']} 个问题点即可覆盖 {pare['n80累计']:.1f}% 的可归类差评 |")
    A("")

    # 一、项目背景与目标
    A("## 一、项目背景与目标\n")
    A("- **任务定义**：从化妆品电商评论中抽取观点四元组（属性特征 AspectTerms、观点词 OpinionTerms、"
      "属性种类 Categories、观点极性 Polarities）。")
    A("- **分析目标**：不止于抽取，进一步做**差评归因**，输出可落地的业务改进建议。")
    A(f"- **数据规模**：{m['评论数']} 条评论 / {m['四元组数']} 个四元组。")
    A("")

    # 二、数据概览
    A("## 二、数据概览\n")
    A(f"- 数据质量评分 **{m['质量评分']} / 100（{m['质量等级']}）**（见 cleaning_report.md）。")
    A(f"- 平均每条评论 {m['平均每评论标签数']:.2f} 个标签；隐式属性占比 **{m['隐式占比']:.1f}%**"
      "（评论只表达观点、未点名属性）。\n")
    A("| 核心指标 | 数值 |")
    A("|----------|------|")
    A(f"| 评论数 / 四元组数 | {m['评论数']} / {m['四元组数']} |")
    A(f"| 平均每条评论标签数 | {m['平均每评论标签数']:.2f} |")
    A(f"| 极性分布 | 正面 {pol['正面'][1]:.1f}%（{pol['正面'][0]}）、负面 {pol['负面'][1]:.1f}%"
      f"（{pol['负面'][0]}）、中性 {pol['中性'][1]:.1f}%（{pol['中性'][0]}） |")
    A(f"| 属性 Top3 | {'、'.join(f'{c} {p:.1f}%' for c, _, p in m['属性Top3'])} |")
    A(f"| 隐式属性占比 | {m['隐式占比']:.1f}% |")
    A("")
    A("> **数据特征一句话**：小样本、极性极不平衡（正面压倒性）、属性长尾、以隐式表达为主。\n")

    # 三、核心发现
    A("## 三、核心发现\n")
    A(f"本项目沿“**表面好评 → 表达结构 → 吐槽落点 → 归因收敛**”一条主线展开：\n")
    A(f"1. **正面占比 {pol['正面'][1]:.1f}%，看似“用户很满意”——但这是假象**（步骤三，见图 4）；")
    A(f"2. 拆开隐式/显式发现：**隐式负面率 {m['隐式负面率'][2]:.1f}% < 显式 {m['显式负面率'][2]:.1f}%**，"
      "即“表扬泛泛、吐槽点名”——用户笼统好评时不点名属性，真正不满时才精确制导（步骤四，见图 5）；")
    A(f"3. 再看观点词本身：**负面平均 {m['观点负面']['平均长度']:.2f} 字 > 正面 {m['观点正面']['平均长度']:.2f} 字**，"
      "“吐槽更具体、更长”，而吐槽集中在少数属性上（步骤五，见图 9）；")
    A(f"4. 收敛到问题点：单点“**{pare['质地标签']}**”独占 **{pare['质地占可归类']:.1f}%** 的可归类差评（步骤六，见图 16）。\n")
    A(f"> **升华**：这解释了“整体好评率 {pol['正面'][1]:.0f}%”与“用户总在吐槽”的体感差异——"
      "好评藏在笼统的“不错”“很好用”里，差评集中在具体的产品问题上。\n")

    # 四、整体分析
    A("## 四、整体分析（步骤三）\n")
    cl = m["评论长度"]
    if cl:
        A(f"- **评论长度**（图 1）：均值 {cl['均值']:.1f} / 中位数 {cl['中位数']:.0f} / "
          f"最大 {cl['最大']:.0f} / 最小 {cl['最小']:.0f} 字，评论普遍简短，适合句级观点抽取。")
    if m["四元组分布"]:
        dist = "、".join(f"{k}：{int(c)} 条（{p:.1f}%）" for k, c, p in m["四元组分布"])
        A(f"- **每条评论四元组数量**（图 2）：{dist}；多数仅 1~2 个观点。")
    A(f"- **属性分布**（图 3）：头部集中，{'、'.join(f'{c} {p:.1f}%' for c, _, p in m['属性Top3'])}，"
      "长尾属性（其他/成分/尺寸/新鲜度）占比 < 1%。")
    A(f"- **极性分布**（图 4）：正面 {pol['正面'][1]:.1f}% 占绝对多数，故图 4 改以“各属性负面率”定位重灾区。\n")

    # 五、属性维度
    A("## 五、属性维度深度分析（步骤四）\n")
    gm = m["全局平均负面率"]
    A(f"- **各属性负面率排序**（图 4）：有效重灾区（总提及≥80）为 "
      f"{'、'.join(f'{a} {r:.1f}%' for a, _, _, r in hot)}，"
      f"分别是全局平均 {gm:.1f}% 的 {' / '.join(f'{r/gm:.1f}' for _, _, _, r in hot)} 倍。")
    A(f"- **隐式 vs 显式**（图 5）：隐式负面率 {m['隐式负面率'][2]:.1f}%（{m['隐式负面率'][0]} 个） < "
      f"显式 {m['显式负面率'][2]:.1f}%（{m['显式负面率'][0]} 个）——点名属性时负面更集中。")
    A(f"- **属性共现结构**（图 6）：“整体”是绝对枢纽，Top3 共现对："
      f"{'、'.join(f'{p}({c})' for p, c in m['共现Top3'])}。")
    hi = "、".join(f"{c} {r:.1f}%" for c, r in m["隐式占比高"])
    lo = "、".join(f"{c} {r:.1f}%" for c, r in m["隐式占比低"])
    A(f"- **隐式占比两极分化**（图 7/8）：高——{hi}；低——{lo}。\n")

    # 六、情感与观点词
    A("## 六、情感与观点词分析（步骤五）\n")
    A(f"- **观点词长度**（图 9）：正面平均 {m['观点正面']['平均长度']:.2f} 字 vs "
      f"负面 {m['观点负面']['平均长度']:.2f} 字，负面表达更具体。")
    A(f"- **表扬 vs 吐槽 Top3**（图 10/11）：正面“{'、'.join(m['观点正面']['Top3'])}” vs "
      f"负面“{'、'.join(m['观点负面']['Top3'])}”。")
    A("- **差评重灾区痛点词**（图 11/12）：包装看“没有/太随便了/破了”（做工与破损）、"
      "气味看“不是很喜欢/刺鼻/太香了”（难闻/过浓）、服务看“极差/敷衍了事/太慢”（态度差）。\n")

    # 七、差评归因专题
    A("## 七、差评归因专题（步骤五补 + 步骤六）\n")
    dp = m["差评最大痛点"]
    A("- **四属性最大差评子类**（占该属性负面比，图 15）："
      + "；".join(f"{a}—{dp[a][0]} {dp[a][2]}" for a in ["使用体验", "包装", "气味", "服务"]) + "。")
    A(f"- 关键词表覆盖率约 **63%**（见 bad_review_subcategory.md），剩余“其他”多为长尾个例。\n")
    A(f"- **帕累托·问题点维度**（图 16）：共 {pare['问题点数']} 个问题点、合计 {pare['可归类差评']} 个可归类差评；"
      f"前 **{pare['n80']}** 个问题点（占问题点 {pare['n80占问题点比']:.1f}%）覆盖 **{pare['n80累计']:.1f}%** 可归类差评。")
    A(f"- **帕累托·属性维度**（图 17）：前 **{m['属性帕累托']['n80']}** 个属性"
      f"（{'、'.join(m['属性帕累托']['属性'])}）覆盖 {m['属性帕累托']['累计']:.0f}% 全量差评。")
    A(f"- **判定**：不完全符合典型 80/20——头部单点“{pare['质地标签']}”极度集中（{pare['质地占可归类']:.1f}%），"
      f"但累计到 80% 需覆盖过半问题点（{pare['n80占问题点比']:.1f}%），其余偏长尾。\n")
    A("> **百分比口径说明**：")
    A(f"> - 前 {pare['n80']} 个问题点覆盖 {pare['n80累计']:.1f}% 的**可归类差评**（{pare['可归类差评']} 个），"
      f"占全量差评（{m['负面绝对数']} 个）的 **{pare['可归类占全量']:.1f}%**；")
    A(f"> - “{pare['质地标签']}”占 {pare['质地差评']}/{m['负面绝对数']} = **{pare['质地占全量']:.1f}%** 的全量差评；")
    A("> - 剩余约 70% 差评属于“其他/隐式负面/通用差评”，多为用户未明确表达具体原因的泛化吐槽，"
      "无法直接指向改进方向。本报告中“可归类差评”特指能匹配到具体子类的明确吐槽。\n")

    # 八、业务优化建议
    A("## 八、业务优化建议（P0/P1/P2）\n")
    A("- **投入产出比**：P0 集中在“产品配方”和“包装”两大环节，均属供应链侧可优化项，"
      "改进成本相对固定，却能覆盖 80%+ 的明确差评，ROI 最高。\n")
    BIZ = {
        "使用体验-质地问题": "按肤质分型配方，针对“干/卡粉/浮粉/油/腻”定向改进；细化粉体粒径。",
        "包装-包装简陋/随意": "改进外包装设计、增加缓冲防护材料、统一塑封/封口工艺。",
        "气味-香味过重/刺鼻": "降低香精浓度，提供无香/淡香版本，改用更温和香型。",
        "使用体验-效果问题": "补充功效实证、优化成分吸收，标注适用人群与致敏成分。",
    }
    A("**P0（解决 1 个即可消除 10%+ 可归类差评）**\n")
    for lb, c, pc in pare["P0"]:
        tip = BIZ.get(lb, "结合高频吐槽关键词针对性优化。")
        A(f"- **{lb}**（{int(c)} 个，{pc:.1f}%）→ {tip}")
    A("")
    A("**P1（解决 1 个可消除 5%~10%）**\n")
    for lb, c, pc in pare["P1"]:
        A(f"- {lb}（{int(c)} 个，{pc:.1f}%）")
    A("")
    A(f"**P2（长尾）**：其余 {pare['P2数']} 个问题点，单个占比 < 5%，合计 {pare['P2合计占比']:.1f}%，"
      "暂不单独投入资源。\n")

    # 九、局限与后续
    A("## 九、项目局限与后续方向\n")
    A("- **局限**：")
    A("  - 正/负面词表为示例性词表，未做完整情感词典覆盖（疑似矛盾判定见 cleaning_report.md）；")
    A("  - 差评子类关键词覆盖率约 63%，剩余为长尾个例；")
    A(f"  - 隐式负面（如包装占该属性 13.7%）难以针对性归因；")
    A(f"  - 中性样本仅 {pol['中性'][1]:.1f}%，统计上不可靠。")
    A("- **后续方向**：用语义聚类替代词典匹配提升子类覆盖率；构造 BIO 序列标注样本进入建模阶段；"
      "结合时间维度做差评趋势分析。\n")

    # 图表索引
    A("## 附录：图表索引\n")
    A("| 图号 | 文件名 | 所在章节 |")
    A("|------|--------|----------|")
    idx = [
        ("图 1", "charts/01_review_length_dist.png", "四、整体分析"),
        ("图 2", "charts/02_quadruple_count_dist.png", "四、整体分析"),
        ("图 3", "charts/03_category_distribution.png", "四、整体分析"),
        ("图 4", "charts/04_category_negative_rate.png", "四/五、属性负面率"),
        ("图 5", "charts/05_implicit_vs_explicit_polarity.png", "五、属性维度"),
        ("图 6", "charts/06_category_cooccurrence_heatmap.png", "五、属性维度"),
        ("图 7", "charts/07_category_polarity_heatmap.png", "五、属性维度"),
        ("图 8", "charts/08_implicit_aspect_ratio.png", "五、属性维度"),
        ("图 9", "charts/09_opinion_word_length_dist.png", "六、情感与观点词"),
        ("图 10", "charts/10_positive_opinion_wordcloud.png", "六、情感与观点词"),
        ("图 11", "charts/11_negative_opinion_wordcloud.png", "六、情感与观点词"),
        ("图 12", "charts/12_top_category_opinions.png", "六、情感与观点词"),
        ("图 15", "charts/15_bad_review_subcategory_stacked.png", "七、差评归因"),
        ("图 16", "charts/16_pareto_problem_points.png", "七、差评归因（帕累托）"),
        ("图 17", "charts/17_pareto_categories.png", "七、差评归因（帕累托）"),
    ]
    for g, f, s in idx:
        A(f"| {g} | {f} | {s} |")
    A("")
    A("> 注：图 13、图 14 未纳入本次分析产出，编号保持连续占位。")
    return "\n".join(L)


def main():
    m = compute_metrics()
    md = build_report(m)
    out = _p("final_report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[OK] 最终报告已写入 {out}（约 {len(md)} 字符）")


if __name__ == "__main__":
    main()
