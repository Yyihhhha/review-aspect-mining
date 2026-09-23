# -*- coding: utf-8 -*-
"""
天池化妆品电商评论数据 —— 情感与观点词分析脚本 eda_sentiment.py
================================================================
输入（清洗后）：cleaned_reviews.csv、cleaned_labels.csv、review_wide.csv
输出：charts/09~12 四张 PNG（300 DPI）
      + opinion_word_freq.csv（观点词词频）
      + eda_sentiment_summary.md（核心数据汇总）

全局配色约定：正面=绿、负面=红、中性=灰。
观点词直接取清洗后的 OpinionTerms（已是抽取好的观点短语，不再二次分词）。
"""

import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

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

# 词云/图表中文字体路径
FONT_CANDIDATES = [r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\msyh.ttc"]
FONT_PATH = next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)
if FONT_PATH is None:
    raise RuntimeError("未找到可用的中文字体（simhei.ttf / msyh.ttc），词云无法渲染中文。")

POL_ORDER = ["正面", "负面", "中性"]
POS_COLOR, NEG_COLOR, NEU_COLOR = "#5CB85C", "#D9534F", "#9E9E9E"

STOPWORDS = ['的', '了', '是', '很', '非常', '就是', '都', '也', '还', '有', '我', '买', '用', '这个',
             '真的', '太', '就', '才', '在', '和', '与', '啊', '吧', '呢', '呀', '哦', '嗯', '_',
             '一点', '有些', '感觉']
STOP_SET = set(STOPWORDS)

# 差评重灾区（样本量>=80 的有效重灾区）
HOT_CATS = ["包装", "气味", "服务"]

# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
reviews = pd.read_csv(os.path.join(BASE_DIR, "cleaned_reviews.csv"), encoding="utf-8")
labels = pd.read_csv(os.path.join(BASE_DIR, "cleaned_labels.csv"), encoding="utf-8")
wide = pd.read_csv(os.path.join(BASE_DIR, "review_wide.csv"), encoding="utf-8")

# 规范观点词：去空白，过滤纯停用词/空串
labels["Opinion"] = labels["OpinionTerms"].astype(str).str.strip()
labels["word_len"] = labels["Opinion"].str.len()
valid_mask = (labels["Opinion"] != "") & (~labels["Opinion"].isin(STOP_SET)) & labels["Opinion"].notna()
DATA = labels[valid_mask].copy()          # 参与词频/词云/长度分析的有效观点词


def announce(name, *frames):
    print(f"\n[{name}] 用到的数据形状：")
    for df, cols in frames:
        sub = df[cols] if cols else df
        print(f"    - {df.shape} 使用列: {list(sub.columns)}  行数={len(df)}")


def freq_of(series):
    return Counter(series.tolist())


def top_str(counter, n=3):
    return "、".join(w for w, _ in counter.most_common(n))


# ---------------------------------------------------------------------------
# 图表 9：观点词长度分布（正面 vs 负面 双直方图）
# ---------------------------------------------------------------------------
def chart9():
    announce("图9 观点词长度分布", (DATA, ["Opinion", "word_len", "Polarities"]))
    buckets = ["1", "2", "3", "4", "5及以上"]

    def bucketize(L):
        return "5及以上" if L >= 5 else str(L)

    pos = DATA[DATA["Polarities"] == "正面"].copy()
    neg = DATA[DATA["Polarities"] == "负面"].copy()
    max_len = int(DATA["word_len"].max())

    pos_c = Counter(pos["word_len"].map(bucketize))
    neg_c = Counter(neg["word_len"].map(bucketize))
    pos_total, neg_total = len(pos), len(neg)

    x = np.arange(len(buckets))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 6.5))
    pb = ax.bar(x - width / 2, [pos_c.get(b, 0) for b in buckets], width,
                color=POS_COLOR, label="正面观点词", edgecolor="white")
    nb = ax.bar(x + width / 2, [neg_c.get(b, 0) for b in buckets], width,
                color=NEG_COLOR, label="负面观点词", edgecolor="white")
    for rect, b, tot in list(zip(pb, buckets, [pos_total] * 5)) + list(zip(nb, buckets, [neg_total] * 5)):
        cnt = int(rect.get_height())
        ax.text(rect.get_x() + rect.get_width() / 2, cnt,
                f"{cnt}\n{cnt / tot * 100:.1f}%", ha="center", va="bottom", fontsize=9.5)

    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_xlabel("观点词字数")
    ax.set_ylabel("数量")
    ax.set_title(f"观点词长度分布（正面 vs 负面）——5及以上（实际最大 {max_len} 字）")
    ax.legend(title="极性")

    pos_avg = pos["word_len"].mean()
    neg_avg = neg["word_len"].mean()
    note = f"正面观点词平均长度 {pos_avg:.2f} 字，负面观点词平均长度 {neg_avg:.2f} 字"
    ax.text(0.5, -0.14, note, transform=ax.transAxes, ha="center", fontsize=11.5, color="#333333")

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out = os.path.join(CHART_DIR, "09_opinion_word_length_dist.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"    [OK] 已保存 {out}")
    return {"pos_avg": pos_avg, "neg_avg": neg_avg, "max_len": max_len,
            "pos_total": pos_total, "neg_total": neg_total,
            "pos_c": pos_c, "neg_c": neg_c,
            "pos_median": pos["word_len"].median(), "neg_median": neg["word_len"].median(),
            "pos_max": int(pos["word_len"].max()), "neg_max": int(neg["word_len"].max())}


# ---------------------------------------------------------------------------
# 词云辅助：生成一张子图
# ---------------------------------------------------------------------------
def draw_wordcloud(ax, counter, cmap, title):
    if not counter:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", fontsize=16)
        ax.set_title(title)
        ax.axis("off")
        return
    items = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)[:50]
    wc = WordCloud(font_path=FONT_PATH, background_color="white",
                   colormap=cmap, max_words=50, width=900, height=650,
                   prefer_horizontal=0.9).generate_from_frequencies(dict(items))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=12.5)


def chart10():
    announce("图10 正面观点词云", (DATA, ["Opinion", "Polarities", "Categories"]))
    pos = DATA[DATA["Polarities"] == "正面"]
    all_c = freq_of(pos["Opinion"])
    hot_c = freq_of(pos[pos["Categories"].isin(HOT_CATS)]["Opinion"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    draw_wordcloud(axes[0], all_c, "Greens",
                   f"全量正面观点词云（{len(pos)} 词次 / {len(all_c)} 种）\nTop3：{top_str(all_c)}")
    draw_wordcloud(axes[1], hot_c, "Greens",
                   f"差评重灾区(包装/气味/服务)正面观点词云（{int((pos['Categories'].isin(HOT_CATS)).sum())} 词次）\nTop3：{top_str(hot_c)}")
    fig.suptitle("正面观点词云：全量 vs 差评重灾区专题", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(CHART_DIR, "10_positive_opinion_wordcloud.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"    [OK] 已保存 {out}")
    return all_c, hot_c


def chart11():
    announce("图11 负面观点词云", (DATA, ["Opinion", "Polarities", "Categories"]))
    neg = DATA[DATA["Polarities"] == "负面"]
    all_c = freq_of(neg["Opinion"])
    hot_c = freq_of(neg[neg["Categories"].isin(HOT_CATS)]["Opinion"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    draw_wordcloud(axes[0], all_c, "Reds",
                   f"全量负面观点词云（{len(neg)} 词次 / {len(all_c)} 种）\nTop3：{top_str(all_c)}")
    draw_wordcloud(axes[1], hot_c, "Reds",
                   f"差评重灾区(包装/气味/服务)负面观点词云（{int((neg['Categories'].isin(HOT_CATS)).sum())} 词次）\nTop3：{top_str(hot_c)}")
    fig.suptitle("负面观点词云：全量 vs 差评重灾区痛点", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(CHART_DIR, "11_negative_opinion_wordcloud.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"    [OK] 已保存 {out}")
    return all_c, hot_c


# ---------------------------------------------------------------------------
# 图表 12：分属性观点词 Top5 对比（2x2 蝴蝶图）
# ---------------------------------------------------------------------------
def chart12():
    announce("图12 分属性观点词Top5", (DATA, ["Opinion", "Polarities", "Categories"]))
    top4 = ["整体", "使用体验", "功效", "价格"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.ravel()

    for ax, cat in zip(axes, top4):
        sub = DATA[DATA["Categories"] == cat]
        tot = len(labels[labels["Categories"] == cat])
        neg_n = int((labels[(labels["Categories"] == cat)]["Polarities"] == "负面").sum())
        neg_rate = neg_n / tot * 100 if tot else 0

        pos5 = freq_of(sub[sub["Polarities"] == "正面"]["Opinion"]).most_common(5)
        neg5 = freq_of(sub[sub["Polarities"] == "负面"]["Opinion"]).most_common(5)

        y = np.arange(5)
        maxp = max([c for _, c in pos5], default=1)
        maxn = max([c for _, c in neg5], default=1)
        span = max(maxp, maxn)
        thresh = span * 0.14   # 短于此长度的条，词名放到条外侧，避免挤在中心线

        # 正面：向左（绿）
        for i, (w, c) in enumerate(pos5):
            ax.barh(i, -c, color=POS_COLOR, edgecolor="white", height=0.7)
            if c >= thresh:
                ax.text(-c / 2, i, w, ha="center", va="center", color="white", fontsize=10, fontweight="bold")
                ax.text(-c - 1, i, str(c), ha="right", va="center", color=POS_COLOR, fontsize=9.5)
            else:
                ax.text(-c - 1, i, f"{w} {c}", ha="right", va="center", color=POS_COLOR, fontsize=9.5)
        # 负面：向右（红）
        for i, (w, c) in enumerate(neg5):
            ax.barh(i, c, color=NEG_COLOR, edgecolor="white", height=0.7)
            if c >= thresh:
                ax.text(c / 2, i, w, ha="center", va="center", color="white", fontsize=10, fontweight="bold")
                ax.text(c + 1, i, str(c), ha="left", va="center", color=NEG_COLOR, fontsize=9.5)
            else:
                ax.text(c + 1, i, f"{w} {c}", ha="left", va="center", color=NEG_COLOR, fontsize=9.5)

        ax.axvline(0, color="#888888", linewidth=1)
        ax.set_xlim(-maxp * 1.5, maxn * 1.5)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{i+1}" for i in y], fontsize=8)
        ax.set_xlabel("← 正面观点词（绿）    负面观点词（红） →")
        ax.set_title(f"{cat}（总提及 {tot}，负面率 {neg_rate:.1f}%）", fontsize=13)
        ax.grid(False)

    fig.suptitle("Top4 属性的正 / 负面观点词 Top5 对比", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(CHART_DIR, "12_top_category_opinions.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"    [OK] 已保存 {out}")


# ---------------------------------------------------------------------------
# 观点词词频 CSV
# ---------------------------------------------------------------------------
def build_freq_csv():
    announce("词频表 opinion_word_freq.csv", (DATA, ["Opinion", "Polarities", "Categories", "word_len"]))
    rows = []
    for (term, pol), g in DATA.groupby(["Opinion", "Polarities"]):
        top_cat = g["Categories"].value_counts().index[0]
        rows.append({
            "OpinionTerm": term,
            "Polarity": pol,
            "Count": len(g),
            "AvgLength": round(g["word_len"].mean(), 2),
            "TopCategory": top_cat,
        })
    df = pd.DataFrame(rows).sort_values(["Polarity", "Count"], ascending=[True, False])
    csv_path = os.path.join(BASE_DIR, "opinion_word_freq.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"    [OK] 词频表已写入 {csv_path}（{len(df)} 行）")
    return df


# ---------------------------------------------------------------------------
# 汇总报告
# ---------------------------------------------------------------------------
def write_summary(len_info, pos_all, neg_all, hot_neg, freq_df):
    L = []
    L.append("# 情感与观点词分析总结（eda_sentiment）\n")
    L.append(f"> 有效观点词 {len(DATA)} 个（过滤停用词后）| 全局配色：正面=绿 / 负面=红 / 中性=灰\n")

    # 一、长度分布
    L.append("## 一、观点词长度分布对比（正面 vs 负面）\n")
    L.append("| 指标 | 正面 | 负面 |")
    L.append("|------|------|------|")
    L.append(f"| 样本数 | {len_info['pos_total']} | {len_info['neg_total']} |")
    L.append(f"| 平均长度(字) | {len_info['pos_avg']:.2f} | {len_info['neg_avg']:.2f} |")
    L.append(f"| 中位数(字) | {len_info['pos_median']:.0f} | {len_info['neg_median']:.0f} |")
    L.append(f"| 最长(字) | {len_info['pos_max']} | {len_info['neg_max']} |")
    L.append("")
    L.append(f"> 观点词以 2~4 字为主，最长 {len_info['max_len']} 字；"
             f"负面平均长度（{len_info['neg_avg']:.2f}）略高于正面（{len_info['pos_avg']:.2f}），"
             f"吐槽时表达更具体、更长。\n")

    # 二、Top20 高频
    L.append("## 二、正面 / 负面 Top20 高频观点词\n")
    L.append("| 排名 | 正面观点词 | 次数 | 负面观点词 | 次数 |")
    L.append("|------|-----------|------|-----------|------|")
    pt, nt = pos_all.most_common(20), neg_all.most_common(20)
    for i in range(20):
        pw, pc = pt[i] if i < len(pt) else ("", "")
        nw, nc = nt[i] if i < len(nt) else ("", "")
        L.append(f"| {i+1} | {pw} | {pc} | {nw} | {nc} |")
    L.append("")

    # 三、差评重灾区 Top10 负面观点词
    L.append("## 三、差评重灾区（包装/气味/服务）各自 Top10 负面观点词\n")
    for cat in HOT_CATS:
        sub = DATA[(DATA["Polarities"] == "负面") & (DATA["Categories"] == cat)]
        cnt = freq_of(sub["Opinion"])
        total = int((labels["Categories"] == cat).sum())
        negn = int((labels[(labels["Categories"] == cat)]["Polarities"] == "负面").sum())
        L.append(f"**{cat}**（总提及 {total}，负面 {negn}，负面率 {negn/total*100:.1f}%）：")
        top = cnt.most_common(10)
        L.append("　" + "、".join(f"{w}({c})" for w, c in top) if top else "　（无负面观点词）")
        L.append("")

    # 四、一句话结论
    L.append("## 四、一句话结论\n")
    hot_neg_all = freq_of(DATA[(DATA["Polarities"] == "负面") & (DATA["Categories"].isin(HOT_CATS))]["Opinion"])
    L.append(
        f"- **用户表扬时最常用**：{top_str(pos_all)}；\n"
        f"- **用户吐槽时最常用**：{top_str(neg_all)}；\n"
        f"- **差评重灾区痛点关键词**：{top_str(hot_neg_all)}"
        f"（包装看“做工/破损”、气味看“难闻”、服务看“态度差”是主要负向来源）。"
    )
    L.append("")

    md = "\n".join(L)
    md_path = os.path.join(BASE_DIR, "eda_sentiment_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\n[OK] 汇总报告已写入 {md_path}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    len_info = chart9()
    pos_all, _ = chart10()
    neg_all, hot_neg = chart11()
    chart12()
    freq_df = build_freq_csv()
    write_summary(len_info, pos_all, neg_all, hot_neg, freq_df)
    print("\n" + "=" * 60)
    print(f"[完成] 图表 9~12 已生成于 {CHART_DIR}")


if __name__ == "__main__":
    main()
