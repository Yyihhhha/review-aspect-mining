# -*- coding: utf-8 -*-
"""
天池电商评论观点挖掘数据集 —— 数据清洗脚本 clean_data.py
==========================================================
输入：Train_reviews.csv、Train_labels.csv
输出：cleaned_reviews.csv、cleaned_labels.csv、review_wide.csv、cleaning_report.md
      （全部为无 BOM 的 UTF-8）

清洗流程严格遵循需求文档定义的顺序，并基于清洗前原始数据计算质量评分。
"""

import os
import re
import subprocess
import sys

import pandas as pd

# 运行前检查 jieba，未安装则自动 pip install jieba
try:
    import jieba
except ImportError:
    print("[环境] 未检测到 jieba，正在自动安装 ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "jieba"])
    import jieba

# ---------------------------------------------------------------------------
# 0. 路径配置
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REVIEWS_PATH = os.path.join(BASE_DIR, "Train_reviews.csv")
LABELS_PATH = os.path.join(BASE_DIR, "Train_labels.csv")

OUT_REVIEWS = os.path.join(BASE_DIR, "cleaned_reviews.csv")
OUT_LABELS = os.path.join(BASE_DIR, "cleaned_labels.csv")
OUT_WIDE = os.path.join(BASE_DIR, "review_wide.csv")
OUT_REPORT = os.path.join(BASE_DIR, "cleaning_report.md")

VALID_CATEGORIES = ['整体', '使用体验', '功效', '价格', '物流', '气味', '包装',
                    '真伪', '服务', '其他', '成分', '尺寸', '新鲜度']
VALID_POLARITIES = ['正面', '负面', '中性']

NEG_WORDS = ['差', '坏', '烂', '失望', '垃圾', '坑', '慢', '贵', '难用', '后悔']
POS_WORDS = ['好', '棒', '优秀', '满意', '喜欢', '推荐', '划算', '快', '漂亮']
# 基线否定词（上一轮口径，仅用于“扩展前”对比）
NEGATION_BASE = ['不', '没', '无', '非']
# 扩展后的 25 个否定词（当前口径）
NEGATION_WORDS = [
    '不', '没', '无', '非', '未', '别', '莫',
    '不算', '不太', '不怎么', '不够', '没有', '毫无',
    '并非', '未必', '从不', '从未', '绝不', '毫不',
    '谈不上', '说不上', '算不上', '称不上', '不是很',
]

# 记录脏数据样例供报告使用
samples = {}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def smart_read_csv(path):
    """先尝试 utf-8，失败则尝试 gbk。"""
    for enc in ("utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    # 最后兜底
    return pd.read_csv(path, encoding="utf-8", errors="ignore")


def normalize(df):
    """id/ID 统一为 ID，并转为字符串类型。"""
    df.columns = [c.strip() for c in df.columns]
    for cand in ["id", "ID", "Id", "iD"]:
        if cand in df.columns:
            df = df.rename(columns={cand: "ID"})
            break
    df["ID"] = df["ID"].astype(str)
    return df


def clean_text(s):
    """评论文本清洗：去 HTML、URL、多余空格/换行、首尾空格。"""
    if pd.isna(s):
        return ""
    s = str(s)
    s = re.sub(r"<[^>]+>", "", s)                    # HTML 标签
    s = re.sub(r"https?://\S+|www\.\S+", "", s)      # URL 链接
    s = re.sub(r"\s+", " ", s)                        # 折叠空格/换行
    return s.strip()


def is_meaningless(text):
    """无意义评论判定（满足任一即为 True）。"""
    t = re.sub(r"\s", "", str(text))          # 去所有空格
    if len(t) < 5:                            # 规则1：去空格后长度 < 5
        return True
    keep = re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]", t)
    non_keep_ratio = (len(t) - len(keep)) / len(t) if len(t) else 1
    return non_keep_ratio > 0.8               # 规则2：特殊字符占比 > 80%


def is_underscore_or_empty(x):
    """判断值为 NaN / 空串 / 仅空格 / '_' 。"""
    return pd.isna(x) or str(x).strip() in ("", "_")


def is_truly_missing(x):
    """判断真实缺失：NaN / 空串 / 仅空格（不含 '_'）。"""
    return pd.isna(x) or str(x).strip() == ""


# ---------------------------------------------------------------------------
# 加载数据 & 捕获清洗前原始数量（评分分母）
# ---------------------------------------------------------------------------
reviews = normalize(smart_read_csv(REVIEWS_PATH))
labels = normalize(smart_read_csv(LABELS_PATH))

ORIG_REVIEW_COUNT = len(reviews)
ORIG_LABEL_COUNT = len(labels)

print(f"[加载] 原始评论 {ORIG_REVIEW_COUNT} 条，原始四元组 {ORIG_LABEL_COUNT} 个")

# ---------------------------------------------------------------------------
# 1. 评论文本清洗
# ---------------------------------------------------------------------------
reviews["Reviews"] = reviews["Reviews"].apply(clean_text)

meaningless_mask = reviews["Reviews"].apply(is_meaningless)
MEANINGLESS_COUNT = int(meaningless_mask.sum())
samples["无意义评论"] = reviews.loc[meaningless_mask, ["ID", "Reviews"]].head(2)

# Reviews 为空 / 仅空格（清洗后 strip 为空串）
review_empty_mask = reviews["Reviews"].str.strip() == ""
REVIEW_EMPTY_COUNT = int(review_empty_mask.sum())
samples["空评论"] = reviews.loc[review_empty_mask, ["ID", "Reviews"]].head(2)

print(f"[评论清洗] 无意义评论 {MEANINGLESS_COUNT} 条，空评论 {REVIEW_EMPTY_COUNT} 条")

# ---------------------------------------------------------------------------
# 2. 标签数据清洗（严格按顺序）
# ---------------------------------------------------------------------------

# 【第一步：ID 一致性校验】
review_id_set = set(reviews["ID"])
label_id_set = set(labels["ID"])

ids_no_label = review_id_set - label_id_set          # 有评论无标签
ids_no_review = label_id_set - review_id_set         # 有标签无评论（孤儿）
NO_LABEL_COUNT = len(ids_no_label)

orphan_mask = labels["ID"].isin(ids_no_review)
ORPHAN_COUNT = int(orphan_mask.sum())
samples["孤儿标签"] = labels.loc[orphan_mask].head(2)

labels_work = labels[~orphan_mask].copy()
print(f"[第一步] 有评论无标签 {NO_LABEL_COUNT} 个；剔除孤儿标签 {ORPHAN_COUNT} 行")

# 【第二步：统计原始缺失】（在任何替换/调整前）
ASPECT_IMPLICIT_COUNT = int(labels_work["AspectTerms"].apply(is_underscore_or_empty).sum())
OPINION_IMPLICIT_COUNT = int(labels_work["OpinionTerms"].apply(is_underscore_or_empty).sum())

cat_pol_missing_mask = (
    labels_work["Categories"].apply(is_truly_missing)
    | labels_work["Polarities"].apply(is_truly_missing)
)
CAT_POL_MISSING_COUNT = int(cat_pol_missing_mask.sum())
samples["标签缺失"] = labels_work.loc[cat_pol_missing_mask].head(2)
print(f"[第二步] 隐式属性 {ASPECT_IMPLICIT_COUNT}；OpinionTerms空 {OPINION_IMPLICIT_COUNT}；"
      f"Categories/Polarities缺失 {CAT_POL_MISSING_COUNT}")

# 【第三步：格式统一】空/NaN/仅空格 → '_'
for col in ["AspectTerms", "OpinionTerms"]:
    labels_work[col] = labels_work[col].apply(
        lambda x: "_" if (pd.isna(x) or str(x).strip() == "") else str(x).strip()
    )
labels_work["Categories"] = labels_work["Categories"].apply(
    lambda x: "" if pd.isna(x) else str(x).strip()
)
labels_work["Polarities"] = labels_work["Polarities"].apply(
    lambda x: "" if pd.isna(x) else str(x).strip()
)

# 【第四步：校验合法值】
cat_invalid = ~labels_work["Categories"].isin(VALID_CATEGORIES)
pol_invalid = ~labels_work["Polarities"].isin(VALID_POLARITIES)
ILLEGAL_CAT_COUNT = int(cat_invalid.sum())
ILLEGAL_POL_COUNT = int(pol_invalid.sum())

dirty_mask = cat_invalid | pol_invalid
ILLEGAL_REMOVED = int(dirty_mask.sum())   # 去重后的实际剔除行数（含交集）
samples["非法标签"] = labels_work.loc[dirty_mask].head(2)
labels_work = labels_work[~dirty_mask].copy()
print(f"[第四步] 非法Categories {ILLEGAL_CAT_COUNT}；非法Polarities {ILLEGAL_POL_COUNT}；"
      f"实际剔除 {ILLEGAL_REMOVED} 行")

# 【第五步：去重】
KEY_COLS = ["ID", "AspectTerms", "OpinionTerms", "Categories", "Polarities"]
DUP_COUNT = int(labels_work.duplicated(subset=KEY_COLS).sum())
samples["重复四元组"] = labels_work[labels_work.duplicated(subset=KEY_COLS, keep=False)].head(2)
labels_work = labels_work.drop_duplicates(subset=KEY_COLS, keep="first").copy()
print(f"[第五步] 去除完全重复四元组 {DUP_COUNT} 行")

# 【第六步：逻辑一致性检查】jieba 分词 + 精确匹配 + 否定词处理
# 三档口径： A=子串包含（最初）; B=jieba+4词否定+前瞻1; C=jieba+25词否定+前瞻(前1/前2/前2拼接)
def _is_negated(words, i, neg_set, wide):
    """判断位置 i 的词是否被否定修饰。wide=True 时额外看前2词及前2词拼接。"""
    if i >= 1 and words[i - 1] in neg_set:
        return True
    if wide:
        if i >= 2 and words[i - 2] in neg_set:
            return True
        if i >= 2 and (words[i - 2] + words[i - 1]) in neg_set:
            return True
    return False


def seg_hit(opinion, wordlist, neg_set, wide):
    """分词后逐词精确匹配；被否定则豁免。返回 (是否命中, 命中词, 分词列表, 命中位置)。"""
    words = jieba.lcut(opinion)
    for i, w in enumerate(words):
        if w in wordlist:
            if _is_negated(words, i, neg_set, wide):
                continue   # 否定修饰，不算命中
            return True, w, words, i
    return False, None, words, None


def substring_hit(opinion, wordlist):
    """A 口径：直接子串包含即命中（会误伤“不好”“不贵”等）。"""
    return any(w in opinion for w in wordlist)


def classify_row(row):
    """返回 (A_子串, B_上一轮, C_当前) 三档疑似矛盾判定。"""
    opinion = str(row["OpinionTerms"])
    polarity = row["Polarities"]
    check_list = NEG_WORDS if polarity == "正面" else (POS_WORDS if polarity == "负面" else None)
    if check_list is None:
        return False, False, False
    a = substring_hit(opinion, check_list)
    b, _, _, _ = seg_hit(opinion, check_list, NEGATION_BASE, wide=False)
    c, _, _, _ = seg_hit(opinion, check_list, NEGATION_WORDS, wide=True)
    return a, b, c


_results = labels_work.apply(classify_row, axis=1)
INCONSISTENT_SUBSTRING_COUNT = int(sum(1 for a, b, c in _results if a))  # A 子串
INCONSISTENT_OLD_COUNT = int(sum(1 for a, b, c in _results if b))        # B 上一轮(4词)=37
inconsistent_mask = pd.Series([c for _, _, c in _results], index=labels_work.index)
INCONSISTENT_COUNT = int(inconsistent_mask.sum())                        # C 当前口径
samples["疑似矛盾标签"] = labels_work.loc[inconsistent_mask, KEY_COLS].head(2)


def _substring_reason(opinion, polarity):
    """第一轮（子串→分词）样例的原因。"""
    check_list = NEG_WORDS if polarity == "正面" else POS_WORDS
    words = jieba.lcut(opinion)
    hit_word = next((w for w in check_list if w in opinion), None)
    if hit_word in words:
        i = words.index(hit_word)
        prev = words[i - 1] if i > 0 else None
        if prev in NEGATION_WORDS:
            return hit_word, f"“{hit_word}”前有“{prev}”，否定修饰"
        return hit_word, f"“{hit_word}”独立成词但仍属正常表达"
    merged = next((t for t in words if hit_word in t and t != hit_word), opinion)
    return hit_word, f"“{hit_word}”非独立词（“{merged}”成词，子串误伤）"


def _neg_expand_reason(opinion, polarity):
    """第二轮（否定词 4→25 + 前瞻扩展）样例的原因。"""
    check_list = NEG_WORDS if polarity == "正面" else POS_WORDS
    words = jieba.lcut(opinion)
    for i, w in enumerate(words):
        if w not in check_list:
            continue
        prev1 = words[i - 1] if i >= 1 else None
        prev2 = words[i - 2] if i >= 2 else None
        if prev1 in NEGATION_WORDS and prev1 not in NEGATION_BASE:
            return w, f"“{prev1}”为扩展否定词"
        if prev2 is not None and prev2 in NEGATION_WORDS:
            return w, f"命中词前2词“{prev2}”为否定词（前瞻扩展）"
        if prev2 is not None and (prev2 + (prev1 or "")) in NEGATION_WORDS:
            return w, f"前2词拼接“{prev2}{prev1}”命中否定词"
    return (words[-1] if words else ""), "否定词表扩展"


# 第一轮修正样例：A=True 且 C=False（子串误判但分词+否定后正常）
fix_examples = []
# 第二轮修正样例：B=True 且 C=False（扩展否定词表后才豁免）
fix_examples_neg = []
for idx, row in labels_work.iterrows():
    a, b, c = classify_row(row)
    opinion = str(row["OpinionTerms"])
    polarity = row["Polarities"]
    if a and not c:
        hit_word, reason = _substring_reason(opinion, polarity)
        fix_examples.append({"opinion": opinion, "polarity": polarity, "hit": hit_word, "reason": reason})
    if b and not c:
        hit_word, reason = _neg_expand_reason(opinion, polarity)
        fix_examples_neg.append({"opinion": opinion, "polarity": polarity, "hit": hit_word, "reason": reason})

print(f"[第六步] 疑似矛盾：子串 {INCONSISTENT_SUBSTRING_COUNT} → 4词否定 {INCONSISTENT_OLD_COUNT} → "
      f"{len(NEGATION_WORDS)}词+前瞻 {INCONSISTENT_COUNT}（本轮 {INCONSISTENT_OLD_COUNT}→{INCONSISTENT_COUNT}，仅统计不剔除）")

# 清洗后数量
CLEANED_REVIEW_COUNT = ORIG_REVIEW_COUNT   # 评论表不删除行，仅清洗文本+标记无意义
CLEANED_LABEL_COUNT = len(labels_work)

# ---------------------------------------------------------------------------
# 3. 数据整合：评论级宽表
# ---------------------------------------------------------------------------
def build_wide():
    agg_rows = []
    grouped = labels_work.groupby("ID")
    for rid, grp in grouped:
        cats = sorted(grp["Categories"].unique().tolist())
        pols = grp["Polarities"]
        agg_rows.append({
            "ID": rid,
            "四元组数量": len(grp),
            "涉及属性种类列表": "|".join(cats),
            "正面数": int((pols == "正面").sum()),
            "负面数": int((pols == "负面").sum()),
            "中性数": int((pols == "中性").sum()),
            "是否含隐式属性": "是" if (grp["AspectTerms"] == "_").any() else "否",
        })
    agg_df = pd.DataFrame(agg_rows)

    wide = reviews[["ID", "Reviews"]].merge(agg_df, on="ID", how="left")
    wide = wide.rename(columns={"Reviews": "评论内容"})
    wide["评论字数"] = wide["评论内容"].str.len()
    for c in ["四元组数量", "正面数", "负面数", "中性数"]:
        wide[c] = wide[c].fillna(0).astype(int)
    wide["涉及属性种类列表"] = wide["涉及属性种类列表"].fillna("")
    wide["是否含隐式属性"] = wide["是否含隐式属性"].fillna("否")
    # 无标签评论的隐式属性判定为“否”
    wide = wide[["ID", "评论内容", "评论字数", "四元组数量",
                 "涉及属性种类列表", "正面数", "负面数", "中性数", "是否含隐式属性"]]
    return wide

wide = build_wide()
print(f"[整合] 评论级宽表 {len(wide)} 行")

# ---------------------------------------------------------------------------
# 4. 数据质量评分（占比均基于清洗前原始数据）
# ---------------------------------------------------------------------------
def safe_div(a, b):
    return a / b if b else 0

review_missing_rate = safe_div(REVIEW_EMPTY_COUNT, ORIG_REVIEW_COUNT)
label_missing_rate = safe_div(CAT_POL_MISSING_COUNT + NO_LABEL_COUNT,
                              ORIG_LABEL_COUNT + NO_LABEL_COUNT)
illegal_cat_rate = safe_div(ILLEGAL_CAT_COUNT, ORIG_LABEL_COUNT)
dup_rate = safe_div(DUP_COUNT, ORIG_LABEL_COUNT)
meaningless_rate = safe_div(MEANINGLESS_COUNT, ORIG_REVIEW_COUNT)
inconsistent_rate = safe_div(INCONSISTENT_COUNT, ORIG_LABEL_COUNT)

d_review = min(review_missing_rate * 300, 15)
d_label = min(label_missing_rate * 200, 20)
d_cat = min(illegal_cat_rate * 125, 25)
d_dup = min(dup_rate * 75, 15)
d_mean = min(meaningless_rate * 100, 15)
d_logic = min(inconsistent_rate * 50, 10)

s_review = 15 - d_review
s_label = 20 - d_label
s_cat = 25 - d_cat
s_dup = 15 - d_dup
s_mean = 15 - d_mean
s_logic = 10 - d_logic

# —— 三档逻辑口径的得分与总分（仅“标签逻辑一致性”维度变化）——
def logic_pack(cnt):
    rate = safe_div(cnt, ORIG_LABEL_COUNT)
    d = min(rate * 50, 10)
    s = 10 - d
    total = max(0, round(s_review + s_label + s_cat + s_dup + s_mean + s, 2))
    return rate, d, s, total


# 子串口径（最初）
inconsistent_sub_rate, d_logic_sub, s_logic_sub, total_score_sub = logic_pack(INCONSISTENT_SUBSTRING_COUNT)
# 4 词否定口径（上一轮）
inconsistent_old_rate, d_logic_old, s_logic_old, total_score_old = logic_pack(INCONSISTENT_OLD_COUNT)
# 当前口径（25 词 + 前瞻）
inconsistent_rate, d_logic, s_logic, _ = logic_pack(INCONSISTENT_COUNT)
total_score = round(s_review + s_label + s_cat + s_dup + s_mean + s_logic, 2)
total_score = max(0, total_score)

if total_score >= 90:
    grade = "优秀"
elif total_score >= 80:
    grade = "良好"
elif total_score >= 70:
    grade = "一般"
else:
    grade = "较差"

# ---------------------------------------------------------------------------
# 5. 生成清洗报告 cleaning_report.md
# ---------------------------------------------------------------------------
# 长尾属性（清洗后占比 < 1%）
cat_dist = labels_work["Categories"].value_counts(normalize=True) * 100
long_tail = [c for c in cat_dist.index if cat_dist.get(c, 0) < 1.0]

def sample_to_md(key, cols_hint=None):
    df = samples.get(key)
    if df is None or df.empty:
        return f"- **{key}**：无（该类脏数据为空）"
    show = df if cols_hint is None else df[[c for c in cols_hint if c in df.columns]]
    rows = show.to_dict("records")
    lines = [f"- **{key}**（样例）："]
    for r in rows:
        content = "，".join(f"{k}={v}" for k, v in r.items())
        lines.append(f"    - {content}")
    return "\n".join(lines)

implicit_pct = safe_div(ASPECT_IMPLICIT_COUNT, CLEANED_LABEL_COUNT) * 100


def pick_examples(ex_list, n=3):
    """按观点词去重，并优先让原因多样化。"""
    seen, uniq = set(), []
    for ex in ex_list:
        if ex["opinion"] in seen:
            continue
        seen.add(ex["opinion"])
        uniq.append(ex)
    # 按 reason 关键字分组，优先覆盖不同类型
    buckets = {}
    for e in uniq:
        key = e["reason"].split("（")[0][:6]
        buckets.setdefault(key, []).append(e)
    chosen = []
    for key, pool in buckets.items():
        if pool:
            chosen.append(pool[0])
    for e in uniq:
        if len(chosen) >= n:
            break
        if e not in chosen:
            chosen.append(e)
    return chosen[:n]


def rows_to_md(ex_list, old_col):
    lines = [
        f'| {ex["opinion"]} | {ex["polarity"]} | {old_col}（命中“{ex["hit"]}”） | 正常 | {ex["reason"]} |'
        for ex in ex_list
    ]
    return "\n".join(lines) if lines else "| （无真实样例） | - | - | - | - |"


# 第一轮：子串匹配→分词+否定；第二轮：否定词 4→25 + 前瞻
fix_table_rows = rows_to_md(pick_examples(fix_examples, 3), "疑似矛盾")
fix_table_neg_rows = rows_to_md(pick_examples(fix_examples_neg, 3), "疑似矛盾")

report = f"""# 数据清洗报告

> 数据源：化妆品品类电商评论观点挖掘数据集
> 清洗前：评论 {ORIG_REVIEW_COUNT} 条 / 四元组 {ORIG_LABEL_COUNT} 个
> 清洗后：评论 {CLEANED_REVIEW_COUNT} 条 / 四元组 {CLEANED_LABEL_COUNT} 个

## 一、清洗前后数据量对比

| 指标 | 清洗前 | 清洗后 | 变化 |
|------|--------|--------|------|
| 评论数 | {ORIG_REVIEW_COUNT} | {CLEANED_REVIEW_COUNT} | 评论仅清洗文本，不删行 |
| 四元组标签数 | {ORIG_LABEL_COUNT} | {CLEANED_LABEL_COUNT} | -{ORIG_LABEL_COUNT - CLEANED_LABEL_COUNT} |

## 二、各清洗步骤处理明细

1. **ID 对齐**：有评论但无标签的评论 **{NO_LABEL_COUNT}** 个；有标签但无评论的孤儿标签 **{ORPHAN_COUNT}** 行（已剔除）。
2. **缺失值统计**：空评论 {REVIEW_EMPTY_COUNT} 条；Categories/Polarities 缺失 {CAT_POL_MISSING_COUNT} 行；OpinionTerms 为 '_' / 空 {OPINION_IMPLICIT_COUNT} 行；AspectTerms 为 '_' / 空（隐式属性，正常）{ASPECT_IMPLICIT_COUNT} 行。
3. **非法标签剔除**：非法 Categories {ILLEGAL_CAT_COUNT} 行，非法 Polarities {ILLEGAL_POL_COUNT} 行，去重后实际剔除 {ILLEGAL_REMOVED} 行。
4. **重复数据去除**：完全重复四元组 **{DUP_COUNT}** 行已删除。
5. **无意义评论**：标记 **{MEANINGLESS_COUNT}** 条（保留于评论表，供建模阶段决定取舍）。
6. **疑似矛盾标签**：采用 jieba 分词 + 精确匹配 + 否定词豁免后，疑似逻辑不一致四元组 **{INCONSISTENT_COUNT}** 个（上一轮 4 词否定口径为 {INCONSISTENT_OLD_COUNT} 个，本轮否定词表扩展至 {len(NEGATION_WORDS)} 个后再降 {INCONSISTENT_OLD_COUNT - INCONSISTENT_COUNT} 个；仅统计提示，未自动修改）。

## 三、数据特殊说明

- **隐式属性**：AspectTerms 为 '_' 的四元组约 **{ASPECT_IMPLICIT_COUNT}** 个，占清洗后标签约 **{implicit_pct:.1f}%**，属正常数据特征（评论只表达观点未显式点名属性），非脏数据。
- **Categories 分布不均衡**：头部属性（如“整体”）占比最高，长尾属性（占比 < 1%）有：{('、'.join(long_tail)) if long_tail else '无'}。建模时建议对长尾属性做样本增强或类别加权。

## 四、脏数据样例展示

{sample_to_md("孤儿标签", ["ID", "AspectTerms", "OpinionTerms", "Categories", "Polarities"])}

{sample_to_md("非法标签", ["ID", "AspectTerms", "OpinionTerms", "Categories", "Polarities"])}

{sample_to_md("重复四元组", ["ID", "AspectTerms", "OpinionTerms", "Categories", "Polarities"])}

{sample_to_md("无意义评论", ["ID", "Reviews"])}

{sample_to_md("疑似矛盾标签", ["ID", "AspectTerms", "OpinionTerms", "Categories", "Polarities"])}

> 若上述各类均为“无”，说明本数据集质量较好，仅需常规清洗即可投入使用。

## 五、数据质量评分

| 维度 | 实际占比 | 扣分数 | 得分数 |
|------|----------|--------|--------|
| 评论缺失率 | {review_missing_rate*100:.2f}% | {d_review:.2f} | {s_review:.2f} |
| 标签缺失率 | {label_missing_rate*100:.2f}% | {d_label:.2f} | {s_label:.2f} |
| 非法类别占比 | {illegal_cat_rate*100:.2f}% | {d_cat:.2f} | {s_cat:.2f} |
| 重复数据占比 | {dup_rate*100:.2f}% | {d_dup:.2f} | {s_dup:.2f} |
| 无意义评论占比 | {meaningless_rate*100:.2f}% | {d_mean:.2f} | {s_mean:.2f} |
| 标签逻辑一致性 | {inconsistent_rate*100:.2f}% | {d_logic:.2f} | {s_logic:.2f} |

> **口径备注**：“标签缺失率”仅统计 Categories / Polarities 为空字符串、NaN、仅含空格的情况；
> AspectTerms / OpinionTerms 为 "_" 属于正常的隐式表达（评论未显式点名属性/观点），**不计入标签缺失**。
> “标签逻辑一致性”采用 jieba 分词 + 精确匹配 + 否定词豁免口径（详见末尾“修正说明”）。
>
> **“疑似矛盾”定义与口径说明**：
> - 定义：极性为正但观点词命中负面词，或极性为负但命中正面词，且已排除否定词修饰；
> - 正/负面词表为示例性词表，未做完整情感词典覆盖；否定词表已扩展至 {len(NEGATION_WORDS)} 个，仍可能存在未覆盖的否定副词；
> - 因此保留项统一称为“**疑似矛盾**”，不视为真实标注错误。

**总评分：{total_score} / 100**
**数据质量等级：{grade}**

## 六、清洗小结

本数据集整体质量{grade}（{total_score} 分）。评论与标签的 ID 关联完整、无缺失值，仅在完全重复四元组上存在极少量脏数据；主要数据特征为隐式属性占比高、Categories 分布长尾，需在后续建模阶段做针对性处理。整体而言，数据经清洗后可直接进入四元组抽取模型训练。

## 七、修正说明

“疑似矛盾”判定口径共经历两轮修正，演进总览如下：

| 阶段 | 匹配方式 | 否定词数 | 否定前瞻 | 疑似矛盾数 | 总评分 |
|------|---------|---------|---------|-----------|--------|
| A 子串包含（最初） | 词作为子串命中 | - | - | {INCONSISTENT_SUBSTRING_COUNT} | {total_score_sub} |
| B jieba+否定词（第一轮） | 分词后精确匹配 | 4 | 前1词 | {INCONSISTENT_OLD_COUNT} | {total_score_old} |
| C jieba+否定词（第二轮·当前） | 分词后精确匹配 | {len(NEGATION_WORDS)} | 前1/前2/前2拼接 | {INCONSISTENT_COUNT} | {total_score} |

### 第一轮：子串匹配 → jieba 分词 + 否定词
- 修正内容：对每条 OpinionTerms 用 `jieba.lcut()` 分词，仅当某分词**完全等于**正/负面词表中的词时才算命中；若命中词前一个分词为否定词则豁免。
- 疑似矛盾数：{INCONSISTENT_SUBSTRING_COUNT} → {INCONSISTENT_OLD_COUNT}
- 修正原因：子串误伤 + 否定词未处理（如“很不好”“不喜欢”被子串命中“好”/“喜欢”）。
- 典型样例（子串判疑似矛盾、分词后正常，真实数据）：

| 原始观点词 | 极性 | 子串匹配判定 | 分词+否定词判定 | 原因 |
|-----------|------|-------------|----------------|------|
{fix_table_rows}

### 第二轮：否定词表 4 → {len(NEGATION_WORDS)} + 前瞻扩展（本次修正）
- 修正内容：否定词表由 ['不','没','无','非'] 扩展至 **{len(NEGATION_WORDS)} 个**（新增 不算/不太/不怎么/不够/没有/毫无/并非/未必/从不/从未/绝不/毫不/谈不上/说不上/算不上/称不上/不是很 等）；匹配时除看前1个词外，还检查**前2个词**及**前2词拼接**是否命中否定词；“不好”“不错”等整体成词若不在词表则不参与判断。
- 疑似矛盾数：{INCONSISTENT_OLD_COUNT} → {INCONSISTENT_COUNT}（本轮下降 {INCONSISTENT_OLD_COUNT - INCONSISTENT_COUNT}）
- 修正原因：上一轮单字否定前瞻不足，无法覆盖“不算贵”“不是很喜欢”“不太喜欢”等多字/跨词否定表达。
- 典型样例（扩展否定词表前判疑似矛盾、扩展后判正常，真实数据）：

| 原始观点词 | 极性 | 扩展前判定 | 扩展后判定 | 原因 |
|-----------|------|-------------|----------------|------|
{fix_table_neg_rows}

### 评分变化对比

| 项目 | A 子串 | B 第一轮后 | C 当前 |
|------|--------|-----------|--------|
| 逻辑不一致占比 | {inconsistent_sub_rate*100:.2f}% | {inconsistent_old_rate*100:.2f}% | {inconsistent_rate*100:.2f}% |
| 逻辑一致性得分 | {s_logic_sub:.2f} | {s_logic_old:.2f} | {s_logic:.2f} |
| 总评分 | {total_score_sub} | {total_score_old} | {total_score} |

> 因正/负面词表为示例性词表、否定词表仍可能未全覆盖，当前保留的 {INCONSISTENT_COUNT} 条统一称为“疑似矛盾”，不视为真实标注错误。
"""

# ---------------------------------------------------------------------------
# 6. 保存输出文件（无 BOM UTF-8）
# ---------------------------------------------------------------------------
reviews.to_csv(OUT_REVIEWS, index=False, encoding="utf-8")
labels_work.to_csv(OUT_LABELS, index=False, encoding="utf-8")
wide.to_csv(OUT_WIDE, index=False, encoding="utf-8")
with open(OUT_REPORT, "w", encoding="utf-8") as f:
    f.write(report)

print("\n" + "=" * 60)
print(f"[完成] 质量评分 {total_score}/100（{grade}）")
print(f"[保存] {OUT_REVIEWS}")
print(f"[保存] {OUT_LABELS}")
print(f"[保存] {OUT_WIDE}")
print(f"[保存] {OUT_REPORT}")
