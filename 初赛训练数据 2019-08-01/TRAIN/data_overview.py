# -*- coding: utf-8 -*-
"""
天池电商评论观点挖掘数据集 —— 数据摸底脚本
=================================================
功能：
  1. 读取评论文件与标签文件，打印形状与前5行
  2. 统计评论总数、标签总数、每条评论平均标签数
  3. 统计 Categories（属性种类）分布及占比
  4. 统计 Polarities（极性）分布及占比
  5. 统计 AspectTerms 为空/下划线“_”（隐式属性）的数量与占比
  6. 检查缺失值、重复值
  7. 汇总为可读文本报告 data_overview.txt

使用说明：
  - 直接把 REVIEWS_PATH / LABELS_PATH 改成你的文件路径即可
  - 依赖：pandas（pip install pandas）
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# 配置区：按需修改路径（默认使用当前目录下的 TRAIN 数据）
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REVIEWS_PATH = os.path.join(BASE_DIR, "Train_reviews.csv")
LABELS_PATH = os.path.join(BASE_DIR, "Train_labels.csv")
REPORT_PATH = os.path.join(BASE_DIR, "data_overview.txt")

# 报告内容同时收集到列表，最后写入文件
report_lines = []


def log(msg=""):
    """既打印到控制台，又收集到报告。"""
    print(msg)
    report_lines.append(str(msg))


def to_list(series):
    """把可能含逗号的字符串字段安全处理（此处标签每行一个四元组，无需拆分）。"""
    return series.astype(str).str.strip()


# ---------------------------------------------------------------------------
# 1. 读取数据
# ---------------------------------------------------------------------------
log("=" * 60)
log("天池电商评论观点挖掘数据集 —— 数据摸底报告")
log("=" * 60)
log()

log("【1】数据读取")
log("-" * 60)

try:
    reviews = pd.read_csv(REVIEWS_PATH, encoding="utf-8")
except UnicodeDecodeError:
    reviews = pd.read_csv(REVIEWS_PATH, encoding="gbk")

try:
    labels = pd.read_csv(LABELS_PATH, encoding="utf-8")
except UnicodeDecodeError:
    labels = pd.read_csv(LABELS_PATH, encoding="gbk")

# 统一列名：兼容 ID / id 大小写差异，方便后续按 ID 关联
reviews.columns = [c.strip() for c in reviews.columns]
labels.columns = [c.strip() for c in labels.columns]


def normalize_id_col(df):
    """把 id/ID 列统一重命名为 'ID'。"""
    for cand in ["id", "ID", "Id", "iD"]:
        if cand in df.columns:
            return df.rename(columns={cand: "ID"})
    return df


reviews = normalize_id_col(reviews)
labels = normalize_id_col(labels)

log("评论文件 (Train_reviews.csv):")
log(f"  形状 (行, 列): {reviews.shape}")
log(f"  列名: {list(reviews.columns)}")
log("  前5行:")
log(reviews.head().to_string(index=False))
log()

log("标签文件 (Train_labels.csv):")
log(f"  形状 (行, 列): {labels.shape}")
log(f"  列名: {list(labels.columns)}")
log("  前5行:")
log(labels.head().to_string(index=False))
log()


# ---------------------------------------------------------------------------
# 2. 基础数量统计
# ---------------------------------------------------------------------------
log("【2】基础数量统计")
log("-" * 60)

n_reviews = reviews.shape[0]
n_labels = labels.shape[0]
avg_labels = n_labels / n_reviews if n_reviews else 0

log(f"  评论总数: {n_reviews}")
log(f"  标签(四元组)总数: {n_labels}")
log(f"  每条评论平均标签数: {avg_labels:.2f}")
log()


# ---------------------------------------------------------------------------
# 3. Categories 分布及占比
# ---------------------------------------------------------------------------
log("【3】Categories（属性种类）分布及占比")
log("-" * 60)

if "Categories" in labels.columns:
    cat_counts = labels["Categories"].astype(str).str.strip().value_counts()
    cat_pct = labels["Categories"].astype(str).str.strip().value_counts(normalize=True) * 100
    log(f"  {'类别':<12}{'数量':>8}{'占比':>10}")
    for name, cnt in cat_counts.items():
        pct = cat_pct.get(name, 0)
        log(f"  {str(name):<12}{cnt:>8}{pct:>9.2f}%")
    log(f"  {'合计':<12}{cat_counts.sum():>8}{'100.00%':>10}")
else:
    log("  [警告] 未找到 Categories 列")
log()


# ---------------------------------------------------------------------------
# 4. Polarities 分布及占比
# ---------------------------------------------------------------------------
log("【4】Polarities（极性）分布及占比")
log("-" * 60)

if "Polarities" in labels.columns:
    pol_counts = labels["Polarities"].astype(str).str.strip().value_counts()
    pol_pct = labels["Polarities"].astype(str).str.strip().value_counts(normalize=True) * 100
    log(f"  {'极性':<12}{'数量':>8}{'占比':>10}")
    for name, cnt in pol_counts.items():
        pct = pol_pct.get(name, 0)
        log(f"  {str(name):<12}{cnt:>8}{pct:>9.2f}%")
    log(f"  {'合计':<12}{pol_counts.sum():>8}{'100.00%':>10}")
else:
    log("  [警告] 未找到 Polarities 列")
log()


# ---------------------------------------------------------------------------
# 5. AspectTerms 为空/下划线（隐式属性）统计
# ---------------------------------------------------------------------------
log("【5】AspectTerms 隐式属性（值为 '_' 或空）统计")
log("-" * 60)

if "AspectTerms" in labels.columns:
    aspect = labels["AspectTerms"].astype(str).str.strip()
    # 判定为空：值为 '_'、空字符串、或 pandas 的 NaN
    is_implicit = (aspect == "_") | (aspect == "") | (labels["AspectTerms"].isna())
    n_implicit = int(is_implicit.sum())
    n_total = len(labels)
    pct_implicit = n_implicit / n_total * 100 if n_total else 0

    log(f"  隐式属性数量 (AspectTerms 为 '_'/空): {n_implicit}")
    log(f"  标签总数: {n_total}")
    log(f"  隐式属性占比: {pct_implicit:.2f}%")
    log(f"  显式属性占比: {100 - pct_implicit:.2f}%")
else:
    log("  [警告] 未找到 AspectTerms 列")
log()


# ---------------------------------------------------------------------------
# 6. 缺失值 & 重复值检查
# ---------------------------------------------------------------------------
log("【6】缺失值与重复值检查")
log("-" * 60)

log("  [评论文件] 各列缺失值数量:")
rev_na = reviews.isna().sum()
for col, v in rev_na.items():
    log(f"    {col}: {v}")
log(f"  [评论文件] 整行重复数量: {int(reviews.duplicated().sum())}")
if "ID" in reviews.columns:
    log(f"  [评论文件] ID 重复数量: {int(reviews['ID'].duplicated().sum())}")
log()

log("  [标签文件] 各列缺失值数量:")
lab_na = labels.isna().sum()
for col, v in lab_na.items():
    log(f"    {col}: {v}")
log(f"  [标签文件] 整行重复数量: {int(labels.duplicated().sum())}")
if "ID" in labels.columns:
    log(f"  [标签文件] 关联评论的 ID 去重数: {labels['ID'].nunique()}")
log()

# 关联完整性检查：标签中的 ID 是否都能在评论中找到
if "ID" in reviews.columns and "ID" in labels.columns:
    review_ids = set(reviews["ID"].dropna().tolist())
    label_ids = set(labels["ID"].dropna().tolist())
    orphan = label_ids - review_ids
    log("  [关联检查] 标签存在但评论缺失的 ID 数: "
        f"{len(orphan)}")
    no_label_ids = review_ids - label_ids
    log(f"  [关联检查] 评论存在但无任何标签的 ID 数: {len(no_label_ids)}")
log()


# ---------------------------------------------------------------------------
# 7. 生成报告文件
# ---------------------------------------------------------------------------
log("=" * 60)
log("报告生成完毕！")

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"\n[已保存] 统计报告已写入: {REPORT_PATH}")
