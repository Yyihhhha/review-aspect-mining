# 电商评论观点挖掘与差评归因（review-aspect-mining）

天池化妆品电商评论数据：从评论中抽取观点四元组（属性特征 AspectTerms、观点词 OpinionTerms、属性种类 Categories、观点极性 Polarities），并进一步做差评归因分析，输出可落地的业务改进建议。

## 分析流程（scripts/）

| 步骤 | 脚本 | 产出 |
|------|------|------|
| 数据总览 | `scripts/data_overview.py` | `reports/data_overview.txt` |
| 数据清洗 | `scripts/clean_data.py` | `data/cleaned_*.csv`、`data/review_wide.csv`、`reports/cleaning_report.md` |
| EDA 概览 | `scripts/eda_overview.py` | 图表 1–4、`reports/eda_overview_summary.md` |
| EDA 属性维度 | `scripts/eda_category.py` | 图表 5–8、`data/category_polarity_table.csv`、`reports/eda_category_summary.md` |
| EDA 情感与观点词 | `scripts/eda_sentiment.py` | 图表 9–12、`data/opinion_word_freq.csv`、`reports/eda_sentiment_summary.md` |
| 差评子类细粒度分析 | `scripts/bad_review_subcategory.py` | 图表 13、`data/bad_review_subcategories.csv`、`reports/bad_review_subcategory.md` |
| 差评帕累托分析 | `scripts/pareto_analysis.py` | 图表 14–15、`reports/pareto_analysis.md` |
| 最终报告整合 | `scripts/final_report.py` | `reports/final_report.md` |

运行顺序：`data_overview → clean_data → eda_* → bad_review_subcategory → pareto_analysis → final_report`。

## 项目结构

```
review-aspect-mining/
├── charts/
│   ├── 01_review_length_dist.png
│   ├── 02_quadruple_count_dist.png
│   ├── 03_category_distribution.png
│   ├── 04_category_negative_rate.png
│   ├── 05_implicit_vs_explicit_polarity.png
│   ├── 06_category_cooccurrence_heatmap.png
│   ├── 07_category_polarity_heatmap.png
│   ├── 08_implicit_aspect_ratio.png
│   ├── 09_opinion_word_length_dist.png
│   ├── 10_positive_opinion_wordcloud.png
│   ├── 11_negative_opinion_wordcloud.png
│   ├── 12_top_category_opinions.png
│   ├── 13_bad_review_subcategory_stacked.png
│   ├── 14_pareto_problem_points.png
│   └── 15_pareto_categories.png
├── data/
│   ├── raw/
│   │   ├── Train_reviews.csv
│   │   └── Train_labels.csv
│   ├── cleaned_reviews.csv
│   ├── cleaned_labels.csv
│   ├── review_wide.csv
│   ├── opinion_word_freq.csv
│   ├── category_polarity_table.csv
│   ├── bad_review_subcategories.csv
│   └── 数据说明.md
├── reports/
│   ├── data_overview.txt
│   ├── cleaning_report.md
│   ├── eda_overview_summary.md
│   ├── eda_category_summary.md
│   ├── eda_sentiment_summary.md
│   ├── bad_review_subcategory.md
│   ├── pareto_analysis.md
│   └── final_report.md
└── scripts/
    ├── data_overview.py
    ├── clean_data.py
    ├── eda_overview.py
    ├── eda_category.py
    ├── eda_sentiment.py
    ├── bad_review_subcategory.py
    ├── pareto_analysis.py
    └── final_report.py
```

## 如何运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 按顺序执行（在项目根目录下运行）
python scripts/data_overview.py
python scripts/clean_data.py
python scripts/eda_overview.py
python scripts/eda_category.py
python scripts/eda_sentiment.py
python scripts/bad_review_subcategory.py
python scripts/pareto_analysis.py
python scripts/final_report.py
```

## 图表索引（编号 1–15 连续）

| 图号 | 文件 | 内容 |
|------|------|------|
| 图 1 | `charts/01_review_length_dist.png` | 评论长度分布 |
| 图 2 | `charts/02_quadruple_count_dist.png` | 每条评论四元组数量分布 |
| 图 3 | `charts/03_category_distribution.png` | 属性分布 |
| 图 4 | `charts/04_category_negative_rate.png` | 各属性负面率 |
| 图 5 | `charts/05_implicit_vs_explicit_polarity.png` | 隐式 vs 显式极性 |
| 图 6 | `charts/06_category_cooccurrence_heatmap.png` | 属性共现热力图 |
| 图 7 | `charts/07_category_polarity_heatmap.png` | 属性×极性热力图 |
| 图 8 | `charts/08_implicit_aspect_ratio.png` | 各属性隐式占比 |
| 图 9 | `charts/09_opinion_word_length_dist.png` | 观点词长度分布 |
| 图 10 | `charts/10_positive_opinion_wordcloud.png` | 正面观点词云 |
| 图 11 | `charts/11_negative_opinion_wordcloud.png` | 负面观点词云 |
| 图 12 | `charts/12_top_category_opinions.png` | 头部属性痛点词 |
| 图 13 | `charts/13_bad_review_subcategory_stacked.png` | 差评子类构成（100% 堆叠） |
| 图 14 | `charts/14_pareto_problem_points.png` | 问题点维度帕累托图 |
| 图 15 | `charts/15_pareto_categories.png` | 属性维度帕累托图 |

## 数据说明

- 化妆品品类评论数据，已脱敏（品牌名以\*\*代替）；
- `id` 字段唯一标识评论原文与四元组标签，一条评论可对应多个四元组；
- AspectTerm / OpinionTerm 为空时用 `_` 表示（隐式表达）；
- Category 取值集合：{包装，成分，尺寸，服务，功效，价格，气味，使用体验，物流，新鲜度，真伪，整体，其他}；
- Polarity 取值集合：{正面，中性，负面}。

详见 `data/数据说明.md`，完整分析结论见 `reports/final_report.md`。
