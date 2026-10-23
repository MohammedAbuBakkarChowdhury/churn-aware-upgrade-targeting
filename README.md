# StreamPlus Premium-Tier Upgrade Targeting

### Predictive Modeling for Revenue Growth and Churn-Aware Campaign Targeting

An end-to-end machine learning and business analytics project that uses subscriber behavior to identify which Basic-tier customers should receive a Premium upgrade offer.

The project goes beyond simply predicting who is likely to upgrade. It converts model probabilities into an actionable targeting policy by explicitly balancing the financial value of successful upgrades against the estimated lifetime-value cost of poorly targeted offers.

---

## Project Overview

StreamPlus is a fictional direct-to-consumer streaming service with approximately **4.9 million paying subscribers in North America**.

The company offers two subscription tiers:

- **Basic:** $9/month, ad-supported, one-device access
- **Premium:** $17/month, ad-free, up to four simultaneous devices, and early access to original content

The subscription-growth team wants to increase **Average Revenue per User (ARPU)** by converting Basic subscribers to Premium.

At first glance, the obvious strategy would be to send upgrade offers to as many Basic subscribers as possible.

However, the case introduces an important business constraint:

> Sending an irrelevant Premium offer can increase subscriber churn risk.

Historical untargeted campaigns convert only about **11%** of recipients. The case also assumes that an unsuccessful, poorly targeted offer creates approximately **$55 in expected customer lifetime-value loss**.

This transforms the problem from a standard marketing-classification task into a business optimization problem.

The objective is not simply:

> **Who is most likely to upgrade?**

The actual business question is:

> **Which subscribers should StreamPlus contact in order to maximize expected business value while controlling churn-related risk?**

---

# Business Problem

Every targeting decision has a different financial consequence.

| Outcome | Business Meaning | Financial Impact |
|---|---|---:|
| True Positive | Subscriber receives the offer and upgrades | **+$96** annual incremental revenue |
| False Positive | Subscriber receives the offer but does not upgrade | **-$55** expected LTV impact |
| False Negative | Potential upgrader is not targeted | Missed revenue opportunity |
| True Negative | Subscriber is correctly left untargeted | No incremental value or targeting cost |

A successful Basic-to-Premium conversion creates approximately:

```text
($17 - $9) × 12 months = $96
```

of incremental annual subscription revenue.

The project therefore evaluates targeting performance using:

```text
Net Value = (True Positives × $96) - (False Positives × $55)
```

This is the central business metric used throughout the analysis.

---

## Why Traditional Model Accuracy Is Not Enough

A classification model can perform well statistically while still producing a poor business decision.

For example, a highly conservative model may:

- produce very few false positives,
- achieve strong precision,
- but miss a large number of legitimate upgrade opportunities.

Alternatively, an aggressive model may:

- capture most potential upgraders,
- achieve strong recall,
- but generate so many false positives that churn-related losses outweigh the incremental revenue.

For that reason, the project does **not** automatically optimize for:

- Accuracy
- Precision
- Recall
- F1 Score
- The standard 0.50 classification threshold

These metrics remain important diagnostics, but the final campaign rule is selected according to **expected economic value**.

---

# Business Questions

The analysis is designed to answer five main questions:

1. **Which subscriber behaviors are associated with Premium upgrade acceptance?**
2. **How effectively can a logistic regression model distinguish likely upgraders from non-upgraders?**
3. **How does the default 0.50 classification threshold perform?**
4. **What probability threshold maximizes expected Net LTV impact?**
5. **How should StreamPlus operationalize the resulting model in a future upgrade campaign?**

---

# Dataset

The project analyzes **2,500 historical subscriber records** containing behavioral and engagement information.

## Dataset Summary

| Measure | Result |
|---|---:|
| Subscriber records | 2,500 |
| Columns | 8 |
| Missing values | 0 |
| Accepted upgrades | 777 |
| Declined upgrades | 1,723 |
| Upgrade acceptance rate | 31.08% |
| Train/Test Split | 70% / 30% |
| Training records | 1,750 |
| Holdout records | 750 |

The target variable is moderately imbalanced:

```text
Accepted Upgrade: 31.08%
Declined Upgrade: 68.92%
```

This imbalance is one reason accuracy should not be used as the only measure of model quality.

---

# Variables

| Variable | Description |
|---|---|
| `subscriber_id` | Unique subscriber identifier |
| `months_subscribed` | Length of subscriber tenure |
| `content_hours_per_week` | Average weekly content consumption |
| `devices_used` | Number of devices used by the subscriber |
| `recent_search_rate` | Measure of recent content-search activity |
| `past_offer_responses` | Number of previous positive promotional responses |
| `age_range_code` | Encoded customer age group |
| `accepted_upgrade` | Target variable: 1 = accepted upgrade, 0 = declined |

`subscriber_id` is removed before modeling because it is an identifier rather than a meaningful behavioral predictor.

---

# Tools & Technologies

| Tool / Library | Purpose |
|---|---|
| Python | Core analytical workflow |
| Pandas | Data inspection and manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| scikit-learn | Machine learning and evaluation |
| Jupyter Notebook | Reproducible analysis |
| GitHub | Version control and portfolio presentation |

---

# Project Workflow

```text
Business Problem
       ↓
Data Quality Review
       ↓
Exploratory Data Analysis
       ↓
Feature Selection
       ↓
Stratified Train/Test Split
       ↓
Logistic Regression
       ↓
Probability Estimation
       ↓
Baseline Model Evaluation
       ↓
Threshold Simulation
       ↓
Business-Value Optimization
       ↓
Recommended Targeting Rule
       ↓
Historical Simulation
       ↓
Recommendations & Limitations
```

---

# Data Preparation

The dataset required relatively little cleaning.

The initial data-quality review found:

```text
Rows:              2,500
Missing Values:    0
Duplicate Review:  Performed
Target Variable:   accepted_upgrade
```

No missing-value imputation was required.

The subscriber identifier was removed from the predictive feature set.

The resulting model uses six behavioral variables:

```text
months_subscribed
content_hours_per_week
devices_used
recent_search_rate
past_offer_responses
age_range_code
```

---

# Exploratory Data Analysis

## Class Balance

Of the 2,500 subscribers:

```text
1,723 declined the Premium upgrade
777 accepted the Premium upgrade
```

This produces an upgrade acceptance rate of:

```text
31.08%
```

The positive class is therefore smaller than the negative class, although the imbalance is not extreme.


# Subscriber Behavior Findings

The exploratory analysis reveals meaningful differences between subscribers who accepted the Premium offer and those who declined it.

## Average Behavioral Profile

| Feature | Accepted Upgrade | Declined Upgrade |
|---|---:|---:|
| Months subscribed | 19.47 | 13.01 |
| Content hours per week | 8.51 | 5.93 |
| Devices used | 2.39 | 1.92 |

Subscribers who accepted Premium upgrades therefore tended to:

- have longer subscription tenure,
- watch more content per week,
- and use more devices.

These findings support the business hypothesis that more engaged and established subscribers are generally more receptive to Premium-tier offers.

However, these results should be interpreted as **associations**, not causal effects.

The project does not prove that increasing someone's viewing hours or number of devices would cause them to upgrade.

### Suggested Visual

```markdown
![Subscriber Behavior Profile](assets/behavior_profile.png)
```

---

# Model Development

## Why Logistic Regression?

A **logistic regression classifier** was selected as the core predictive model.

This model is particularly suitable for the project because StreamPlus requires more than a binary prediction.

The business needs a probability estimate for each subscriber.

Logistic regression provides:

1. Probability of upgrade acceptance
2. Interpretable model coefficients
3. Odds-ratio analysis
4. Straightforward classification diagnostics
5. A probability score that can be converted into a targeting threshold

---

# Train/Test Split

The dataset was divided using a **stratified 70/30 split**.

```text
Training Set:  1,750 subscribers
Test Set:        750 subscribers
```

Stratification preserves approximately the same upgrade-acceptance rate in both samples.

A fixed random state is used for reproducibility:

```python
RANDOM_STATE = 42
```

The test set remains unseen during model training and is later used to evaluate performance and select the targeting threshold.

---

# Model Interpretation

Logistic regression coefficients were converted into **odds ratios**.

An odds ratio:

```text
> 1.0 → Higher upgrade odds
< 1.0 → Lower upgrade odds
```

## Model Results

| Feature | Coefficient | Odds Ratio | Relationship |
|---|---:|---:|---|
| `recent_search_rate` | 2.803 | 16.501 | Higher upgrade odds |
| `past_offer_responses` | 0.492 | 1.635 | Higher upgrade odds |
| `devices_used` | 0.170 | 1.185 | Higher upgrade odds |
| `content_hours_per_week` | 0.031 | 1.031 | Higher upgrade odds |
| `months_subscribed` | 0.007 | 1.007 | Higher upgrade odds |
| `age_range_code` | -0.174 | 0.840 | Lower upgrade odds |

---

# Feature Interpretation

## Recent Search Activity

`recent_search_rate` shows a strong positive association with upgrade probability.

Subscribers actively searching for content may:

- interact more frequently with the platform,
- consume more content,
- and potentially perceive greater value from Premium features.

---

## Past Offer Responses

Subscribers who previously responded positively to promotional offers also demonstrate higher Premium-upgrade odds.

Historical promotional responsiveness therefore appears useful as a targeting signal.

---

## Devices Used

Subscribers using more devices demonstrate greater upgrade propensity.

This result is intuitively consistent with the Premium tier because Premium supports more simultaneous device usage.

---

## Content Consumption

Subscribers consuming more hours of content per week are modestly more likely to upgrade.

This suggests that platform engagement is associated with greater willingness to pay for enhanced service.

---

## Subscription Tenure

Longer-tenured subscribers demonstrate slightly higher upgrade odds.

This is consistent with the descriptive EDA, where accepted subscribers averaged:

```text
19.47 months subscribed
```

compared with:

```text
13.01 months
```

for non-upgraders.

---

## Age Range

Higher values of `age_range_code` are associated with somewhat lower upgrade odds.

This suggests younger subscriber groups may demonstrate slightly greater Premium-upgrade propensity.

---

## Important Interpretation Caveat

The features use different measurement scales.

Therefore, the raw odds ratios should **not** be interpreted as a direct feature-importance ranking.

For example:

```text
1 unit of recent_search_rate
```

is not comparable to:

```text
1 month of subscriber tenure
```

The coefficient values therefore describe feature-specific relationships rather than a standardized ranking.

### Suggested Visual

```markdown
![Logistic Regression Odds Ratios](assets/odds_ratios.png)
```

---

# Baseline Model Evaluation

The first evaluation uses the standard probability threshold:

```text
0.50
```

A subscriber is predicted to upgrade when:

```text
Predicted Probability >= 0.50
```

## Baseline Performance

| Metric | Result |
|---|---:|
| Accuracy | 72.00% |
| Precision | 62.92% |
| Recall | 24.03% |
| F1 Score | 34.78% |
| True Negatives | 484 |
| False Positives | 33 |
| False Negatives | 177 |
| True Positives | 56 |
| Estimated Net Value | $3,561 |

---

# Confusion Matrix

The holdout confusion matrix at the default threshold is:

```text
                         Predicted
                   No Upgrade   Upgrade

Actual No Upgrade      484         33
Actual Upgrade         177         56
```

### Interpretation

### True Negatives: 484

The model correctly identifies subscribers who would not accept the upgrade.

### False Positives: 33

These subscribers would receive the offer despite ultimately not upgrading.

Under the business assumptions, each false positive creates:

```text
-$55 expected LTV impact
```

### False Negatives: 177

These subscribers actually accepted the upgrade but would not have been targeted by the model.

This represents a substantial missed-revenue opportunity.

### True Positives: 56

These subscribers are correctly identified as likely upgraders.

Each successful upgrade generates approximately:

```text
+$96 annual incremental revenue
```

### Suggested Visual

```markdown
![Confusion Matrix](assets/confusion_matrix.png)
```

---

# Why the Default 0.50 Threshold Is Too Conservative

The model achieves respectable precision at 0.50:

```text
62.92%
```

However, recall is only:

```text
24.03%
```

The model therefore identifies less than one-quarter of the subscribers who actually upgrade.

The standard 0.50 threshold is mathematically convenient but has no inherent connection to StreamPlus's campaign economics.

Because:

```text
True Positive Value  = +$96
False Positive Cost  = -$55
```

StreamPlus may benefit from accepting additional false positives if doing so captures enough additional true upgraders.

---

# Threshold Optimization

The model generates probabilities rather than only class labels.

Those probabilities make it possible to simulate different targeting strategies.

Thresholds were tested across the probability range.

For every threshold, the analysis recalculated:

- True Positives
- False Positives
- True Negatives
- False Negatives
- Precision
- Recall
- F1 Score
- Net Value

The key objective is:

```text
Net Value = (TP × $96) - (FP × $55)
```

---

# Threshold Comparison

| Threshold | TP | FP | FN | Precision | Recall | Net Value |
|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 151 | 176 | 82 | 46.18% | 64.81% | $4,816 |
| **0.31** | **147** | **163** | **86** | **47.42%** | **63.09%** | **$5,147** |
| 0.32 | 138 | 150 | 95 | 47.92% | 59.23% | $4,998 |
| 0.39 | 97 | 84 | 136 | 53.59% | 41.63% | $4,692 |
| 0.45 | 75 | 48 | 158 | 60.98% | 32.19% | $4,560 |
| 0.50 | 56 | 33 | 177 | 62.92% | 24.03% | $3,561 |

The strongest holdout threshold is:

# **0.31**

---

# Optimal Threshold Results

At the 0.31 probability threshold:

```text
Subscribers evaluated:       750
Subscribers contacted:       310
True positives:              147
False positives:             163
False negatives:              86
Precision / conversion:    47.42%
Recall:                    63.09%
Estimated Net LTV Impact: $5,147
```

This threshold produces the highest estimated holdout Net Value among the tested decision rules.

### Suggested Visual

```markdown
![Net Value by Probability Threshold](assets/threshold_net_value.png)
```

---

# Precision vs. Business Value

One of the most important findings from this project is that:

> **Higher precision does not necessarily mean higher business value.**

Compare the two thresholds.

## Default Threshold — 0.50

```text
Precision:    62.92%
Recall:       24.03%
Net Value:    $3,561
```

## Optimized Threshold — 0.31

```text
Precision:    47.42%
Recall:       63.09%
Net Value:    $5,147
```

Precision decreases because the campaign becomes less selective.

However, the lower threshold captures many additional true upgraders.

The incremental revenue generated from those additional true positives exceeds the additional churn-related cost created by false positives.

The resulting Net Value therefore increases substantially.

---

# Improvement Relative to the Default Threshold

Moving from 0.50 to 0.31 increases estimated Net Value from:

```text
$3,561
```

to:

```text
$5,147
```

That is approximately a:

# **44.5% increase in estimated Net Value**

relative to using the default classification threshold.

This is the core analytical result of the project.

### Suggested Visual

```markdown
![Precision and Recall by Threshold](assets/precision_recall.png)
```

---

# Recommended Campaign Strategy

The recommended operational policy is:

> **Send a Premium upgrade offer only when a Basic subscriber's predicted probability of accepting is at least 31%.**

The threshold is chosen using the **held-out test set**.

This distinction matters.

The test set contains subscribers the model did not see while training.

The resulting threshold therefore provides a more defensible estimate of how the strategy may perform on future subscribers.

---

# Holdout Results at the Recommended Threshold

| Measure | Result |
|---|---:|
| Subscribers evaluated | 750 |
| Subscribers contacted | 310 |
| Conversion / Precision | 47.42% |
| Recall | 63.09% |
| Estimated Net LTV Impact | **$5,147** |

Historical untargeted campaigns converted approximately:

```text
11%
```

while the modeled targeting group achieves approximately:

```text
47.42%
```

within the holdout data.

This difference should **not** be interpreted as proof that the model causes a fourfold conversion increase.

The analysis is observational.

A randomized experiment would be required to establish causal incremental lift.

---

# Full Historical Simulation

After selecting 0.31 using the holdout dataset, the same policy was applied retrospectively across all:

```text
2,500 subscribers
```

## Full-History Results

| Measure | Result |
|---|---:|
| Subscribers evaluated | 2,500 |
| Subscribers contacted | 1,060 |
| Share contacted | 42.4% |
| Conversion / Precision | 46.23% |
| Recall | 63.06% |
| Estimated Net LTV Impact | **$15,690** |

The simulation suggests that approximately:

```text
42.4%
```

of the historical subscriber base would have received the offer.

---

# Holdout vs. Historical Simulation

| Measure | Holdout @ 0.31 | Full Historical @ 0.31 |
|---|---:|---:|
| Subscribers evaluated | 750 | 2,500 |
| Subscribers contacted | 310 | 1,060 |
| Conversion / Precision | 47.42% | 46.23% |
| Recall | 63.09% | 63.06% |
| Estimated Net LTV Impact | $5,147 | $15,690 |
| Interpretation | Out-of-sample evidence | In-sample sizing simulation |

The distinction between these two results is critical.

The full historical dataset includes observations used during model training.

Therefore:

> **$15,690 should not be treated as an unbiased forecast of future campaign performance.**

It is primarily useful for illustrating campaign scale.

The holdout result remains the stronger estimate of model generalization.

---

# Overfitting and Threshold Selection

When the full historical dataset is evaluated directly, a threshold near:

```text
0.39
```

produces approximately:

```text
$16,333
```

of simulated Net Value.

That value is higher than the $15,690 produced by 0.31.

However, selecting 0.39 on this basis would be methodologically weak.

Because most of those observations were used to train the model, optimizing the threshold against the full historical dataset would introduce **optimistic bias**.

The 0.31 threshold remains preferable because:

- it was chosen using unseen holdout observations,
- it provides stronger out-of-sample evidence,
- and it reduces the risk of overfitting the marketing policy to historical data.

---

# Key Findings

## 1. Engagement Is Associated with Upgrade Propensity

Subscribers who upgrade tend to demonstrate:

- longer tenure,
- greater weekly viewing activity,
- more device usage,
- higher search activity,
- and stronger historical responsiveness to offers.

---

## 2. Recent Search Behavior Is an Important Signal

`recent_search_rate` has a strong positive relationship with upgrade probability in the logistic regression model.

---

## 3. Previous Offer Responses Matter

Subscribers who responded positively to prior offers also show greater Premium-upgrade likelihood.

---

## 4. Device Usage Supports Premium Propensity

Multi-device users show higher upgrade odds, which is consistent with the additional device flexibility offered by Premium.

---

## 5. The Default 0.50 Threshold Misses Too Many Upgraders

At 0.50:

```text
False Negatives = 177
Recall = 24.03%
```

The strategy is too conservative for the stated economics.

---

## 6. Lower Precision Can Produce More Value

The 0.31 threshold produces lower precision than 0.50 but dramatically better recall.

The additional successful upgrades generate enough value to offset the greater number of false positives.

---

## 7. Business Metrics Should Drive Deployment Decisions

Classification metrics alone do not answer the real business question.

The deployment rule should reflect:

```text
Revenue Opportunity
+
Churn Risk
+
Customer Lifetime Value
=
Economic Decision
```

---

# Business Recommendations

## 1. Replace Broad Campaigns with Probability-Based Targeting

StreamPlus should move away from broad untargeted upgrade campaigns.

Basic subscribers should first receive a model-generated upgrade probability.

Under the current economics, only subscribers with:

```text
Predicted Probability >= 0.31
```

should receive the Premium offer.

---

## 2. Use Net LTV Impact as the Primary Campaign KPI

Model performance should ultimately be judged by:

```text
Incremental Upgrade Revenue
-
Churn-Related LTV Loss
=
Net Campaign Value
```

Precision, recall, F1, and accuracy should remain supporting metrics.

---

## 3. Monitor False Positives Carefully

The model assumes that poorly targeted upgrade offers create meaningful churn-related damage.

Production monitoring should therefore track:

- targeted subscribers,
- accepted offers,
- declined offers,
- post-offer cancellations,
- customer lifetime value,
- realized campaign revenue,
- and realized campaign profitability.

---

## 4. Conduct an A/B Test Before Full Deployment

The model identifies subscribers who are likely to upgrade.

It does not establish whether receiving the offer **causes** the upgrade.

A production experiment should randomly compare:

### Treatment

Subscribers selected by the model receive the Premium offer.

### Control

Comparable eligible subscribers do not receive the targeted offer or continue receiving the existing strategy.

This would allow StreamPlus to estimate true incremental campaign lift.

---

## 5. Recalculate the Threshold When Economics Change

The 0.31 threshold depends on:

```text
Upgrade Revenue
False-Positive Cost
Customer Lifetime Value
Churn Risk
Subscriber Behavior
```

If any of these assumptions change, the threshold should be recomputed.

---

## 6. Monitor Probability Calibration

Because the targeting system uses predicted probabilities directly, the scores must remain meaningful.

A subscriber receiving:

```text
0.31
```

should continue to represent roughly the same level of upgrade propensity over time.

Calibration should therefore be monitored periodically.

---

## 7. Monitor Model Drift

Subscriber behavior can change because of:

- pricing changes,
- new content,
- competitor actions,
- product redesign,
- economic conditions,
- and changing customer demographics.

Model performance should be continuously reviewed.

---

# Limitations

## Fictional Business Scenario

StreamPlus is fictional.

The following are case assumptions:

```text
$96 annual incremental upgrade value
$55 false-positive LTV cost
11% historical untargeted conversion rate
Churn response assumptions
```

These should not be interpreted as independently validated industry benchmarks.

---

## Prediction Is Not Causation

The model answers:

> Who is likely to upgrade?

It does not answer:

> Who will upgrade specifically because we sent the offer?

Some high-propensity customers may have upgraded without receiving any campaign.

This is a fundamental difference between:

```text
Propensity Modeling
```

and:

```text
Causal / Uplift Modeling
```

---

## Limited Feature Set

The model uses only six behavioral predictors.

Potential future variables include:

- content genre preferences,
- content completion rates,
- payment history,
- failed payment behavior,
- account-sharing activity,
- customer-support interactions,
- device-level activity,
- geographic information,
- richer demographics,
- promotional history,
- and viewing recency.

---

## Class Imbalance

Accepted upgrades represent only:

```text
31.08%
```

of the dataset.

Although the imbalance is not severe, it demonstrates why accuracy alone is an incomplete performance measure.

---

## Single Core Model

The project focuses on logistic regression.

This provides strong interpretability but may not capture nonlinear relationships.

Alternative models could include:

- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

Any alternative should still be evaluated using **out-of-sample business value** rather than accuracy alone.

---

# Future Work

Potential project extensions include:

### Tree-Based Modeling

Compare logistic regression against nonlinear classifiers.

### Probability Calibration

Evaluate whether predicted probabilities accurately match observed upgrade frequencies.

### Uplift Modeling

Identify subscribers who are likely to upgrade specifically **because of** an offer.

### Causal Inference

Estimate incremental treatment effects rather than simple correlations.

### A/B Testing

Validate the targeting rule through randomized experimentation.

### Real-Time Scoring

Deploy the model within a subscriber marketing platform.

### Drift Monitoring

Track changes in model performance over time.

### Automated Threshold Optimization

Recalculate the optimal targeting threshold when pricing or campaign economics change.

---

# Repository Structure

A recommended repository structure is:

```text
streamplus-premium-upgrade-targeting/
│
├── README.md
│
├── StreamPlus_Premium_Upgrade_Targeting_Portfolio.ipynb
│
├── StreamPlus_Project_Report_Expanded.pdf
│
├── assets/
│   ├── class_balance.png
│   ├── behavior_profile.png
│   ├── odds_ratios.png
│   ├── confusion_matrix.png
│   ├── threshold_net_value.png
│   └── precision_recall.png
│
└── presentation/
    └── Final_Project_Presentation.pdf
```

---

# How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/streamplus-premium-upgrade-targeting.git
```

Then:

```bash
cd streamplus-premium-upgrade-targeting
```

---

## 2. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

---

## 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
StreamPlus_Premium_Upgrade_Targeting_Portfolio.ipynb
```

Run the notebook from top to bottom.

---

# Reproducibility

The notebook uses a fixed random state:

```python
RANDOM_STATE = 42
```

and a stratified:

```text
70% Training
30% Testing
```

split.

This ensures that the core model results can be reproduced when using the same dataset and package versions.

---

# Project Deliverables

This repository includes:

- cleaned and documented Jupyter Notebook
- exploratory data analysis
- model-ready data tables
- descriptive statistics
- class-balance analysis
- logistic regression classifier
- coefficient analysis
- odds-ratio interpretation
- confusion-matrix analysis
- precision, recall, F1, and accuracy evaluation
- probability-threshold simulation
- Net LTV optimization
- holdout evaluation
- full-history scenario simulation
- business recommendations
- limitations and future-work analysis
- expanded written project report

---

# Skills Demonstrated

## Data Analysis

- Exploratory data analysis
- Descriptive statistics
- Behavioral comparison
- Data-quality review
- Class-balance analysis

## Machine Learning

- Binary classification
- Logistic regression
- Train/test splitting
- Stratification
- Probability prediction
- Confusion matrices
- Precision
- Recall
- F1 score
- Threshold optimization
- Odds ratios

## Business Analytics

- ARPU analysis
- Customer lifetime value
- Churn-risk evaluation
- Campaign targeting
- Conversion analysis
- Economic decision modeling
- Scenario simulation

## Decision Science

- Translating model probabilities into business rules
- Explicit cost-benefit optimization
- Separating statistical performance from business performance
- Out-of-sample validation
- Avoiding in-sample threshold optimization
- Experimental design recommendations

---

# Final Conclusion

The StreamPlus project demonstrates that building a predictive model is only part of the analytical problem.

At the standard 0.50 threshold, the logistic regression model achieves:

```text
Accuracy:   72.00%
Precision:  62.92%
Recall:     24.03%
Net Value:  $3,561
```

The model is precise but excessively conservative.

It identifies only:

```text
56
```

of the:

```text
233
```

actual upgraders in the holdout dataset.

By incorporating campaign economics directly into the decision rule, the analysis identifies:

# **0.31**

as the strongest holdout probability threshold.

At 0.31:

```text
Subscribers Contacted:       310 / 750
True Positives:              147
False Positives:             163
Precision / Conversion:    47.42%
Recall:                    63.09%
Estimated Net LTV Impact: $5,147
```

Compared with the default threshold, estimated holdout Net Value increases by approximately:

# **44.5%**

Applying the same decision rule retrospectively across the complete historical dataset produces:

```text
Subscribers Contacted:       1,060 / 2,500
Conversion Rate:             46.23%
Recall:                      63.06%
Historical Net LTV Impact:  $15,690
```

That full-history result is useful for sizing and scenario analysis, but it should not be interpreted as an unbiased forecast because the dataset includes records used to train the model.

The holdout result remains the stronger evidence of expected generalization.

The broader analytical lesson is:

> **The best machine-learning decision is not necessarily the one with the highest accuracy, precision, or recall. It is the decision that creates the strongest defensible business value.**

---

## Project Context

The original StreamPlus case was developed as an academic team project.

This repository contains my independently cleaned, refactored, documented, and portfolio-ready implementation of the analytical workflow.

---

## Author

**Mohammed Abu Bakkar Chowdhury**

Data Analytics | Machine Learning | Python | Business Analytics
