# 🏦 TASK 2: Bank Marketing – Classification Model & Feature Pipeline

## 📌 Overview
This project builds a classification model that predicts whether a bank customer will accept a **term deposit** offer. It covers the full machine learning workflow: data cleaning, categorical encoding, numerical scaling, train-test split, model training, and evaluation using accuracy, precision, recall, and F1-score.

## 🎯 Objectives
- Clean the dataset and handle missing or `unknown` values
- Encode categorical features into numerical values
- Normalize numerical features
- Split the data into training and testing sets
- Train a classification model to predict the customer's response
- Evaluate the model using accuracy, precision, recall, and F1-score

## 📁 Dataset
The Bank Marketing dataset contains **41,188 records and 21 columns**, covering customer, financial, contact, campaign, and economic information.

**Target variable:** `y`

| Value | Meaning |
|---|---|
| `yes` | Customer accepted the term deposit offer |
| `no` | Customer did not accept the offer |

## 🧾 Features

| Group | Features |
|---|---|
| **Customer** | `age`, `job`, `marital`, `education` |
| **Financial** | `default`, `housing`, `loan` |
| **Contact & Campaign** | `contact`, `month`, `day_of_week`, `campaign`, `pdays`, `previous`, `poutcome` |
| **Economic** | `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed` |

### ⚠️ Duration removed (data leakage)
`duration` is the length of the last call in seconds. It is only known **after** the call happens, so it was removed to keep the model a realistic **pre-call predictor** and to avoid data leakage.

After removing `y` and `duration`, **19 features** remain.

**Numerical (9):** `age`, `campaign`, `pdays`, `previous`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`

**Categorical (10):** `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `poutcome`

## 🧹 Data Cleaning
The dataset has no actual `NaN` values, but several categorical columns contain the value `unknown`:

| Column | Unknown Values |
|---|---:|
| `job` | 330 |
| `marital` | 80 |
| `education` | 1,731 |
| `default` | 8,597 |
| `housing` | 990 |
| `loan` | 990 |

`default` has the most unknown values. In this pipeline, `unknown` is kept as its own category and handled by the one-hot encoder.

## ⚙️ Preprocessing Pipeline

**Numerical pipeline**
```text
Median Imputation → StandardScaler
```

**Categorical pipeline**
```text
Most Frequent Imputation → One-Hot Encoding
```

Both pipelines are combined with `ColumnTransformer`.

## ✂️ Train-Test Split
- **80%** training data, **20%** testing data
- `stratify=y` keeps the `yes`/`no` proportion similar in both sets

## 🤖 Model
A **Logistic Regression** classifier is combined with the preprocessing steps in a single `Pipeline`, so the same transformations are applied during training and prediction.

```python
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])
```

## 📊 Model Performance

**Accuracy: 90.09%**

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `no` | 0.91 | 0.99 | 0.95 | 7,310 |
| `yes` | 0.69 | 0.22 | 0.33 | 928 |

**Confusion Matrix**

|  | Predicted `no` | Predicted `yes` |
|---|---:|---:|
| **Actual `no`** | 7,219 | 91 |
| **Actual `yes`** | 725 | 203 |

- **7,219** customers correctly predicted as `no`
- **203** customers correctly predicted as `yes`
- **91** customers predicted `yes` but actually `no`
- **725** customers predicted `no` but actually `yes`

## 🔍 Key Insights
- The dataset has both numerical and categorical features and no `NaN` values, but several columns contain `unknown`.
- The dataset is **imbalanced**: far more `no` responses than `yes`.
- Logistic Regression reaches about **90% accuracy**, but accuracy is misleading here because of the imbalance.
- Recall for `no` is **99%**, while recall for `yes` is only **22%**.
- The model missed **725 customers who would have said `yes`**.
- The main limitation is the model's weak ability to identify likely subscribers.

## 🚀 Possible Improvements
- Use `class_weight="balanced"` in Logistic Regression
- Apply oversampling such as **SMOTE**
- Compare Random Forest and Gradient Boosting
- Tune hyperparameters
- Add ROC-AUC and PR-AUC as evaluation metrics

## 🔄 Project Workflow
```text
Load Dataset
     ↓
Explore Dataset
     ↓
Check Missing / Unknown Values
     ↓
Define Features and Target
     ↓
Remove Duration (avoid data leakage)
     ↓
Separate Numerical and Categorical Features
     ↓
Train-Test Split
     ↓
Build Preprocessing Pipeline
     ↓
Train Logistic Regression
     ↓
Predict and Evaluate
     ↓
Analyze Results
```

## 🛠️ Technologies Used
Python • Pandas • NumPy • Scikit-learn (Logistic Regression, StandardScaler, OneHotEncoder, SimpleImputer, Pipeline, ColumnTransformer)

## ▶️ How to Run
```bash
pip install pandas numpy scikit-learn
```
Place the dataset CSV in the project folder, open the notebook in Jupyter or Google Colab, and run all cells.

## 📝 Conclusion
This project demonstrates a complete machine learning workflow, from data cleaning and feature preparation to model training and evaluation. The Logistic Regression model achieved **90.09% accuracy**, but its **22% recall for the `yes` class** shows it still needs improvement before it can reliably identify customers likely to accept a term deposit.

## 👤 Author
**Prathana Kamlesh Tandel**
