# Encoding Categorical Data Using ColumnTransformer

This repository contains my practical work on **encoding categorical data in Machine Learning** using Scikit-learn's `ColumnTransformer`.

Categorical features contain values such as names, types, groups, or categories rather than numerical measurements. Machine Learning algorithms generally require numerical input, so categorical values need to be converted into numerical representations before they can be used by many ML models.

In this practice, I worked with **One-Hot Encoding** and **Ordinal Encoding**, and learned how to apply them to different types of categorical features using `ColumnTransformer`.

---

# What is Categorical Data?

Categorical data represents groups or categories.

For example:

```text
City
Peshawar
Lahore
Karachi
Peshawar
```

Another example:

```text
Education
School
Bachelor
Master
PhD
```

These values are categorical and need appropriate encoding before being passed to many Machine Learning algorithms.

---

# 1. One-Hot Encoding

**One-Hot Encoding** converts each category into a separate binary column.

For example:

```text
Color
Red
Blue
Green
```

can become:

```text
Color_Blue    Color_Green    Color_Red
     0             0             1
     1             0             0
     0             1             0
```

Each category receives its own column.

One-Hot Encoding is generally appropriate for **nominal categorical data**, where the categories do not have a natural order.

### Scikit-learn Example

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(
    handle_unknown="ignore"
)

X_encoded = encoder.fit_transform(X)
```

The `handle_unknown="ignore"` option helps prevent errors when the test data contains a category that was not present in the training data.

---

# 2. Ordinal Encoding

**Ordinal Encoding** converts categories into numerical values while preserving their order.

For example:

```text
Education
School
Bachelor
Master
PhD
```

can be represented as:

```text
School      → 0
Bachelor    → 1
Master      → 2
PhD         → 3
```

Ordinal Encoding is useful when the categories have a meaningful order.

For example:

```text
Low < Medium < High
```

The order contains useful information, so ordinal encoding can represent that relationship.

### Scikit-learn Example

```python
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder()

X_encoded = encoder.fit_transform(X)
```

For ordered categories, the category order should be defined carefully rather than relying on an unintended ordering.

---

# 3. ColumnTransformer

`ColumnTransformer` allows different preprocessing techniques to be applied to different columns of the same dataset.

For example, a dataset might contain:

```text
Age          → Numerical
Salary       → Numerical
Gender       → Categorical
City         → Categorical
Education    → Ordinal
```

We may want to:

```text
Numerical columns
        ↓
StandardScaler

Nominal columns
        ↓
OneHotEncoder

Ordinal columns
        ↓
OrdinalEncoder
```

`ColumnTransformer` allows these transformations to be combined into one preprocessing workflow.

### Example

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_columns),
        ("nom", OneHotEncoder(handle_unknown="ignore"), nominal_columns),
        ("ord", OrdinalEncoder(), ordinal_columns)
    ]
)

X_transformed = preprocessor.fit_transform(X_train)
```

---

# 4. Why Use ColumnTransformer?

ColumnTransformer is useful because it allows us to apply different preprocessing methods to different groups of features.

For example:

```text
                Dataset
                   ↓
          ┌────────┴────────┐
          ↓                 ↓
     Numerical          Categorical
          ↓                 ↓
 StandardScaler       Encoding
          ↓                 ↓
          └────────┬────────┘
                   ↓
           Transformed Data
```

This makes preprocessing more organized and reduces the need to manually transform each group of columns.

---

# One-Hot vs Ordinal Encoding

| Technique | Suitable Data | Example |
|---|---|---|
| One-Hot Encoding | Nominal categories | City, Color, Gender |
| Ordinal Encoding | Ordered categories | Low, Medium, High |
| ColumnTransformer | Multiple feature types | Numerical + categorical |

---

# Important Machine Learning Practice

The encoder should be fitted on the **training data** and then used to transform the test data.

```python
preprocessor.fit(X_train)

X_train = preprocessor.transform(X_train)
X_test = preprocessor.transform(X_test)
```

This helps prevent information from the test dataset from influencing the preprocessing process.

---

# Combining with Pipeline

`ColumnTransformer` can also be combined with a Machine Learning Pipeline.

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression())
])

model.fit(X_train, y_train)
```

This allows preprocessing and model training to be handled as one complete workflow.

---

# Workflow

```text
Raw Dataset
     ↓
Identify Numerical & Categorical Columns
     ↓
Separate Nominal & Ordinal Features
     ↓
ColumnTransformer
     ↓
One-Hot / Ordinal / Numerical Transformation
     ↓
Transformed Data
     ↓
Machine Learning Model
     ↓
Prediction & Evaluation
```

# Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Jupyter Notebook

# Learning Goal

The goal of this repository is to develop a practical understanding of **categorical data encoding** and learn how `ColumnTransformer` can be used to apply different preprocessing techniques to different types of features.

Through this practice, I learned the difference between **One-Hot Encoding and Ordinal Encoding**, when each technique can be appropriate, and how to combine categorical and numerical preprocessing into a structured Machine Learning workflow.

This repository is part of my ongoing **Machine Learning and AI learning journey**.
