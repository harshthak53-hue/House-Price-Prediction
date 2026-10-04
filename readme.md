# House Price Prediction

**An End-to-End Machine Learning Pipeline for Predicting House Prices 🏡**

## 📌 Quick Project Overview

### Problem Statement

**California's** housing market varies significantly across different regions, with house prices influenced by factors such as location, income levels, population, and housing characteristics. Understanding these relationships can be difficult without a **data-driven approach**.

This project builds a **house price prediction model for California** to:

- Understand how geographical and housing-related factors relate to **house prices**.
- Estimate house prices based on historical housing data.
- Demonstrate an **end-to-end machine learning workflow**, from data preprocessing and model selection to prediction.

### Dataset

The project uses the **California Housing dataset**, a well-known dataset containing housing and demographic information for different areas of California.

Key features include:

- Median income of the area
- Housing-related room and household statistics
- Population
- Geographical location (latitude and longitude)
- Ocean proximity
- Median house value — **the target variable**

The dataset used by the project is loaded from `housing.csv` and contains the target column `median_house_value`.

### Model

Several regression models were evaluated using **RMSE and Cross-Validation**, after which **Random Forest Regressor** was selected for the final model.

The model testing and selection process is shown in the [Model Selection](#model-selection) section.
The final implementation uses `RandomForestRegressor` to train on the prepared housing data.

### What the Project Produces

The model predicts **one continuous value: the median house value** based on multiple housing and geographical features.

Therefore, this is a **supervised learning regression problem**:

- **Input:** Multiple housing and demographic features
- **Target:** `median_house_value`
- **Output:** A predicted house value

During inference, the trained model processes new housing data and produces a predicted `median_house_value` for each record.

---

## 📋 Table of Contents

- [⚙️ How It Works](#how-it-works)
- [📁 Project Structure](#project-structure)
- [🔍 Technical Implementation (Deep Dive)](#technical-implementation)
- [📚 What I Learned](#what-i-learned)
- [👤 Author](#author)


---
<a id="how-it-works"></a>

## ⚙️ How It Works

The two workflows are controlled by this condition:

> **If `model.pkl` doesn't exist → train the model.**  
> **If it already exists → load it and perform inference.**

### Initial Run — Training Workflow

When the project is run for the first time, `main.py` follows this flow:

1. **Load the housing data** from `housing.csv`.

2. **Create a stratified 80/20 train-test split** using `StratifiedShuffleSplit`. Before splitting, the `median_income` feature is grouped into income categories so the distribution is maintained across both sets. The 20% test data is saved as `test.csv`, while the remaining 80% becomes the training data.

3. **Separate features and target** from the training data. `median_house_value` becomes the target, while the remaining columns are used as input features.

4. **Separate numerical and categorical features.** All columns except `ocean_proximity` are treated as numerical, while `ocean_proximity` is treated as categorical.

5. **Preprocess the features through dedicated pipelines.**

   - Numerical data → median imputation → standard scaling
   - Categorical data → one-hot encoding

   Both are combined using a `ColumnTransformer`.

6. **Transform the training data** using the preprocessing pipeline.

7. **Train the Random Forest Regressor** on the transformed training data.

8. **Save the trained model and preprocessing pipeline** using Joblib as `model.pkl` and `pipeline.pkl` inside the `models` folder.

9. **Print a confirmation message** indicating that training has completed.

### Inference Workflow

Once the model artifacts already exist, `main.py` switches to inference mode:

1. **Load the saved model and preprocessing pipeline** from the `models` folder using Joblib.

2. **Load the saved 20% test dataset** from `test.csv`. This data was held out during training and therefore was not used to fit the model.

3. **Transform the test data** using the already-fitted preprocessing pipeline.

4. **Generate predictions** using the trained Random Forest model.

5. **Add the predictions** to the input data under the `median_house_value` column and save the resulting dataset as `output.csv` in the `predictions` folder.

6. **Print a confirmation message** when inference is complete.

---
<a id="project-structure"></a>

## 📁 Project Structure

```text
House-Price-Prediction/
│
├── data/
│   ├── housing.csv
│   └── test.csv
│
├── models/
│   ├── model.pkl
│   └── pipeline.pkl
│
├── predictions/
│   └── output.csv
│
├── src/
│   ├── main.py
│   └── testing.ipynb
│
└── README.md
```

| Path | Purpose |
|---|---|
| `data/housing.csv` | Original California housing dataset |
| `data/test.csv` | Held-out test data created during the initial split |
| `models/model.pkl` | Serialized trained Random Forest model |
| `models/pipeline.pkl` | Serialized preprocessing pipeline |
| `predictions/output.csv` | Predictions generated during inference |
| `src/main.py` | Main script handling training and inference |
| `src/testing.ipynb` | Notebook used for workflow testing and model selection based on RMSE |
| `README.md` | Project documentation |

<a id="technical-implementation"></a>

## 🔍 Technical Implementation (Deep Dive)

This section explains the key technical decisions behind the project, from model selection and data splitting to preprocessing and model persistence.

<a id="model-selection"></a>
### 1. Model Selection

Before finalizing the model, multiple regression algorithms were evaluated using **10-fold Cross-Validation** with **Root Mean Squared Error (RMSE)** as the evaluation metric.

The tested models included:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Random Forest achieved the lowest RMSE among the tested models and was therefore selected for the final implementation.

The complete model testing and comparison can be found in [Model Testing](src/testing.ipynb).

The final implementation uses `RandomForestRegressor` with `random_state=42`.

### 2. Stratified Train-Test Split

The dataset is divided into **80% training data and 20% test data** using `StratifiedShuffleSplit`.

To perform stratification, the continuous `median_income` feature is temporarily divided into five income categories using the `income_cat` column.

```python
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5]
)
```

These categories are used only to create a more representative train-test split. The temporary `income_cat` column is removed after the split.

The split uses:

- **80%** → Training data
- **20%** → Test data
- `random_state=42` → Reproducible split

### 3. Feature and Target Separation

After the split, `median_house_value` is separated as the target variable, while all remaining columns are used as input features.

The input features are then divided into:

- **Numerical features** — all features except `ocean_proximity`
- **Categorical feature** — `ocean_proximity`

This allows each type of feature to go through the appropriate preprocessing steps.

### 4. Numerical Data Preprocessing

Numerical features pass through a two-step pipeline:

```text
Numerical Features
       ↓
SimpleImputer(strategy="median")
       ↓
StandardScaler
```

#### `SimpleImputer(strategy="median")`

- Missing numerical values are replaced with the median value of their respective feature.
- Using the median provides a simple way to handle missing values while being less sensitive to extreme values than the mean.

#### `StandardScaler`

- The numerical features are then standardized so that they are placed on a comparable scale.
- Both operations are kept together inside a `Pipeline`, ensuring that the same preprocessing sequence can be reused during inference.

### 5. Categorical Data Preprocessing

The categorical feature, `ocean_proximity`, is processed using `OneHotEncoder`.

```text
Categorical Features
       ↓
OneHotEncoder
```

One-hot encoding converts categorical values into numerical indicator features that can be passed to the regression model.

The encoder also uses:

```python
OneHotEncoder(handle_unknown="ignore")
```

This prevents the preprocessing step from failing if an unseen category appears during inference.

### 6. Combining the Preprocessing Pipelines

The numerical and categorical pipelines are combined using `ColumnTransformer`.

```text
                   ┌── Numerical Pipeline ──→ Impute → Scale ──┐
Input Features ────┤                                             ├──→ Transformed Data
                   └── Categorical Pipeline → One-Hot Encode ──┘
```

This allows different transformations to be applied to different columns while producing a single transformed dataset for the model.

The complete preprocessing structure is created inside the `build_pipeline()` function.

### 7. Why Use Pipelines?

Using `Pipeline` and `ColumnTransformer` keeps preprocessing consistent between training and inference.

During training, the pipeline is fitted and used to transform the training features:

```python
housing_prepared = pipeline.fit_transform(housing_features)
```

During inference, the already-fitted pipeline is reused:

```python
transformed_input = pipeline.transform(input_data)
```

This ensures that the same preprocessing logic is applied at both stages.

### 8. Model and Pipeline Persistence

After training, both the trained model and preprocessing pipeline are saved using **Joblib**:

```text
models/
├── model.pkl
└── pipeline.pkl
```

- `model.pkl` → Serialized trained Random Forest model
- `pipeline.pkl` → Serialized preprocessing pipeline

Saving these artifacts allows the project to reuse the trained model and preprocessing logic without retraining every time.

During inference, the saved objects are loaded using `joblib.load()`.

The `.pkl` files contain serialized Python objects, while **Joblib** is the library responsible for serializing and loading them.

### 9. Training vs. Inference Decision

The project uses the existence of `model.pkl` to determine which workflow to execute:

```text
Does model.pkl exist?
       │
   ┌───┴───┐
  No       Yes
  ↓         ↓
Train     Inference
```

If `model.pkl` does not exist, the project trains the model and saves both the model and preprocessing pipeline.

If it already exists, the project loads the saved artifacts and proceeds directly to inference.

---
<a id="what-i-learned"></a>

## 📚 What I Learned
This project was my **first hands-on experience with Machine Learning**, and it gave me practical exposure to the complete process of building a regression model.

- Gained practical experience with **Scikit-learn** and concepts such as `StratifiedShuffleSplit`, `OneHotEncoder`, `Pipeline`, `ColumnTransformer`, and more.
- Learned how raw data is **prepared, transformed, and preprocessed** before it can be used for model training — probably where 80% of the work lives.
- Understood how a machine learning model is **trained using prepared data** and how it learns patterns from historical examples.
- Learned how to **evaluate and compare different models** using RMSE and Cross-Validation to select a suitable model.
- Learned how to organize preprocessing steps into **clean, reusable pipelines** instead of handling each transformation separately.
- Explored **Joblib** and **`.pkl` files** for the first time and learned how trained models and preprocessing pipelines can be saved and reused for inference.

> **Don't just build the model. Understand the journey from data to prediction.**

---
<a id="author"></a>

## 👤 Author

**Harshad Thakur**

Data Science / Machine Learning Portfolio Project

[LinkedIn](https://www.linkedin.com/in/harshad-thakur-94a124341/) · [GitHub](https://github.com/harshthak53-hue)
