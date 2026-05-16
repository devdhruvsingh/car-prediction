# Car Price Prediction Using Linear Regression

This project focuses on predicting car prices based on features such as vehicle model, year of manufacture, mileage, tax, fuel economy (mpg), engine size, transmission, and fuel type. The objective is to analyze the dataset, preprocess numerical and categorical variables, build a predictive Linear Regression model, and evaluate how different categorical encoding strategies impact model performance.

## 🚀 Project Overview
* **Goal**: Build an optimized regression model to accurately forecast car valuations.
* **Techniques Used**: Exploratory Data Analysis (EDA), Standard Scaling, Categorical Encoding (One-Hot vs. Label Encoding), Train-Test Splitting, and Multi-Variable Linear Regression.
* **Core Comparison**: Evaluating performance changes when representing categorical features via binary indicator columns (One-Hot Encoding) versus integer mapping (Label Encoding).

---

## 🛠️ Tech Stack & Libraries
* **Language**: Python 3.13
* **Data Manipulation**: `pandas`, `numpy`
* **Machine Learning**: `scikit-learn`
  * `LinearRegression` (Model building)
  * `train_test_split` (Validation strategy)
  * `StandardScaler`, `LabelEncoder` (Preprocessing)
  * `r2_score`, `mean_squared_error`, `mean_absolute_error` (Evaluation)

---

## 📊 Workflow & Methodology

### 1. Data Preprocessing & Feature Engineering
* **Numerical Scaling**: Features with wildly varying scales (`year`, `mileage`, `tax`, `mpg`, `engineSize`) were normalized using `StandardScaler` to bring them onto a uniform scale, ensuring the optimization solver treats them fairly.
* **Categorical Handling**: The features `model`, `transmission`, and `fuelType` were processed using two distinct methodologies to compare outcomes:
  * **Approach A (One-Hot Encoding)**: Transformed categories into binary sparse vectors (`0` or `1`) using `pd.get_dummies()`, dropping the first column to prevent the dummy variable trap (multicollinearity).
  * **Approach B (Label Encoding)**: Translated unique categories into sequential numerical integers using `LabelEncoder()`.

### 2. Validation Setup
For a robust validation strategy, data was partitioned into training ($67\%$) and testing ($33\%$) datasets using a fixed `random_state=42` to guarantee reproducibility across experiments.

---

## 📉 Models Evaluated

### Model 1: One-Hot Encoded Linear Regression
* **Input**: Standardized numerical features + One-Hot encoded categoricals.
* **Training Size**: 12,037 samples | **Testing Size**: 5,929 samples

### Model 2: Label Encoded Linear Regression
* **Input**: Standardized numerical features + Integer-mapped categorical labels.
* **Objective**: Evaluate how treating nominal categories as ordinal sequences impacts the geometric hyper-plane of a linear algorithm.

---

## 🏆 Key Metrics & Evaluation
The notebook evaluates performance utilizing three core metrics:
1. **Mean Absolute Error (MAE)**: Measures the average magnitude of absolute errors (robust to outliers).
2. **Mean Squared Error (MSE)**: Penalizes larger prediction deviations heavily.
3. **Adjusted $R^2$ Score**: Provides the proportion of variance explained by the model while explicitly penalizing the addition of irrelevant features that could lead to overfitting ($1 - (1 - r) \times \frac{n - 1}{n - p - 1}$).

### Key Insight: One-Hot vs. Label Encoding
> **Note**: Linear Regression inherently assumes numerical relationships reflect continuous ordering. In your analysis, **One-Hot Encoding** is structural for linear models when handling arbitrary categories (like car models), as **Label Encoding** falsely forces a mathematical ranking (e.g., Model 2 is twice as large as Model 1), distorting the final regression weights.

---

## 💻 How to Run the Project

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/yourusername/car-prediction.git](https://github.com/devdhruvsingh/car-prediction.git)
   cd car-prediction
