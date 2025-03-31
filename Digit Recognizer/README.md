# Digit Recognizer

## Overview
This project aims to recognize handwritten digits using various machine learning algorithms and techniques.

## Requirements
- Python 3
- Libraries: numpy, pandas, seaborn, matplotlib, scikit-learn

## Dataset
The dataset consists of handwritten digit images in grayscale format. It is commonly used for training various image processing systems. You can download the dataset from [Kaggle](https://www.kaggle.com/competitions/digit-recognizer/data).

## Usage
1. Clone this repository.
2. Ensure you have the required dependencies installed.
3. Run the notebook to perform exploratory data analysis and train machine learning models.
4. Analyze the results and model performance.

## Models Used
- Support Vector Classifier (SVC) (Best Score: 0.975)
- Logistic Regression (Score: 0.91617)
- Neural Network (Custom 4-layer, 3 Activation Functions) (Max Score: 0.89)

## Custom Approach
The custom approach involves building a model based on matrix multiplication to recognize digits. The highest achieved score using this approach is 0.82.

## Performance
- Best Score: 0.975 (SVC)
- Logistic Regression Score: 0.91617
- Custom Neural Network Score: 0.89
- Custom Approach Score: 0.82

| Scaling Method        | Best Suited For                                     | Description |
|----------------------|--------------------------------------------------|-------------|
| **StandardScaler**   | Logistic Regression, KNN, PCA, Gradient Descent  | Centers data to mean 0 and standard deviation 1. Works well for models sensitive to feature scales. |
| **MinMaxScaler**     | SVM, Neural Networks, K-Means                     | Scales features to a `[0,1]` (or `[-1,1]`) range. Suitable for models sensitive to feature ranges. |
| **RobustScaler**     | Tree-Based Models, SVM (with outliers)            | Uses the median and IQR to scale data, making it robust to outliers. |
| **MaxAbsScaler**     | Neural Networks, Data with positive & negative values | Scales features by their maximum absolute value, preserving signs and proportions. |
| **PowerTransformer** | Data with non-normal distribution                 | Applies Yeo-Johnson or Box-Cox transformations to make data more Gaussian-like. |
| **QuantileTransformer** | Models requiring uniform distribution         | Maps data to a uniform or normal distribution, distorting outliers but preserving relative ranks. |
| **Normalizer**       | KNN, Deep Learning, Cosine Similarity-based models | Normalizes each sample to have unit norm (`L2 norm = 1`), useful for distance-based models. |

**How to choose a scaling method?**  
- If **outliers are present** → `RobustScaler`  
- If **data is not normally distributed** → `PowerTransformer` / `QuantileTransformer`  
- If **features have large ranges** → `MinMaxScaler` / `MaxAbsScaler`  
- If **data is normally distributed** → `StandardScaler`  

⚡ **If unsure, start with `StandardScaler` or `MinMaxScaler` and compare results.**

# Hyperparameters of `LogisticRegression` and `SVC`

| Parameter          | LogisticRegression                                     | SVC                                  |
|--------------------|------------------------------------------------------|--------------------------------------|
| `penalty`         | `'l1'`, `'l2'`, `'elasticnet'`, `None`                | `'l1'`, `'l2'` (only for `linear` kernel) |
| `dual`            | `True`, `False` (only for `l2` with `liblinear`)      | Not applicable                      |
| `tol`             | Convergence tolerance (default: `1e-4`)               | Convergence tolerance (default: `1e-3`) |
| `C`               | Regularization strength (default: `1.0`)              | Regularization parameter (default: `1.0`) |
| `fit_intercept`   | `True`, `False` (whether to add intercept)            | Not applicable                      |
| `intercept_scaling` | Scaling for intercept (only for `liblinear`)         | Not applicable                      |
| `class_weight`    | `None`, `'balanced'`                                  | `None`, `'balanced'`                |
| `random_state`    | Controls randomness (default: `None`)                 | Controls randomness (default: `None`) |
| `solver`          | `'lbfgs'`, `'liblinear'`, `'saga'`, `'newton-cg'`, `'sag'` | `'linear'`, `'poly'`, `'rbf'`, `'sigmoid'` |
| `max_iter`        | Maximum iterations (default: `100`)                   | Maximum iterations (default: `-1`, no limit) |
| `multi_class`     | `'auto'`, `'ovr'`, `'multinomial'`                    | Not applicable                      |
| `verbose`         | `0` (silent) or higher for more output                | `0` (silent) or higher for more output |
| `warm_start`      | `True`, `False` (continue training from previous state) | Not applicable                      |
| `n_jobs`          | Number of CPU cores (default: `None`, single core)    | Not applicable                      |
| `l1_ratio`        | Only for `elasticnet` (range: `0` to `1`)             | Not applicable                      |
| `kernel`         | Not applicable                                         | `'linear'`, `'poly'`, `'rbf'`, `'sigmoid'` |
| `degree`         | Not applicable                                         | Degree of polynomial kernel (`poly` only) |
| `gamma`          | Not applicable                                         | Kernel coefficient (`scale`, `auto`, or float) |
| `coef0`         | Not applicable                                         | Independent term in `poly` and `sigmoid` kernels |
| `shrinking`      | Not applicable                                         | Whether to use shrinking heuristic (`True`, `False`) |
| `probability`    | Not applicable                                         | Whether to enable probability estimates (`True`, `False`) |
| `cache_size`     | Not applicable                                         | Memory size (in MB) for kernel cache (default: `200`) |
| `decision_function_shape` | Not applicable                                | `'ovr'` (one-vs-rest) or `'ovo'` (one-vs-one) |
| `break_ties`     | Not applicable                                         | Whether to break ties in `decision_function_shape='ovr'` |

```pyton
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    penalty="l2",            # Type of regularization: 'l1', 'l2' (default), 'elasticnet', or None
    dual=False,              # Whether to use the dual formulation (only for 'l2' and binary classification)
    tol=1e-4,                # Tolerance for stopping criteria (smaller = higher precision)
    C=1.0,                   # Regularization strength (smaller values = stronger regularization)
    fit_intercept=True,      # Whether to add an intercept term
    intercept_scaling=1,     # Scaling factor for the intercept (only relevant for solver='liblinear')
    class_weight=None,       # Class weighting: None or 'balanced'
    random_state=None,       # Seed for randomness (for reproducibility)
    solver="lbfgs",          # Optimization algorithm ('lbfgs', 'liblinear', 'saga', 'newton-cg', 'sag')
    max_iter=100,            # Maximum number of iterations
    multi_class="auto",      # 'auto' (default), 'ovr' (one-vs-rest), 'multinomial'
    verbose=0,               # Print training details (0 = no output)
    warm_start=False,        # If True, continues training from the previous model state
    n_jobs=None,             # Number of CPU cores for parallel processing (None = single core)
    l1_ratio=None            # Only relevant if penalty='elasticnet', controls balance between L1 and L2
)
```
```pyton
from sklearn.svm import SVC

model = SVC(
    C=1.0,                   # Regularization parameter (smaller value = stronger regularization)
    kernel="rbf",            # Kernel type: 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
    degree=3,                # Degree of the polynomial kernel (only used if kernel='poly')
    gamma="scale",           # Kernel coefficient for 'rbf', 'poly', and 'sigmoid': 'scale', 'auto', or numeric value
    coef0=0.0,               # Independent term in 'poly' and 'sigmoid' kernels
    shrinking=True,          # Use shrinking heuristic to speed up computation
    probability=False,       # Whether to enable probability estimates (slower when True)
    tol=1e-3,                # Tolerance for stopping criteria
    cache_size=200,          # Cache size (MB) for storing computation
    class_weight=None,       # Class weighting: None or 'balanced'
    verbose=False,           # Print training details
    max_iter=-1,             # Maximum number of iterations (-1 means unlimited)
    decision_function_shape="ovr", # 'ovr' (one-vs-rest) or 'ovo' (one-vs-one)
    break_ties=False,        # Whether to break ties when making multi-class decisions
    random_state=None        # Seed for randomness (relevant if probability=True)
)
```