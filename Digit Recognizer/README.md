# Digit Recognizer
```markdown
 ├── Dictionary save and load
 │   ├── dictionary_learning_model_64.pkl
 │   ├── load the model n.ipynb
 │   ├── model n Dictionary Learning.ipynb
 ├── Math for ML
 │   ├── SVD (& jpeg for SVD)
 │   ├── 🚀Linear Algebra Score 82%.ipynb
 │   ├── 🤖Neural network 4layer 3Activation Score 89%.ipynb
 ├── Sklearn standard model
 │   ├── 🔪SVC Score 95%.ipynb
 │   ├── 🧩Logistic Regression Score 91%.ipynb
```

## Overview
This project aims to recognize handwritten digits using various machine learning algorithms and techniques.
- [x] Math for ML: `Linear Algebra`, `NN`, `SVD`
- [x] Standard model: `Logistic Regression`, `SVM`
- [x] Save and load model
- [ ] Grid search techniques
- [ ] Ensemble models techniques

## Dataset
The dataset consists of handwritten digit images in grayscale format. It is commonly used for training various image processing systems. You can download the dataset from [Kaggle](https://www.kaggle.com/competitions/digit-recognizer/data).


## Performance Score
- **SVC**: 97% `Best Score`
- **Logistic Regression**: 92%
- **Custom Linear Algebra**: 82% 
- **Custom Neural network**: 89%

### ├── Dictionary save and load
- Practice saving and loading models.

### ├── Math for ML
- Model of Linear Algebra published in [kaggle My Code](https://www.kaggle.com/code/nisansher/digit-recognizer-only-linear-algebra-score-82)
- Neural network from scratch.
- SVD `sklearn.decomposition` dimension reduction VS linear regression, Image Compression, LSA and more.
### ├──Sklearn standard model
The workflow involves scaling, and for each method, the scaling is adjusted accordingly. The table below summarizes scaling methods and model hyperparameters.

#### Scaling Method
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


#### Hyperparameters of `LogisticRegression` and `SVC`

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
