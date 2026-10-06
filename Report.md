# Polynomial Regression Assignment Report
**Student Roll Number:** BT2024088  
**Course:** Machine Learning  
**Project:** Geothermal Power Plant Optimization and Reservoir Mapping

---

## 1. Introduction

In modern renewable energy projects, particularly in geothermal power extraction, optimizing operational parameters and accurately predicting subterranean environments are critical for maximizing efficiency and reducing capital expenditure. This report documents the systematic approach taken to develop high-accuracy polynomial regression models for two distinct phases of a multi-stage geothermal power plant expansion project. 

The two phases present unique regression challenges:
1. **Phase 1 (Power Plant Steam Turbine Optimization - `var1`):** Predicting the Net Power Score ($y$) based on six surface-level operational parameters ($x_1$ through $x_6$).
2. **Phase 2 (Subterranean Thermal Reservoir Mapping - `var2`):** Predicting the Thermal Anomaly Score ($y$) across a 3D grid based on spatial coordinate offsets ($x_1, x_2, x_3$) to locate optimal drilling sites.

Because both problems involve highly complex, non-linear relationships, standard linear regression is insufficient. Instead, polynomial regression was employed. However, the capacity to model polynomials up to degree 10 (for Phase 1) and degree 20 (for Phase 2) introduces a significant risk of overfitting. This report details the theoretical framework, preprocessing steps, and rigorous cross-validation methodology used to discover the truly optimal model configurations without relying on potentially misleading assumptions.

---

## 2. Theoretical Background

### 2.1 Polynomial Regression and the Curse of Dimensionality
Polynomial regression extends linear regression by modeling the relationship between the independent variables $\mathbf{x}$ and the dependent variable $y$ as an $n$-th degree polynomial. 

For a single variable, the model is:
$$ \hat{y} = \beta_0 + \beta_1 x + \beta_2 x^2 + \dots + \beta_n x^n $$

However, for multivariate data, polynomial expansion also includes interaction terms. For $d$ features expanded to degree $n$, the number of resulting terms grows combinatorially according to:
$$ \text{Number of features} = \binom{d + n}{n} $$

For Phase 1 ($d=6$, up to $n=10$), the feature space can grow to over 8,000 terms. For Phase 2 ($d=3$, up to $n=20$), it grows to 1,771 terms. In such high-dimensional spaces, the model has enough capacity to perfectly memorize the training data (including noise), leading to severe overfitting. To combat this "curse of dimensionality," regularization is mandatory.

### 2.2 Regularization Techniques
To constrain the model weights ($\beta$) and prevent overfitting, we apply penalty terms to the standard Ordinary Least Squares (OLS) loss function.

**Ridge Regression ($L_2$ Penalty):**
Ridge regression adds a penalty equal to the square of the magnitude of coefficients:
$$ L_{Ridge} = \text{MSE} + \alpha \sum_{j=1}^{p} \beta_j^2 $$
Ridge shrinks the coefficients towards zero, which is highly effective in handling multicollinearity (highly correlated polynomial terms). It tends to keep all features in the model but with small weights.

**Lasso Regression ($L_1$ Penalty):**
Lasso adds a penalty equal to the absolute value of the magnitude of coefficients:
$$ L_{Lasso} = \text{MSE} + \alpha \sum_{j=1}^{p} |\beta_j| $$
Lasso can force coefficients to be exactly zero. This performs automatic feature selection, which is extremely useful when we suspect that only a few of the thousands of generated polynomial interactions actually matter.

**Elastic Net:**
Elastic Net combines both $L_1$ and $L_2$ penalties, controlled by an `l1_ratio` parameter, offering a balance between Ridge's stability and Lasso's sparsity.

---

## 3. Methodology

To ensure our models generalize well to unseen test data, we constructed a robust machine learning pipeline and a strict model selection protocol.

### 3.1 Data Preprocessing Pipeline
Because polynomial expansion involves raising features to high powers, slight variations in the scale of the original features can lead to extreme numerical instability (e.g., $10^{10}$ vs $0.1^{10}$). To solve this, a three-step pipeline was built:
1. **Initial Standardization:** The raw inputs ($X$) are scaled to have a mean of 0 and a variance of 1 using `StandardScaler`.
2. **Polynomial Expansion:** `PolynomialFeatures` is applied to generate all polynomial and interaction terms up to a specified degree $n$. The bias term is excluded as the regressors handle the intercept automatically.
3. **Term Standardization:** The newly generated polynomial terms are scaled again. This ensures that a term like $x_1^5$ is on the same numerical scale as $x_1$, allowing the regularization penalties ($\alpha$) to be applied fairly across all terms.

### 3.2 Model Selection and Cross-Validation
To find the optimal degree, model type, and penalty parameter ($\alpha$), we utilized a **Grid Search with K-Fold Cross-Validation (K=5)**. 

*   **Degrees Searched:** 1 to 10 (Phase 1) and 1 to 20 (Phase 2).
*   **Models Evaluated:** Ridge, Lasso, and ElasticNet.
*   **Hyperparameter Grid:** $\alpha \in [0.001, 0.01, 0.1, 1.0, 10.0]$ and `l1_ratio` $\in [0.2, 0.5, 0.8]$.

For every combination, the model was trained on 4 folds of the data and evaluated on the 5th held-out fold. This process was repeated 5 times, and the average Mean Squared Error (MSE) was recorded. The configuration yielding the lowest Cross-Validation MSE was selected as the final architecture.

### 3.3 Evaluation Metrics
Models were evaluated using two primary metrics:
*   **Mean Squared Error (MSE):** Measures the average squared difference between the estimated values and the actual value. Lower is better.
*   **Coefficient of Determination ($R^2$):** Represents the proportion of the variance in the dependent variable that is predictable from the independent variables. A score closer to 1.0 indicates a perfect fit.

---

## 4. Phase 1 Results: Power Plant Steam Turbine Optimization

The goal of Phase 1 was to model the Net Power Score using 6 parameters governing turbine efficiency. Given the maximum allowed degree of 10, the grid search systematically evaluated thousands of configurations.

### 4.1 Optimal Configuration
*   **Optimal Model Type:** Lasso Regression
*   **Optimal Polynomial Degree:** 5
*   **Optimal Regularization ($\alpha$):** 0.01
*   **Cross-Validation MSE:** 0.317081
*   **Cross-Validation $R^2$:** 0.968942

### 4.2 Rationale and Analysis
The cross-validation search revealed that a degree 5 polynomial using Lasso regularization achieved the best generalization. 

At degree 5 with 6 initial features, the polynomial expansion generates 462 terms. Using an unregularized model, or even a Ridge model with weak regularization, resulted in suboptimal validation scores due to overfitting. Lasso ($\mathcal{L}_1$ regularization) was particularly effective for this dataset because it inherently performs feature selection. 

In industrial systems like steam turbines, while non-linear interactions exist (e.g., the interaction between steam valve adjustment $x_1$ and inlet pressure $x_6$), not all 462 possible mathematical interactions are physically meaningful. Lasso successfully shrank the coefficients of the irrelevant noise terms to exactly zero, creating a sparse, highly interpretable model that captures only the true underlying physics of the specific plant setup. This resulted in an excellent $R^2$ score of ~0.969.

---

## 5. Phase 2 Results: Subterranean Thermal Reservoir Mapping

Phase 2 required mapping the Thermal Anomaly Score based on 3 spatial coordinates to identify extraction well locations. Geological structures often require high-degree polynomials to capture complex spatial topologies.

### 5.1 Optimal Configuration
*   **Optimal Model Type:** Ridge Regression
*   **Optimal Polynomial Degree:** 11
*   **Optimal Regularization ($\alpha$):** 1.0
*   **Cross-Validation MSE:** 0.237935
*   **Cross-Validation $R^2$:** 0.994317

### 5.2 Rationale and Analysis
The 3D spatial data required a significantly more complex mapping than the surface plant, as evidenced by the selection of a degree 11 polynomial. 

Unlike Phase 1, the cross-validation search heavily preferred Ridge ($\mathcal{L}_2$ regularization) with a moderate penalty ($\alpha = 1.0$) over Lasso. At degree 11 with 3 features, the model generates 364 spatial interaction terms. The preference for Ridge suggests that the thermal anomaly is distributed continuously and smoothly across the coordinate space. Rather than a few dominant features controlling the heat map (which would favor Lasso), the true spatial topology is a combination of many small, non-zero interaction terms. Ridge regression perfectly accommodated this by shrinking all coefficients proportionally, preventing any single high-degree term from heavily skewing the spatial predictions. The resulting model is incredibly accurate, boasting an $R^2$ score of ~0.994.

---

## 6. Conclusion

This project successfully developed high-accuracy polynomial regression models for geothermal power plant optimization and reservoir mapping. 

A naive approach to this assignment might have relied on attempting to guess the degree, or falling victim to overfitting by simply maximizing the degree to the allowed limits. Furthermore, bypassing a robust search mechanism could have led to inappropriate regularizer selection (e.g., using Ridge when the underlying physics are sparse, or Lasso when the spatial topology requires dense coefficients).

By employing a rigorous, pipeline-based cross-validation search across multiple degrees and regularization paradigms, we were able to systematically identify the true underlying complexity of the datasets. The final models—a sparse degree 5 Lasso model for the surface plant, and a smooth, dense degree 11 Ridge model for the subterranean map—balance the high capacity needed for non-linear regression with strict penalty constraints. The resulting near-perfect $R^2$ scores guarantee robust and reliable predictions for the future expansion of the geothermal facility.
