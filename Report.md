# Polynomial Regression Assignment Report
**Student Roll Number:** BT2024088

## 1. Introduction
This report documents the approach used to develop polynomial regression models for two personalized geothermal power plant datasets. The objective was to predict continuous target variables ($y$) accurately by discovering the optimal polynomial degrees and avoiding both underfitting and overfitting.

## 2. Methodology & Model Selection Strategy
Given the high complexity of the underlying physics (up to degree 10 for Phase 1 and up to degree 20 for Phase 2), a brute-force or manually guessed degree selection is prone to severe overfitting. 

To systematically find the optimal model, we utilized a **Cross-Validation (CV) Grid Search** strategy:
1. **Preprocessing Pipeline:**
   * **Input Scaling:** Applied `StandardScaler` to the raw features to ensure numerical stability.
   * **Polynomial Expansion:** Generated feature interactions using `PolynomialFeatures` (ranging from degree 1 to the maximum allowed degree).
   * **Term Scaling:** Scaled the newly generated high-degree polynomial terms before regression.
2. **Regularization Search:** For each degree, we evaluated Ridge, Lasso, and Elastic Net regressions across a grid of regularization strengths ($\alpha \in [0.001, 0.01, 0.1, 1.0, 10.0]$).
3. **Evaluation:** We used 5-fold cross-validation. The configuration with the lowest Mean Squared Error (MSE) across the validation folds was selected as the final model for that phase.

---

## 3. Phase 1: Power Plant Steam Turbine Optimization (var1)
The objective was to predict the Net Power Score ($y$) using 6 operational parameters ($x_1 \dots x_6$).

**Cross-Validation Search Results:**
The cross-validation pipeline searched polynomial degrees from 1 to 10. 

*   **Optimal Model Type:** Lasso Regression
*   **Optimal Polynomial Degree:** 5
*   **Optimal Regularization ($\alpha$):** 0.01
*   **Cross-Validation MSE:** 0.317081

**Rationale:**
The grid search revealed that a degree 5 polynomial using Lasso regularization achieved the best generalization. Lasso ($\mathcal{L}_1$ regularization) was particularly effective here because, at degree 5 with 6 features, the number of polynomial terms becomes very large (462 terms). Lasso inherently performs feature selection by shrinking the coefficients of less important interaction terms to exactly zero, preventing the model from capturing noise.

---

## 4. Phase 2: Subterranean Thermal Reservoir Mapping (var2)
The objective was to predict the Thermal Anomaly Score ($y$) based on 3D spatial coordinates ($x_1, x_2, x_3$).

**Cross-Validation Search Results:**
The cross-validation pipeline searched polynomial degrees from 1 to 20. 

*   **Optimal Model Type:** Ridge Regression
*   **Optimal Polynomial Degree:** 11
*   **Optimal Regularization ($\alpha$):** 1.0
*   **Cross-Validation MSE:** 0.237935

**Rationale:**
The 3D spatial data required a highly complex mapping, as evidenced by the selection of a degree 11 polynomial. Unlike Phase 1, the search preferred Ridge ($\mathcal{L}_2$ regularization) with a moderate penalty ($\alpha = 1.0$). This suggests that the thermal anomaly is distributed smoothly across the coordinate space, and many small, non-zero interaction terms contribute to the final score rather than a sparse set of dominant features.

## 5. Conclusion
By employing rigorous cross-validation and testing multiple regularization strategies, we successfully identified the optimal polynomial architectures for both geothermal problems. The models balance the complexity needed to map high-degree non-linear relationships with strict regularization penalties to guarantee robust performance on unseen data. The final predictions for both phases have been exported in the requested CSV format.
