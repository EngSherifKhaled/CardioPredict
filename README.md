# CardioPredict: AI-Powered Heart Disease Risk Predictor

CardioPredict leverages the UCI Heart Disease dataset to **estimate the probability of heart disease** in a patient.  
Explainable AI techniques are integrated to ensure model transparency and support clinical trust in predictions.

---

## Project Overview
CardioPredict demonstrates an end-to-end machine learning workflow, including data preprocessing, model training, evaluation, and explainability.  
It is designed for demonstration and portfolio purposes, emphasizing reproducibility and interpretability in healthcare ML.

---

## Dataset
- Source: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) (303 samples, 14 features + target)
- Target: `0` = no heart disease, `1` = heart disease present
- Features include numeric (age, trestbps, chol, thalach, oldpeak) and categorical (sex, cp, fbs, restecg, exang, slope, thal)
- Missing or implausible values handled during preprocessing
- Categorical variables encoded (binary directly, multi-class via pipeline)

---

## Modeling Approach
- Models evaluated: Logistic Regression, Decision Tree, Random Forest, XGBoost
- Hyperparameter tuning applied for Random Forest and XGBoost using GridSearchCV
- Final model: Random Forest
  - Accuracy: ~83%
  - ROC AUC: ~0.89
- Stratified train-test split and cross-validation ensure robust generalization

---

## Preprocessing & Feature Engineering
- Numerical features standardized; outliers capped at 99th percentile
- Skewed features (oldpeak) square-root transformed
- Irrelevant or low-importance features dropped (id, dataset, negligible features)
- Pipeline ensures reproducibility and prevents data leakage

---

## Explainability
- Random Forest feature importance highlights top predictors
- SHAP summary plots provide global interpretability
- SHAP waterfall plots explain patient-level predictions
- Supports transparency and trust for healthcare applications

---

## Inference-Time Handling
- Missing numeric and categorical inputs handled using **median/mode imputation** derived from training data
- Clinical variables **are not inferred** from other features, avoiding compounded uncertainty
- Users are informed of any missing data used in predictions

---

## Results
- Confusion matrix and ROC curve show strong classification performance
- Key predictors: Age, chest pain type (cp), maximum heart rate achieved (thalach)
- SHAP plots highlight patient-level risk factors

---

## Next Steps / Future Work
1. Finalize model artifacts
   - Ensure `rf_pipeline.pkl` and `impute_values.pkl` are saved and versioned
   - Confirm all plots are stored in Drive
2. Documentation & Reporting
   - Refined Step 19 summary in notebook
   - Update README.md with modeling, results, and inference-time handling
3. Demo / Presentation Layer
   - Build a minimal Streamlit app:
     - Input sliders/dropdowns with "Unknown" option
     - Output: predicted probability, class, and risk bar
     - Optional: SHAP waterfall for individual patient explanation
4. Reproducibility & Packaging
   - Clean folder structure
   - Add `requirements.txt`
   - Ensure end-to-end reproducibility
5. Optional / Future Work
   - Validate on an external dataset if available
   - Monitor predictions for drift (conceptual)
   - Ensemble strategies only if performance gains justify

---

## Project Structure

CardioPredict/
├── data/            # Original datasets
├── notebooks/       # Jupyter/Colab notebook
├── plots/           # Saved figures (confusion matrix, ROC, SHAP, waterfall, feature importance)
├── model/           # Final model artifacts (rf_pipeline.pkl, impute_values.pkl)
├── app/             # Streamlit demo application
├── requirements.txt # List of Python packages for reproducibility
└── README.md        # Project overview, methodology, results, disclaimer


---

## Disclaimer
This project is a **research and demonstration tool only**.  
It **does not provide clinical diagnoses**.  
Always consult qualified healthcare professionals for medical decisions.
