#  Cardiovascular Disease Prediction using Machine Learning

This project leverages supervised machine learning algorithms to predict the risk of cardiovascular disease (CVD) based on various health indicators. It involves extensive data cleaning, exploratory data analysis (EDA), model building, evaluation, and actionable recommendations.

---

##  Problem Statement

Cardiovascular disease is one of the leading causes of death globally. Early detection using machine learning can help identify high-risk individuals and take preventative actions. This project aims to build predictive models using health and lifestyle data to classify whether an individual is at risk for cardiovascular disease.

---

## 📂 Project Structure

---

##  Objectives

- Analyze the distribution and relationships between key health features.
- Build classification models to predict cardiovascular disease.
- Evaluate and compare model performance using various metrics.
- Provide data-driven recommendations and highlight potential limitations.

---

##  Technologies & Libraries

- **Python**: Programming language
- **Jupyter Notebook**: For analysis and documentation
- **Pandas / NumPy**: Data manipulation
- **Matplotlib / Seaborn**: Data visualization
- **Scikit-learn**: ML models and metrics
- **XGBoost**: Gradient boosting algorithm
- **LogisticRegressionCV**: Regularized logistic regression with built-in CV

---

##  Dataset Overview

The dataset contains information on several health-related attributes for individuals, including:

- Age (in days)
- Diastolic blood pressure
- Systollic blood pressure
- Gender
- Height and Weight
- Body Mass Index (BMI)
- Cholesterol level
- Glucose level
- Smoking status
- Alcohol intake
- Physical activity
- Presence of cardiovascular disease (target variable)

---

##  Methodology

### 1. Data Cleaning

- Converted age from days to years.
- One-hot encoded categorical features where appropriate.
- Ensured data types and class distributions were clean and usable.

### 2. Exploratory Data Analysis (EDA)
- Visualized age, gender, cholesterol, glucose, activity level, etc.
- Identified key relationships and patterns affecting cardiovascular risk.
- Detected skewed distributions and class imbalance.

### 3. Feature Engineering
- Created BMI from height and weight.
- Categorical feature encoding.
- Normalization of continuous features (optional depending on model).

### 4. Model Building
Trained and evaluated the following classifiers:
- **Logistic Regression with Cross-Validation**
- **Decision Tree**
- **Random Forest**
- **XGBoost Classifier**

### 5. Evaluation Metrics
Each model was evaluated using:
- Accuracy
- Precision, Recall, and F1-Score
- Confusion Matrix
- Cross-validation (where applicable)
- Classification Reports

---

## ✅ Results Summary
![Class balance plot]("Visuals/class balance.png")

![Accuracy Comparison]("Visuals/accuracy comparison.png"

![auc comparison]("Visuals/auc comparison.png")

![log xgboost comparison]("Visuals/log xgboost comparison.png")

![Confusion Matrix]("Visuals/cfm - Copy.png")

![xgboost feature importance](Visuals/xgboost feature importance.png")



- **Key Predictors**: Age,diastol blood pressure,systolic blood pressure cholesterol, glucose, and physical activity.
- **Smokers and alcohol consumers** showed little predictive power in this dataset.

---

##  Recommendations

1. **Use Ensemble Models for Production**  
   XGBoost should be deployed when accuracy is a priority.

2. **Consider Logistic Regression for Clinical Use**  
   It offers easier interpretation for healthcare professionals.

3. **Retrain Periodically**  
   Update models as new data becomes available to maintain performance.

4. **Collect Additional Data**  
   Include features like family history, blood pressure variability, and medication history.

5. **Deploy as a Web App**  
   Use Flask or Streamlit to provide predictions as a web service.

---

##  Limitations


- **Data Bias**  
  More females than males; could affect model fairness.

- **Limited Clinical Variables**  
  Dataset lacks some critical risk factors like medical history or genetics.

- **Interpretability**  
  Complex models (like XGBoost) can be hard to interpret without SHAP/LIME.

- **Static Dataset**  
  Not integrated with a real-time data pipeline.

---

## Future Work

- Visualize model explanations using **SHAP** or **LIME**.
- Deploy the model as an interactive dashboard.
- Add real-time inference or REST API integration.
- Experiment with deep learning models (if more data becomes available).

---



