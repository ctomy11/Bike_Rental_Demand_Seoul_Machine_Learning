# Bike_Rental_Demand_Seoul_Machine_Learning
This is Christo Tomy's report and accompanying materials on the topic of bike rental demand in Seoul, and if it can be accurately classified into demand categories using weather, temporal and calendar related features.


# Multiclass Classification of Bike-Sharing Demand in Seoul

# Project Overview

This project investigates whether hourly bike rental demand in Seoul can be effectively classified into multiple demand levels using weather, temporal, and calendar-based predictors. Rather than predicting exact rental counts, the problem is formulated as a multiclass classification task, categorising demand into low, medium, and high levels to support operational decision-making in bike sharing systems.


---

# Dataset

The analysis is based on the Seoul Bike Sharing Demand dataset, which contains hourly records of bike rental activity along with associated environmental and temporal variables.

- Observations: ~8,700 hourly records  
- Predictors: Weather conditions, time-of-day, seasonality, and calendar indicators  
- Target variable: Bike rental demand (categorised into low, medium, and high demand levels)

The dataset was obtained from a publicly available machine learning repository and satisfies the coursework requirements regarding size, feature count, and suitability for classification.

---

# Methodology

The modelling pipeline follows standard best practices in applied machine learning:

1. Data preprocessing
   - Cleaning and standardisation of variable names
   - Construction of a multiclass target variable using quantile-based thresholds
   - Stratified train/test split to preserve class balance
   - Feature scaling for models sensitive to predictor magnitude

2. Exploratory Data Analysis (EDA)
   - Analysis of temporal demand patterns
   - Examination of weather-related effects
   - Assessment of class balance and predictor correlations

3. Model Development
   Three classification models of increasing complexity were implemented:
   - Multinomial Generalised Linear Model (GLM) – baseline statistical classifier
   - Random forest – an ensemble-based machine learning model
   - Neural Network – nonlinear deep learning approach

4. Model Evaluation
   - Out-of-sample evaluation using a held-out test set
   - Performance assessed via accuracy, confusion matrices, and class-wise metrics
   - Comparative analysis focusing on accuracy, robustness, and interpretability

---

# Results Summary

All models achieved predictive performance substantially above the baseline of no information. The random forest classifier demonstrated the highest overall accuracy and the most balanced performance across demand levels. The neural network improved upon the GLM but did not outperform the random forest, highlighting the effectiveness of ensemble methods for structured tabular data.

---


