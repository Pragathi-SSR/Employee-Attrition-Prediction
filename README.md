# Employee-Attrition-Prediction
The objective is to develop a predictive model(ML) that can accurately forecast whether a data scientist candidate will remain with the company or move to another job post-training.

**PROJECT :** Employee Attrition Analysis in Data Science Training

**PROBLEM STATEMENT :** In the domain of employee attrition, a company specializing in Big Data and Data Science aims to predict which candidates, after completing training courses, are likely to join the company or seek new employment.

**DOMAIN :** Employee Attrition.

**CONTEXT :** The company collects demographic, educational, and professional background data from candidates during their course signup and enrollment processes. This prediction is crucial for optimizing training costs, improving course planning, and enhancing candidate categorization based on demographics, education, and prior experience data provided during enrollment.



![image](https://github.com/user-attachments/assets/1b83292d-e3dc-4625-958a-bd6889e44f80)

## Features
- Treating missing values with a suitable approach.
- Understanding the significant insights in the data by exploratory data analysis.
- Building supervised learning models to predict whether a employee leave the current job or not.
- Comparing accuracies and other scores to get the best model for deployment.
- Data balancing techniques(SMOTE).

## Dataset

https://www.kaggle.com/datasets/pavan9065/predicting-employee-attrition/data?select=train_data.csv

**DATA DESCRIPTION :** Many people sign up for their training. The company wants to know which of these candidates want to work for the company after training or looking for new employment because it helps to reduce the cost and time as well as the quality of training or planning the courses and categorization of candidates. Information related to demographics, education and experience is provided by candidates during signup and enrollment.

## Columns Description :

- **Enrollee_id:** Unique ID for candidate
- **City:** City code
- **City_development_index:** Developement index of the city (scaled)
- **Gender:** Gender of candidate
- **Relevent_experience:** Relevant experience of candidate
- **Enrolled_university:** Type of University course enrolled if any
- **Education_level:** Education level of candidate
- **Major_discipline:** Education major discipline of candidate
- **Experience:** Candidate total experience in years
- **Company_size**: No of employees in current employer's company
- **Company_type:** Type of current employer
- **Last_new_job:** Difference in years between previous job and current job
- **Training_hours:** training hours completed
- **Target:** 0 – Not looking for job change, 1 – Looking for a job change

## Tools and Techniques

- Python
- Numpy
- Pandas
- Matplotlib
- Seaborn
- KNN, SVM, Logistic Regression, Naive Bayes
- Decision tree, Random Forest, AdaBoost, CatBoost
- SMOTE
- GridSearchCV, RandomizedSearchCV
- Microsoft Word.

### Analysis
- There is an imbalance in the dataset's target features, which was addressed using oversampling techniques (SMOTE).
- After applying PCA, it was observed that all the features in the dataset are important in explaining the variance.

**Low Performing models:** 
- (i) F1 score is low for most of the models, but accuracy score and AUC scores are considerable for most of the models.
- (ii) As the data is imbalanced, even after applying oversampling (SMOTE) techniques, models are biased towards class_0.
- (iii) KNN, KNN_Tuned, Base_Decision Tree, and Random Forest models are overfitted as they perform well on training data but not on testing and validation datasets.
- (iv) SVM and AdaBoost models are also overfitted.

**Best Performed models:**

- ((i) Both Gradient Boosting algorithm and CatBoost models are giving similar kinds of scores on the validation dataset and they are not overfitted on training data.
- (ii) LightGBM also has a good accuracy score, but it is giving a lower recall score compared to the above two models.
- (iii) The CatBoost algorithm takes less time compared to the AdaBoost Classifier model.
- (iv) So, the best performer among all the models is **"CatBoost"** Model.


### Model Scores
- **Accuracy:** 75%

- **Precision:** 50%

- **Recall:** 70% 
 
- **F1-Score:** 57%

- **AUC_roc_score:** 74%



 **Outcome:** The model implementation helps the company predict employee attrition.






