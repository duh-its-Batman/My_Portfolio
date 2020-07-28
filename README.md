# Hello World!

You're looking at an organized individual with a knack for data and numbers. Eyeing to be part of a professionally data driven team in an esteemed organization to learn some new relevant skills and work on real-time projects meanwhile upgrading my current skillset, to be an important asset for the company.

## [Project 1: Loan Prediction](https://github.com/duh-its-Batman/Data-Science-and-ML-Projects/blob/master/Loan_prediction_Git.ipynb)
---
### Overview -
- A Kaggle competition classification dataset aiming at predicting the status of **loan approval** (Y = 1 and N = 0) on the basis of multiple variables associated with it.
- Found patterns through EDA which further helped in feature engineering.
- The target variable was having class imbalance, so applied **re-sampling techniques** (Random Undersampling, Random Oversampling and SMOTE) to deal with it.
- Predictive Modeling pipeline of the project used the following:
    
    - Algorithms:
         * **Logistic Regression**
         * **Decision Tree Classifier**
         * **Random Forest Classifier**
         * **K-Nearest Neighbors**
         * **Support Vector Classifier**
   
    - Metrics (for evaluation):
         * **Accuracy Score** (84.53%, after ensembling)
         * **ROC - AUC Score** (76.68%, after ensembling)
    
    - Optimization:
         * used **Grid Search CV** for tuning Hyper-parameters.
         * tried a basic **Ensemble modeling** method, in order to observe improvement (if any) in the metrics score.
 

## [Project 2: House Price prediction (Region - Ames, Iowa, USA)](https://github.com/duh-its-Batman/Data-Science-and-ML-Projects/blob/master/House_Price_prediction_Git.ipynb)
---
### Overview -
- This **regression model** was downloaded from the Kaggle competition with the main objective being, prediction of housing sales prices of Ames city in Iowa state of USA.
- The dataset contains **81** columns to deal with.
- With higher independent variables, comes great responsibility of dealing with missing values, outliers, skewness (since it's regression) and other feature processing complexities. 
- Applied **Principal Component Analysis** for feature selection.
- Predictive Modeling pipeline of the project used the following:

    - Algorithms:
        - **Linear Regression**
        - **Ridge Regression**
        - **Lasso Regression**
        - **Elastic Net Regression**
        - **Kernal Regression**
        - **Gradient Boosting**
        - **XGBoost**
        - **Light GBM**
        
     
     - Metrics (for evaluation):
        - **MAPE**
        - **RMSLE**
        
     
     - Optimization:
        - **Grid Search CV**, for tuning Hyper-parameters.
        - used a basic **Ensemble modeling** method. (RMSLE - 0.0724)
        


## [Project 3: Credit Card Fraud Detection](https://github.com/duh-its-Batman/Data-Science-and-ML-Projects/blob/master/Credit_card_Fraud_Detection_Git.ipynb)
---
### Overview - 
- A supervised classification problem with the underlying objectives were to find patterns about time of fraud during a day (if any) and making a prediction on that basis to identify first-hand whether the transaction belongs to fraud or legit category. The dataset was obtained from Kaggle.
- The dataset has majority of variables unavailable to us due to prior PCA compression, only 3 variables were directly available for our purpose.
- Target variable was highly imbalanced (**0: 99.82%** and **1: 0.18%**), so the application of **re-sampling techniques** (Random Undersampler, Random Oversampler and SMOTE) appeared to be highly prudent.
- Post scrutinizing the dataset, I found that fraud transaction has a much higher probabaility to occur during the night time period specifically between **2:00 am - 3:00 am**.
- Predictive Modeling pipeline of the project used the following: 

    - Algorithms:
        - **Logistic Regression**
        - **Decision Tree Classifier**
        - **Random Forest Classifier**
        - **K-Nearest Neighbors**
        - **Support Vector Classifier**
        
     
     - Metrics (for evaluation):
        - **Accuracy Score**
        - **ROC - AUC Score**
        - **Confusion Matrix**

- Best model:
    - **Random Oversampling with Random Forest Classifier**, reasons:
        - Accuracy Score = **99.993527%**
        - ROC - AUC Score = **99.993508%**
        - Confusion matrix = **False Negatives - 0** and **False Positives - 11**


## [Micro-Project: Weather in Szeged, Hungary](https://github.com/duh-its-Batman/Data-Science-and-ML-Projects/blob/master/Weather_Szeged_Git.ipynb)
---
### Overview -
- To find the factors having an impact over the weather conditions in Szeged, Hungary. A kaggle dataset containing hourly/daily summary of the data between the time period 2006 - 2016.
- Much of the time was spent into the **EDA** to understand the latent patterns.
- Among all the variables, **Apparant Temperature**, **Precipitation (type - rain)**, **Visibility**, **Humidity** and **Precipitation (type - snow)**, were the top 5 most  correlated variables w.r.t Temperature (target) variable.
- After fitting the **Linear Regression** model, the R-squared came out to be **99.04%**, but main focus was put on **ANOVA** and **VIF**.
- In the end, 
    
    1. **Apparent Temperature (in degree C)**

    2. **Humidity**

    3. **Wind speed (km/h)**

    4. **Pressure (in millibars)**
        
were derived to be the most significant variables to impact the Temperature of that region.


## [Micro-Project: Cardiotocography - Fetal State Class Code Prediction](https://github.com/duh-its-Batman/Data-Science-and-ML-Projects/blob/master/Cardiotocography_Git.ipynb)
---
### Overview - 
- This multiclass classification prediction dataset was obtained from **UCI Machine Learning Repository**. Classification was both with respect to a morphologic pattern (A, B, C. ...) and to a fetal state (N, S, P). Therefore, the dataset can be used either for 10-class or 3-class experiments. I focused on the prediction of **3-class fetal state** in my study of the case.
- The target variable has 3 classes:
    - **Normal** (N)
    - **Suspecion** (S)
    - **Pathogenic** (P)
- Predictive Modeling segment:

    - **Logistic Regression** applied.
    
    - Metrics used for evaluation, with their respective scores:
        - Accuracy Score = **94.92%**
        - ROC - AUC Score = **92.24%**
        
    - Tried using repeated **Stratified KFold**, Accuracy Score = **92.5%**


## [Micro-Project: Mall Customers - Customer Segmentation using K-means Clustering](https://github.com/duh-its-Batman/Data-Science-and-ML-Projects/blob/master/Mall_Customers_Git.ipynb)
---
### Overview - 
- Implementation of K-mean Clustering algorithm comprised under unsupervised learning. Focused on customer segmentation application of clustering algorithm by using the mall customers dataset from Kaggle.
- Was able to find some significant patterns in the data which can help us in deriving important buziness decisions later.
- After administering the K-means by using **k-means++** initialisation method for centroid placement and **Elbow Curve** to find optimal number of clusters (**6**), I got **inertia**: **181.9514**, which is an evaluation metric for cluster evaluation. 
    


