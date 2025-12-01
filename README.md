Credit Card Fraud Detection:
*This project builds a machine learning model to detect fraudulent credit card transactions.
*It uses data preprocessing, feature engineering, model training, and evaluation techniques to classify transactions as fraud or legitimate.

Project Overview:

The notebook performs the following:
Loads and cleans the credit card dataset
Converts all features (including Time, Amount, and PCA components) to numeric
Handles missing values and dtype inconsistencies
Splits data into training and testing sets
Builds a scikit-learn Pipeline with preprocessing
Trains a Random Forest Classifier
Evaluates the model with:
Accuracy
Precision
Recall
ROC AUC
Confusion Matrix
ROC Curve

Technologies Used:
Python 3
pandas
matplotlib
seaborn
scikit-learn

How to Run:
1. Clone this repository
git clone https://github.com/your-username/your-repo.git
cd your-repo
2. Install dependencies
pip install -r requirements.txt
3. Run the notebook
Open Jupyter Notebook or VS Code and run:
jupyter notebook readme.ipynb

Model Evaluation:

The notebook outputs:
Classification metrics
Confusion matrix visualization
ROC curve and AUC score
These help assess how well the model identifies fraudulent transactions.

Dataset:
The dataset includes:
PCA-transformed numerical features (V1–V28)
Time and Amount
Target variable: Class (0 = legitimate, 1 = fraud)

Future Improvements:
Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
Balancing methods (SMOTE, undersampling, oversampling)
Trying other models (XGBoost, LightGBM, Logistic Regression)
Deployment as an API or web dashboard
 
 
 Results
  The model's performance was evaluated on the test set (X_test, y_test).
  The key metrics achieved are:MetricScoreAccuracy0.9995Precision0.9655Recall0.8000ROC AUC0.9694
