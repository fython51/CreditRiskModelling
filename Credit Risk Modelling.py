# -*- coding: utf-8 -*-
"""
Created on Jan 21 2023
Last update on Jan 22 2023
@author: fython51
"""

#%% Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, confusion_matrix, precision_recall_curve
sns.set_theme(style = "ticks")

#%% Load datasets
url = 'credit_risk_dataset.csv'                                                  # Load the dataset from a CSV file.
data = pd.read_csv(url)                                                          # Read the csv file and store it in a variable 'data'

#%% Data visualisation
for series in data:                                                              # Plot the boxplots for each column of data.
    try:
        sns.boxplot(data = data, 
                    x = series, 
                    hue = "loan_status")                                         # Plot boxplot of each column with respect to 'loan_status'.
        plt.title(series, fontweight='bold')                                     # Set the title of the plot.
        plt.show()                                                               # Display the plot.
    except:
        continue

# Visually inspect the null values.
mask = data.isnull()                                                             # Create a mask for the null values.
sns.heatmap(data.isnull(), cbar = False, cmap = "gist_gray")                     # Plot heatmap of missing values in the data.
plt.title("Missing values", fontweight='bold')                                   # Set the title of the plot.
plt.show()                                                                       # Display the plot.

#%% Data cleaning
# Manually clean the outliers
data = data[data["person_age"] <= 100]                                           # Remove the rows where age is greater than 100.
data = data[data["person_emp_length"] <= 100]                                    # Remove the rows where employment length is greater than 100.
data = data[data["person_income"] <= 4_000_000]                                  # Remove the rows where income is greater than 4,000,000.

# Count, and deal with, the null values
print(f"""
      There are {data.isna().sum().sum()} null values, which represent {round(data.isna().sum().sum() / len(data), 1)*100}% of observations.
      These are dropped.
      """)

data.dropna(axis = 0, inplace = True)                                           # Drop the rows with null values.

#%% Pipeline
# Setup the steps and the pipeline 
steps = [('scaler', StandardScaler()),                                          # Normalise the data by scaling.
         ('log_reg', LogisticRegression(solver = "liblinear"))]                 # Perform logistic regression.
pipeline = Pipeline(steps)                                                      # Create a pipeline object.

#%% Hyperparameters
# Specify the hyperparameter space
c_space = np.logspace(-5, 8, 15)                                                # Specify the range of C values
param_grid = {'log_reg__C': c_space,
              'log_reg__penalty': ['l1', 'l2']}                                 # Specify the regularization technique

#%% Data manipulation
# One-hot encoding of categorical variables.
df = pd.get_dummies(data = data, columns = ['person_home_ownership',            
                                            'loan_intent',
                                            'loan_grade',
                                            'cb_person_default_on_file']
                    )

# Train-test split.
y = df['loan_status']                                                           # Define the target variable.
X = df.drop('loan_status', axis=1)                                              # Define the feature set.
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = 0.2,
                                                    random_state = 42
                                                    )

#%% Hyperparameter tuning
# Create the GridSearchCV object.
grid_search = GridSearchCV(pipeline, 
                           param_grid, 
                           cv = 5, 
                           scoring = 'roc_auc')                                 # Use roc_auc as the scoring metric.

# Fit the GridSearchCV object to the training data.
print("\n... Fitting the data ...\n")
grid_search.fit(X_train, y_train)

# Show the best hyperparameters.
print(f"Best parameters: {grid_search.best_params_} \n")


#%% Predictions
# use the best hyperparameters to make predictions on the test set
print("\n... Making predictions ...\n")
y_pred = grid_search.predict(X_test)                                            # make predictions on the test set

#%% Evaluation
# Compute and plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)                                           # Calculate the confusion matrix
ax = sns.heatmap(cm, annot = True, fmt = "d", cbar = False)                     # Plot the confusion matrix as a heatmap
ax.set_xticklabels(["Predicted Repayment", "Predicted Default"])
ax.set_yticklabels(["True Repayment", "True Default"])
plt.title("Confusion Matrix", fontweight='bold')
plt.show()

# Compute and print the classification report
print(f"""Classification report
{classification_report(y_test, y_pred)}
""")

# Compute predicted probabilities
y_pred_prob = grid_search.predict_proba(X_test)[:,1]

# Generate ROC curve values
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve', fontweight = 'bold')
plt.show()

# Calculate precision and recall
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)

# Plot precision-recall curve
plt.plot(recall, precision)
plt.title('Precision-Recall Curve', fontweight='bold')
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.show()