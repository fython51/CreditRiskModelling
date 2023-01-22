# Credit Risk Modelling using Logistic Regression
## Objective
Using a logistic regression model to predict loan status (default vs repayment).

<br>

## Dataset
### Source
In the code, the data is imported as a .csv file, but it's originally from Kaggle. [Click here](https://www.kaggle.com/datasets/laotse/credit-risk-dataset?select=credit_risk_dataset.csv) to access it.

### Data
The data contains information on 32,581 borrowers, divided into 11 variables:

| Variable | Type | Description |
| -------: | :--: |:----------- |
| **Age** | Numerical variable | Age in years |
| **Income** | Numerical variable | Annual income in dollars |
| **Home status** | Categorical variable | “Rent”, “mortgage” or “own” |
| **Employment length** | Numerical variable | Employment length in years |
| **Loan intent** | Categorical variable | “Education”, “medical”, “venture”, “home improvement”, “personal” or “debt consolidation” |
| **Loan amount** | Numerical variable | Loan amount in dollars |
| **Loan grade** | Categorical variable | “A”, “B”, “C”, “D”, “E”, “F” or “G” |
| **Interest rate** | Numerical variable | Interest rate in percentage |
| **Loan to income ratio** | Numerical variable | Between 0 and 1 |
| **Historical default** | Binary, categorical variable | “Y” or “N” |
| **Loan status** | Binary, numerical variable | 0 (no default) or 1 (default) → this is going to be our target variable |

_**Note:** thank you [Sarah Beshr](https://towardsdatascience.com/a-machine-learning-approach-to-credit-risk-assessment-ba8eda1cd11f) for the descriptions._

<br>

## Functioning
The script aims to train a logistic regression model to predict loan status using the given dataset, using the grid search to find the optimal hyperparameters. The code also includes some data visualization and cleaning steps to better understand the data and make the model more accurate.

<br>

## Sections
The script is divided into several sections, each with a specific purpose:
- The first section imports the necessary packages, including pandas, numpy, matplotlib, seaborn, and some modules from the scikit-learn library.
- The second section loads the data from a CSV file and stores it in a pandas DataFrame.
- The third section contains some data visualization code, specifically a loop that plots boxplots for each column in the DataFrame, and a heatmap that shows missing values in the data.
- The fourth section contains data cleaning code, which is used to remove outliers and deal with null values in the data.
- The fifth section defines a pipeline that includes a StandardScaler and a LogisticRegression object.
- The sixth section defines the hyperparameter space for the LogisticRegression object and creates a GridSearchCV object that will be used to tune the hyperparameters.
- The seventh section performs one-hot encoding on categorical variables, splits the data into training and testing sets, and fits the GridSearchCV object to the training data.
- The eighth section shows the best hyperparameters found by the GridSearchCV object and makes predictions on the test set using these hyperparameters.

<br>

## Output
Find the output code and the images in the "Output" folder.

<br>

## Other notes
Thanks [ChatGPT](https://chat.openai.com/chat) for writing most of the comments in the script.
