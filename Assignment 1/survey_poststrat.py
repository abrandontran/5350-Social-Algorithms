### Brandon Tran (bat53)
### S&DS 5350 | Social Algorithms
### Assignment 1, Part 1, Steps 4–7



# Setup =======================================================================


## Import Packages ------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split



## Load Data ------------------------------------------------------------------
df = pd.read_csv("https://raw.githubusercontent.com/fivethirtyeight/data/master/comma-survey/comma-survey.csv")



# Core Function ===============================================================

def train_survey_models(df):
    demographics = [
    'Gender',
    'Age',
    'Household Income',
    'Education',
    'Location (Census Region)'
    ]

    questions = [
    'In your opinion, which sentence is more gramatically correct?',
    'Prior to reading about it above, had you heard of the serial (or Oxford) comma?',
    'How much, if at all, do you care about the use (or lack thereof) of the serial (or Oxford) comma in grammar?',
    'How would you write the following sentence?',
    'When faced with using the word "data", have you ever spent time considering if the word was a singular or plural noun?',
    'How much, if at all, do you care about the debate over the use of the word "data" as a singluar or plural noun?',
    'In your opinion, how important or unimportant is proper use of grammar?'
    ]

    # Preprocess Demographics:
    ## (1) Impute NaNs with 'Missing'.
    ## (2) Encode categorical variables with One Hot Encoder.

    preprocessor = ColumnTransformer(
        transformers = [
            ('cat', make_pipeline(
                SimpleImputer(strategy = 'constant', fill_value = 'Unknown'),
                OneHotEncoder(handle_unknown = 'ignore')
            ), demographics)
        ]
    )

    models = {}

    for q_col in questions:

        # Drop rows where answer is missing.
        valid_rows = df.dropna(subset=[q_col])
        X = valid_rows[demographics]
        y = valid_rows[q_col]

        # Encode target labels.
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Create and train model pipeline.
        clf = make_pipeline(
            preprocessor,
            LogisticRegression(solver = 'lbfgs', max_iter = 1000)
        )

        # Train model.
        clf.fit(X, y_encoded)

        # Store model and encoder.
        models[q_col] = {
            'model': clf,
            'encoder': le
        }

    return models



# Execute =====================================================================
trained_models = train_survey_models(df)