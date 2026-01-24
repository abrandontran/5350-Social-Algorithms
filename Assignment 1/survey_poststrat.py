### Brandon Tran (bat53)
### S&DS 5350 | Social Algorithms
### Assignment 1, Part 1, Steps 4–7



# Setup =======================================================================


## Import Packages ------------------------------------------------------------
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split



## Load Data ------------------------------------------------------------------
human_df = pd.read_csv("https://raw.githubusercontent.com/fivethirtyeight/data/master/comma-survey/comma-survey.csv")
llm_df = pd.read_csv("Assignment 1/gpt_survey.csv")



# Core Function ===============================================================

def train_survey_models(df, target_columns):
    
    demographics = [
    'Gender',
    'Age',
    'Household Income',
    'Education',
    'Location (Census Region)'
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

    for col in target_columns:

        # Drop rows where answer is missing.
        valid_rows = df.dropna(subset=[col])
        X = valid_rows[demographics]
        y = valid_rows[col]

        # Encode target labels.
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Bypass training if data contains only one class.
        if len(le.classes_) < 2:
            print(f"Skipping: {col[:40]}... (Only 1 answer type found: {le.classes_[0]})")
            continue

        # Create and train model pipeline.
        clf = make_pipeline(
            preprocessor,
            LogisticRegression(solver = 'lbfgs', max_iter = 1000)
        )

        # Train model.
        clf.fit(X, y_encoded)

        # Store model and encoder.
        models[col] = {
            'model': clf,
            'encoder': le
        }

        acc = clf.score(X, y_encoded)
        print(f"Trained: {col[:40]} accuracy {acc:.1%}")

    return models



# Execute =====================================================================

## Human (Original) Data ------------------------------------------------------
human_questions = [
    'In your opinion, which sentence is more gramatically correct?',
    'Prior to reading about it above, had you heard of the serial (or Oxford) comma?',
    'How much, if at all, do you care about the use (or lack thereof) of the serial (or Oxford) comma in grammar?',
    'How would you write the following sentence?',
    'When faced with using the word "data", have you ever spent time considering if the word was a singular or plural noun?',
    'How much, if at all, do you care about the debate over the use of the word "data" as a singluar or plural noun?',
    'In your opinion, how important or unimportant is proper use of grammar?'
]

human_models = train_survey_models(human_df, human_questions)

## LLM Data -------------------------------------------------------------------
llm_questions = [
    'llm_preference',
    'llm_prior',
    'llm_comma_care',
    'llm_sentence_example',
    'llm_data',
    'llm_data_care',
    'llm_grammar_importance'
]
        
llm_models = train_survey_models(llm_df, llm_questions)