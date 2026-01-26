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



## Load Data ------------------------------------------------------------------
human_df = pd.read_csv("https://raw.githubusercontent.com/fivethirtyeight/data/master/comma-survey/comma-survey.csv")
llm_df = pd.read_csv("Assignment 1/gpt_survey.csv")
census_df = pd.read_csv("Assignment 1/census_demographics.csv")
llm_census_df = pd.read_csv("Assignment 1/gpt_census_survey.csv")


# Core Functions ==============================================================


## Post-Stratification I ------------------------------------------------------

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


## Post-Stratification II -----------------------------------------------------

def get_poststrat_estimates(models_dict, census_df):

    estimates = {}
    demographics = ['Gender', 'Age', 'Household Income', 'Education',
                    'Location (Census Region)']

    for q_col, model_data in models_dict.items():
        model = model_data['model']
        encoder = model_data['encoder']

        cell_probs = model.predict_proba(census_df[demographics])
        total_weight = census_df['count'].sum()
        weighted_probs = np.average(cell_probs, axis = 0, weights = census_df['count'])

        estimates[q_col] = dict(zip(encoder.classes_, weighted_probs))

    return estimates



# Execute =====================================================================

## Train Models ---------------------------------------------------------------
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


## Post-Stratified Census Projections -----------------------------------------
human_estimates = get_poststrat_estimates(human_models, census_df)
llm_estimates = get_poststrat_estimates(llm_models, census_df)


## Raw Means for In Silico Responses ------------------------------------------
llm_census_estimates = {}
for col in llm_questions:
    if col in llm_census_df.columns:
        llm_census_estimates[col] = llm_census_df[col].value_counts(normalize=True).to_dict()



# Analysis ====================================================================


## Comparsion Table -----------------------------------------------------------
print("\n" + "="*160)
# Header keys:
# (i)   Raw Human
# (ii)  Raw LLM (Random)
# (iii) PS Human (Adjusted)
# (iv)  PS LLM (Adjusted)
# (v)   LLM (Census Sampled)
print(f"{'QUESTION / ANSWER':<85} | {'(i)Raw H':>10} | {'(ii)Raw L':>10} | {'(iii)PS H':>10} | {'(iv)PS L':>10} | {'(v)Cen L':>10}")
print("="*160)

comparisons = list(zip(human_questions, llm_questions))

for h_q, l_q in comparisons:
    # Print Full Question
    print(f"\nQ: {h_q}") 
    
    # 1. Get Pre-Calculated Estimates (iii & iv)
    res_iii = human_estimates.get(h_q, {})
    res_iv  = llm_estimates.get(l_q, {})
    
    # 2. Calculate Raw Means on the fly (i, ii, & v)
    res_i  = human_df[h_q].value_counts(normalize=True).to_dict()
    res_ii = llm_df[l_q].value_counts(normalize=True).to_dict()
    res_v  = {}
    if 'llm_census_df' in locals() and not llm_census_df.empty and l_q in llm_census_df.columns:
        res_v = llm_census_df[l_q].value_counts(normalize=True).to_dict()

    # 3. Get all unique answers to align rows
    all_answers = sorted(list(set(
        list(res_i.keys()) + list(res_ii.keys()) + 
        list(res_iii.keys()) + list(res_iv.keys()) + list(res_v.keys())
    )))
    
    for ans in all_answers:
        v_i   = res_i.get(ans, 0)
        v_ii  = res_ii.get(ans, 0)
        v_iii = res_iii.get(ans, 0)
        v_iv  = res_iv.get(ans, 0)
        v_v   = res_v.get(ans, 0)
        
        print(f"   {ans:<85} | {v_i:>9.1%}  | {v_ii:>9.1%}  | {v_iii:>9.1%}  | {v_iv:>9.1%}  | {v_v:>9.1%}")


## Export to CSV --------------------------------------------------------------
pd.DataFrame(human_estimates).to_csv("Assignment 1/human_poststrat_estimates.csv")
pd.DataFrame(llm_estimates).to_csv("Assignment 1/llm_poststrat_estimates.csv")