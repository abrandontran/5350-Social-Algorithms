### Brandon Tran (bat53)
### S&DS 5350 | Social Algorithms
### Assignment 1, Part 1, Step 1



# Setup ======================================================================


## Import Packages -----------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


## Load Data and Rename Columns ----------------------------------------------
df = pd.read_csv("https://raw.githubusercontent.com/fivethirtyeight/data/master/comma-survey/comma-survey.csv")

substantive_rename_dict = {
    'In your opinion, which sentence is more gramatically correct?': 'preference',
    'Prior to reading about it above, had you heard of the serial (or Oxford) comma?': 'prior',
    'How much, if at all, do you care about the use (or lack thereof) of the serial (or Oxford) comma in grammar?': 'comma_care',
    'How would you write the following sentence?': 'sentence_example',
    'When faced with using the word "data", have you ever spent time considering if the word was a singular or plural noun?': 'data',
    'How much, if at all, do you care about the debate over the use of the word "data" as a singluar or plural noun?': 'data_care',
    'In your opinion, how important or unimportant is proper use of grammar?': 'grammar_importance',
}

demographic_rename_dict = {
    'Household Income': 'Income',
    'Location (Census Region)': 'Region'
}

df = df.rename(columns = substantive_rename_dict)
df = df.rename(columns = demographic_rename_dict)



# Analysis ====================================================================


## Count Missingness ----------------------------------------------------------
print(df.isna().sum())


## Count Demographics ---------------------------------------------------------
demographics = ['Gender', 'Age', 'Income', 'Education', 'Region']

for attribute in demographics:
    proportions = df[attribute].value_counts(normalize=True).round(2)
    print(f"\n--- Distribution for {attribute} ---")
    
    for label, percent in proportions.items():
        print(f"{label}: {percent}%")


## Plot Demographics ----------------------------------------------------------

fig, axes = plt.subplots(3, 2, figsize = (16,12))

### Gender Distribution
sns.countplot(data = df, y = 'Gender', ax = axes[0,0])
axes[0,0].set_title('Distribution of Gender')

### Age Distribution
sns.countplot(data = df, y = 'Age', ax = axes[0,1], order = 
    ["18-29",
    "30-44",
    "45-60",
    "> 60"])
axes[0,1].set_title('Distribution of Age')

### Income Distribution
sns.countplot(data = df, y = 'Income', ax = axes[1,0], order = 
    ["$0 - $24,999", 
    "$25,000 - $49,999", 
    "$50,000 - $99,999", 
    "$100,000 - $149,999", 
    "$150,000+"])
axes[1,0].set_title('Distribution of Income')

### Region Distribution
sns.countplot(data = df, y = 'Region', ax = axes[1,1])
axes[1,1].set_title('Distribution of Census Region')

### Education Distribution
sns.countplot(data = df, y = 'Education', ax = axes[2,0], order =
    ["Less than high school degree", 
    "High school degree", 
    "Some college or Associate degree", 
    "Bachelor degree", 
    "Graduate degree"
    ])
axes[2,0].set_title('Distribution of Education')

axes[2,1].axis('off')
plt.tight_layout()
plt.show()


## Plot Substantive Responses -------------------------------------------------
fig, axes = plt.subplots(7, 1, figsize = (16,22))
axes = axes.flatten()

for i, (q,k) in enumerate(substantive_rename_dict.items()):
    sns.countplot(data = df, y = k, ax=axes[i])
    axes[i].set_title(f"{q}")
    axes[i].set_xlabel("Count")
    axes[i].set_ylabel("")

plt.tight_layout()
plt.show()