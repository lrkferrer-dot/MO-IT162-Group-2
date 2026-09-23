import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

# the 3 raw files we downloaded at Finmark site
df_demo = pd.read_csv("customer_demographics_contaminated.csv")
df_trans = pd.read_csv("customer_transactions_contaminated.csv")
df_social = pd.read_csv("social_media_interactions_contaminated.csv")

print(df_demo.shape, df_trans.shape, df_social.shape)
df_demo.head()

# just eyeballing what we're working with first
df_demo.info()
df_trans.info()
df_social.info()

# how many exact duplicate rows are hiding in each file
print("dupes:", df_demo.duplicated().sum(), df_trans.duplicated().sum(), df_social.duplicated().sum())

# where are the blanks
print(df_demo.isnull().sum())
print(df_trans.isnull().sum())
print(df_social.isnull().sum())

# Age has "Unknown" typed in instead of just being blank, and Amount has "Free" instead of 0 - caught these while scrolling through
print(df_demo['Age'].unique()[:5])
print(df_trans['Amount'].unique()[:5])

# dates are inconsistent too - some rows are YYYY-MM-DD, others are DD/MM/YYYY
# wrote this so we can parse both without losing rows
def fix_dates(col):
    def try_parse(val):
        if pd.isna(val):
            return pd.NaT
        try:
            return pd.to_datetime(val, format='%Y-%m-%d')
        except (ValueError, TypeError):
            pass
        try:
            return pd.to_datetime(val, format='%d/%m/%Y')
        except (ValueError, TypeError):
            return pd.NaT
    return col.apply(try_parse)

# ---------- demographics ----------
demo_clean = df_demo.copy()
demo_clean['Age'] = pd.to_numeric(demo_clean['Age'].replace('Unknown', np.nan), errors='coerce')
demo_clean['Age'] = demo_clean['Age'].fillna(demo_clean['Age'].median()).astype(int)  # median so outliers don't skew it
demo_clean['IncomeLevel'] = demo_clean['IncomeLevel'].fillna('Unknown')  # don't want to guess someone's income bracket
demo_clean['SignupDate'] = fix_dates(demo_clean['SignupDate'])

for col in ['Gender', 'Location', 'IncomeLevel']:
    demo_clean[col] = demo_clean[col].str.strip().str.title()  # inconsistent spacing/casing was everywhere

demo_clean = demo_clean.drop_duplicates()

print(demo_clean.isnull().sum().sum(), demo_clean.duplicated().sum())
demo_clean.head()

# ---------- transactions ----------
trans_clean = df_trans.copy()
trans_clean['Amount'] = pd.to_numeric(trans_clean['Amount'].replace('Free', 0), errors='coerce')
trans_clean['Amount'] = trans_clean['Amount'].fillna(trans_clean['Amount'].median())
trans_clean['Amount'] = trans_clean['Amount'].abs().round(2)  # negative purchase amount makes no sense, no refund column to explain it either
trans_clean['ProductCategory'] = trans_clean['ProductCategory'].fillna('Unknown')
trans_clean['TransactionDate'] = fix_dates(trans_clean['TransactionDate'])

trans_clean = trans_clean.drop_duplicates().drop_duplicates(subset='TransactionID', keep='first')  # some rows had same TransactionID but slightly different values

print(trans_clean.isnull().sum().sum(), trans_clean.duplicated().sum())
trans_clean.head()

# ---------- social media ----------
social_clean = df_social.copy()
social_clean['Platform'] = social_clean['Platform'].fillna('Unknown')
social_clean['Sentiment'] = social_clean['Sentiment'].fillna('Not Specified')  # didn't want to default this to "Neutral" since that's an actual sentiment value
social_clean['InteractionDate'] = fix_dates(social_clean['InteractionDate'])

social_clean = social_clean.drop_duplicates().drop_duplicates(subset='InteractionID', keep='first')

print(social_clean.isnull().sum().sum(), social_clean.duplicated().sum())
social_clean.head()

# before/after numbers for the report
summary = pd.DataFrame({
    'Dataset': ['Demographics', 'Transactions', 'Social Media'],
    'Rows Before': [len(df_demo), len(df_trans), len(df_social)],
    'Rows After': [len(demo_clean), len(trans_clean), len(social_clean)],
    'Dupes Before': [df_demo.duplicated().sum(), df_trans.duplicated().sum(), df_social.duplicated().sum()],
    'Nulls Before': [df_demo.isnull().sum().sum(), df_trans.isnull().sum().sum(), df_social.isnull().sum().sum()],
})
summary

# export the final versions for github
demo_clean.to_csv('customer_demographics_cleaned.csv', index=False)
trans_clean.to_csv('customer_transactions_cleaned.csv', index=False)
social_clean.to_csv('social_media_interactions_cleaned.csv', index=False)

print("saved all 3 cleaned files")
