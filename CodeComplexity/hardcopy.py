import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns


file_path = "C:/Users/QinQin/Desktop/Virtual_Env/miniMLproject/codeComplexity/syntheticData.csv"
df = pd.read_csv(file_path)

# ======================
# HARDCODED RULES (Business Logic or rule-based function, not ML)
# Just rules, if data complexity is high, this is not recommended
# ======================

def hardcode_predict(user):   
    # Premium Rules (business logic)
    if (user['income_k'] > 85 and 
        user['job'] in ['Engineer', 'Healthcare'] and 
        user['click_rate'] > 0.35):
        return "Premium"
    
    if (user['loyalty_status'] == 'Gold' and 
        user['income_k'] > 70 and 
        user['avg_browsing_time_mins'] > 20):
        return "Premium"
    
    # Budget Rules
    if ((user['job'] == 'Student' and user['age'] <= 24) or 
        (user['age'] < 25 and user['income_k'] < 35)):
        return "Budget"
    
    if (user['click_rate'] < 0.2 and 
        user['avg_browsing_time_mins'] < 10 and 
        user['cart_additions'] < 2):
        return "Budget"
    
    # Mid-Range Rules
    if (40 <= user['income_k'] <= 75 and 
        user['loyalty_status'] in ['Silver', 'Bronze'] and
        15 <= user['avg_browsing_time_mins'] <= 30):
        return "Mid-Range"
    
    if (user['location'] == 'Urban' and 
        user['job'] in ['Teacher', 'Retail'] and 
        user['income_k'] > 50):
        return "Mid-Range"
    
    # other rules
    if (user['job'] == 'Healthcare' and 
        user['location'] == 'Rural'):
        return "Premium" if user['income_k'] > 60 else "Mid-Range"
    
    if (user['click_rate'] > 0.5 and 
        user['income_k'] < 40):
        return "Mid-Range"
    
    #  if none of the previous 8 rules matched, return "Mid-Range"
    return "Mid-Range"


# Apply hardcoded rules (create new column with predictions : hardcode_pred))
df['hardcode_pred'] = df.apply(hardcode_predict, axis=1)

print("=== Prediction Distribution (compare the Actual & Prediction) ===")
dist_df = pd.DataFrame({
    'Actual': df['product'].value_counts(normalize=True),
    'Hardcoded': df['hardcode_pred'].value_counts(normalize=True)
})
print(dist_df)


print("\n=== Rule Trigger Analysis ===")
rule_counts = {
    'Premium Rules': len(df[df['hardcode_pred'] == 'Premium']),                 #count how many predictions are Premium
    'Budget Rules': len(df[df['hardcode_pred'] == 'Budget']),                   #count how many predictions are Budget
    'Mid-Range Rules': len(df[df['hardcode_pred'] == 'Mid-Range']),             #count how many predictions are Mid-Range
    'Fallback Rules': len(df[df['hardcode_pred'] == 'Mid-Range']) - len(df[     #count how many predictions are not match 8 rules
        (df['income_k'].between(40,75)) |
        (df['location'] == 'Urban') |
        (df['job'] == 'Healthcare')
    ])
}
print(pd.Series(rule_counts))

## Take Note ##
# The fallback count appear negative — not because the prediction is wrong,
# but because we're comparing predicted values against a rough filter that only checks part of the rule logic.
# The actual prediction (`hardcode_pred`) is correct and only assigns one label per user.
# But our fallback estimate subtracts users who *partially* match a rule (e.g., income only),
# even though they didn’t meet the full AND conditions in the actual rule.
# That’s why the result may show a negative fallback value — it's an artifact of how we estimate rule matches.


print("\n=== Sample Predictions ===")
sample_cols = ['age', 'income_k', 'job', 'product', 'hardcode_pred']
print(df[sample_cols].sample(5).to_markdown(tablefmt="grid"))