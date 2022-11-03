import pandas as pd

df = pd.read_csv('doc/code_contributions_record.csv')
with open('doc/code_contributions_record.md', 'w') as md:
    df.to_markdown(buf=md, index=False)