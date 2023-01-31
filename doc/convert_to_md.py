import pandas as pd

df = pd.read_csv('doc/code_contributions_record.csv')
md = df.to_markdown(index=False).replace("(../", "(https://github.com/OSIPI/DCE-DSC-MRI_CodeCollection")
with open('notebook/contributors.md', 'w') as f:
    f.write("# Overview of code collection\n\n")
    f.write(md)