import pandas as pd


input_csv = "autodl-tmp/Uncertianty_quantification_Polymer_informatics-main/data/OMG_polymers_label1.csv"
output_csv = "autodl-tmp/Uncertianty_quantification_Polymer_informatics-main/data/OMG_polymers_label_test.csv"

df = pd.read_csv(input_csv)
new_df = df[["product", df.columns[-1]]]

new_df.to_csv(output_csv, index=False)

