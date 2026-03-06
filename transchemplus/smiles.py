import os
import pandas as pd


input_csv = "/root/autodl-tmp/Uncertianty_quantification_Polymer_informatics-main/data/OMG_polymers.csv"          # 原始 CSV
output_dir = "/root/autodl-tmp/Uncertianty_quantification_Polymer_informatics-main/data"
output_csv = "New_OMG_polymers.csv"


df = pd.read_csv(input_csv)


smiles_series = pd.concat([
    df.iloc[:, 1],
    df.iloc[:, 2],
    df.iloc[:, 3],
], ignore_index=True)


smiles_series = smiles_series.dropna().astype(str)
smiles_series = smiles_series.drop_duplicates().reset_index(drop=True)


out_df = pd.DataFrame({
    "SMILES": smiles_series,
    "label": [1] * len(smiles_series)
})


output_path = os.path.join(output_dir, output_csv)
out_df.to_csv(output_path, index=False)

print(f"已生成文件: {output_path}")
