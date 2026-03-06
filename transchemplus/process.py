import pandas as pd
import numpy as np
import re

def log_columns_only_if_scientific_notation(input_csv, output_csv):

    df = pd.read_csv(input_csv)
    scientific_cols = []

    for col in df.columns:

        col_as_str = df[col].astype(str)


        if col_as_str.str.contains(r"[Ee]").any():
            scientific_cols.append(col)



    for col in scientific_cols:

        df[col] = df[col].astype(str).str.replace("'", "")

        df[col] = pd.to_numeric(df[col], errors='coerce')

        df.loc[df[col] <= 0, col] = np.nan

        df[col] = np.log10(df[col])

    df.to_csv(output_csv, index=False)

log_columns_only_if_scientific_notation(
    input_csv="/root/autodl-tmp/Uncertianty_quantification_Polymer_informatics-main/data/MD_properties_simple.csv",
    output_csv="/root/autodl-tmp/Uncertianty_quantification_Polymer_informatics-main/data/New_properties_simple.csv"
)
