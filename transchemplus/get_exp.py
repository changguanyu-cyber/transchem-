import pandas as pd

def extract_smiles_and_exp_properties(
    input_csv,
    output_csv,
    smiles_col_index=0,
    exp_keyword="exp"
):

    df = pd.read_csv(input_csv)

    smiles_col = df.columns[smiles_col_index]

    exp_cols = [col for col in df.columns if exp_keyword in col.lower()]


    new_df = df[[smiles_col] + exp_cols]

    new_df.to_csv(output_csv, index=False)

extract_smiles_and_exp_properties(
    input_csv="/root/autodl-tmp/Uncertianty_quantification_Polymer_informatics-main/data/Gas_permeability_solubility_diffusivity_wide.csv",
    output_csv="/root/autodl-tmp/Uncertianty_quantification_Polymer_informatics-main/data/Gas_permeability_solubility_diffusivity_wide_exp.csv"
)
