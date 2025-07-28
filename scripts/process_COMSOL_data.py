import pandas as pd
from composite_ltes import root_dir

def process_file(file_path):
    # Load the data from the text file
    data = pd.read_csv(file_path, header=0, skiprows=8, sep=";")

    # Relabel the first header from "% X" to "X"
    data.columns.values[0] = 'X'
    
    # Split the dataset into two based on the column headers
    u_data = data.loc[:, data.columns.str.startswith('u')]
    H_u_data = data.loc[:, data.columns.str.startswith('H(u)')]

    # Add the first and second columns (X and the second column) to both datasets
    u_data = pd.concat([data.iloc[:, :2], u_data], axis=1)
    H_u_data = pd.concat([data.iloc[:, :2], H_u_data], axis=1)

    # Relabel all columns from column 2 onwards with the corresponding float
    for dataset in [u_data, H_u_data]:
        for i in range(2, len(dataset.columns)):
            col_name = dataset.columns[i]
            if '=' in col_name:
                new_label = float(col_name.split("=")[1].split(" s")[0])
                dataset.columns.values[i] = new_label

    # Save the datasets to new csv files
    base_name = file_path.name.split('.')[0]
    u_data.to_csv(root_dir() / "data" / f"{base_name}_u.csv", index=False)
    H_u_data.to_csv(root_dir() / "data" / f"{base_name}_H.csv", index=False)

# Loop over files
file_list = ["cell_data_mush.csv", "cell_data_sharp.csv"]
for file in file_list:
    process_file(root_dir() / "data" / file)