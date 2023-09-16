from train.train import train_kfold
from preprocess.preprocessing import preprocess_data
import pandas as pd
import numpy as np
import json
import os

if __name__ == "__main__":
    # Initialize results dictionary and dataframe
    results = {}
    results_df = pd.DataFrame(columns=[
                              "Data Folder", "Accuracy", "F1-macro", "F1-weighted", "AUC score", "PRC score", "Uncertainty"])

    # Loop through data folders
    # for data_folder in ["COADREAD", "ESCA", "GBMLGG", "SARC", "STAD", "STES", "THCA", "UCEC", "BRCA", "KIPAN", "LGG", "ROSMAP"]:
    for data_folder in ["COADREAD"]:
        print(f"Processing {data_folder}...")

        # Preprocess data
        preprocess_data(data_folder)

        # Set paths and options for training
        data_folder_path = f"../data/{data_folder}/"
        modelpath = "../results/checkpoints"
        testonly = False

        # Train k-fold and store results
        result = train_kfold(data_folder_path, modelpath, testonly)
        results[data_folder] = result

    # Write results to JSON file
    with open("../results/HTML.json", "w") as f:
        json.dump(results, f)

    # Populate results dataframe
    for key, value in results.items():
        new_row = [key]
        for key2, val2 in value.items():
            mean = np.mean(val2)
            std_dev = np.std(val2)
            print("{:.2f} ({:.2f})".format(mean, std_dev))
            new_row.append("{:.2f} ({:.2f})".format(mean, std_dev))
        results_df.loc[len(results_df)] = new_row

    results_df.to_csv("../results/HTML.csv")