import os
import pandas as pd
from train.train import train_kfold
from preprocess.preprocessing import preprocess_data

def initialize_results():
    """Initialize results dictionary and dataframe."""
    results = {}
    results_df = pd.DataFrame(columns=[
        "Data Folder", "Accuracy", "F1-macro", "F1-weighted", 
        "AUC score", "PRC score", "Uncertainty"
    ])
    return results, results_df

def preprocess_and_train(data_folder, model_path):
    """Preprocess data and execute k-fold training.
    
    Args:
        data_folder (str): Name of the data folder to process.
        model_path (str): Path to save model checkpoints.
        
    Returns:
        dict: Results of the training.
    """
    print(f"Processing {data_folder}...")
    
    preprocess_data(data_folder)
    
    data_folder_path = os.path.join("..", "data", data_folder)
    testonly = False
    result = train_kfold(data_folder_path, model_path, testonly)
    
    return result

def main():
    """Main function to execute preprocessing and training."""
    
    # Initialization
    results, results_df = initialize_results()
    
    # Defining model path and dataset names
    model_path = os.path.join("..", "results", "checkpoints")
    datasets = ["COADREAD"]  # Extend this list for processing multiple datasets
    
    for data_folder in datasets:
        result = preprocess_and_train(data_folder, model_path)
        results[data_folder] = result
        
    # Further code to handle results, save to files, etc. can be added here.

if __name__ == "__main__":
    main()