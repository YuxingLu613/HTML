import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

def load_data_from_csv(data_folder, file_name, delimiter=','):
    """Load data from a CSV file.
    
    Args:
        data_folder (str): Path to the data folder.
        file_name (str): Name of the CSV file.
        delimiter (str): Delimiter used in the CSV.
        
    Returns:
        DataFrame: Loaded data.
    """
    return pd.read_csv(os.path.join(data_folder, file_name), delimiter=delimiter)

def preprocess_and_save(data, file_name, scaler):
    """Preprocess data using MinMax scaling and save to CSV.
    
    Args:
        data (DataFrame): Data to preprocess.
        file_name (str): Name of the CSV file to save processed data.
        scaler (MinMaxScaler): Scaler for MinMax scaling.
    """
    data.replace("", np.nan, inplace=True)
    data.fillna(0, inplace=True)
    data_values = data.iloc[:, 1:]
    processed_data = pd.DataFrame(scaler.fit_transform(data_values), columns=data_values.columns)
    processed_data.to_csv(file_name, header=None, index=None)

def prepare_data_kfold(data_folder, k=5, modalities=[1, 2, 3]):
    labels = load_data_from_csv(data_folder, "labels.csv")
    labels = labels["label"].tolist()
    
    dna_data = torch.FloatTensor(load_data_from_csv(data_folder, "1.csv").values)
    mrna_data = torch.FloatTensor(load_data_from_csv(data_folder, "2.csv").values)
    mirna_data = torch.FloatTensor(load_data_from_csv(data_folder, "3.csv").values)
    
    data_list = [dna_data, mrna_data, mirna_data]
    data_list = [data_list[i - 1] for i in modalities]
    
    kf = KFold(n_splits=k, shuffle=True, random_state=40)
    
    data_train_list, data_test_list, train_labels, test_labels = [], [], [], []
    for train_index, test_index in kf.split(data_list[0]):
        train_data, test_data = [], []

        for i in range(len(data_list)):
            train_data.append(data_list[i][train_index])
            test_data.append(data_list[i][test_index])

        data_train_list.append(train_data)
        data_test_list.append(test_data)
        train_labels.append([int(i) for i in labels[train_index]])
        test_labels.append([int(i) for i in labels[test_index]])

    return data_train_list, data_test_list, train_labels, test_labels

def preprocess_data(DATA_TYPE):
    scaler = MinMaxScaler()
    
    meth = load_data_from_csv(f"../data/{DATA_TYPE}/{DATA_TYPE}_meth.csv")
    preprocess_and_save(meth, f"../data/{DATA_TYPE}/1.csv", scaler)
    
    mrna = load_data_from_csv(f"../data/{DATA_TYPE}/{DATA_TYPE}_mRNA.csv")
    preprocess_and_save(mrna, f"../data/{DATA_TYPE}/2.csv", scaler)

    mirna = load_data_from_csv(f"../data/{DATA_TYPE}/{DATA_TYPE}_miRNA.csv")
    preprocess_and_save(mirna, f"../data/{DATA_TYPE}/3.csv", scaler)

    labels = load_data_from_csv(f"../data/{DATA_TYPE}/{DATA_TYPE}_labels.csv")
    labels['label'] = pd.factorize(labels['type'])[0]
    labels["label"].to_csv(f"../data/{DATA_TYPE}/labels.csv", header=None, index=None)

    return labels["label"]
