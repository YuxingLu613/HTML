import os
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold


def prepare_data_kfold(data_folder, k=5, modalities=[1, 2, 3]):
    labels = pd.read_csv(os.path.join(data_folder, "labels.csv"),
                         delimiter=',', names=["label"])["label"]
    dna_data = np.loadtxt(os.path.join(data_folder, "1.csv"),
                          dtype=np.float64, delimiter=',')
    mrna_data = np.loadtxt(os.path.join(
        data_folder, "2.csv"), dtype=np.float64, delimiter=',')
    mirna_data = np.loadtxt(os.path.join(
        data_folder, "3.csv"), dtype=np.float64, delimiter=',')

    data_list = [dna_data, mrna_data, mirna_data]
    data_list = [data_list[i - 1] for i in modalities]

    kf = KFold(n_splits=k, shuffle=True, random_state=40)

    data_train_list, data_test_list, train_labels, test_labels = [], [], [], []
    for train_index, test_index in kf.split(data_list[0]):
        train_data, test_data = [], []

        for i in range(len(data_list)):
            train_data.append(torch.FloatTensor(data_list[i][train_index]))
            test_data.append(torch.FloatTensor(data_list[i][test_index]))

        data_train_list.append(train_data)
        data_test_list.append(test_data)
        train_labels.append([int(i) for i in labels[train_index].tolist()])
        test_labels.append([int(i) for i in labels[test_index].tolist()])

    return data_train_list, data_test_list, train_labels, test_labels


def preprocess_data(DATA_TYPE):
    scaler = MinMaxScaler()

    labels = pd.read_csv(
        f"../data/{DATA_TYPE}/{DATA_TYPE}_labels.csv", names=["patient", "label"])

    meth = pd.read_csv(f"../data/{DATA_TYPE}/{DATA_TYPE}_meth.csv")
    pd.DataFrame(meth.columns[1:]).to_csv(
        f"../data/{DATA_TYPE}/1_featurename.csv", header=None, index=None)
    meth.replace("", np.nan)
    meth = meth.fillna(0)
    meth_value = meth.iloc[:, 1:]
    pd.DataFrame(scaler.fit_transform(meth_value), columns=meth_value.columns).to_csv(f"../data/{DATA_TYPE}/1.csv",
                                                                                      header=None, index=None)

    mrna = pd.read_csv(f"../data/{DATA_TYPE}/{DATA_TYPE}_mRNA.csv")
    pd.DataFrame(mrna.columns[1:]).to_csv(
        f"../data/{DATA_TYPE}/2_featurename.csv", header=None, index=None)
    mrna.replace("", np.nan)
    mrna = mrna.fillna(0)
    mrna_value = mrna.iloc[:, 1:]
    pd.DataFrame(scaler.fit_transform(mrna_value), columns=mrna_value.columns).to_csv(f"../data/{DATA_TYPE}/2.csv",
                                                                                      header=None, index=None)

    mirna = pd.read_csv(f"../data/{DATA_TYPE}/{DATA_TYPE}_miRNA.csv")
    pd.DataFrame(mirna.columns[1:]).to_csv(
        f"../data/{DATA_TYPE}/3_featurename.csv", header=None, index=None)
    mirna.replace("", np.nan)
    mirna = mirna.fillna(0)
    mirna_value = mirna.iloc[:, 1:]
    pd.DataFrame(scaler.fit_transform(mirna_value), columns=mirna_value.columns).to_csv(f"../data/{DATA_TYPE}/3.csv",
                                                                                        header=None, index=None)

    labels = pd.read_csv(f"../data/{DATA_TYPE}/{DATA_TYPE}_labels.csv")

    labels['label'] = pd.factorize(labels['type'])[0]
    labels["label"].to_csv(
        f"../data/{DATA_TYPE}/labels.csv", header=None, index=None)

    return labels["label"]
