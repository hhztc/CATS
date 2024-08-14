import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

class DataPreprocess(object):
    def __init__(self):
        self.data = pd.DataFrame()

    def data_process(self, data: pd.DataFrame, data_length, sequence_length, num_features, method, is_data_enhancement):

        train_data = []
        valid_data = []
        train_labels = []
        valid_labels = []
        pre_data = []
        pre_label = []

        num_clients = data['cust_dk'].nunique()

        client_data = data.iloc[:num_clients * data_length, 1:].values.reshape((num_clients, data_length, num_features+1))

        if method == "train":
            for i in tqdm(range(num_clients), desc="Processing Clients"):
                for j in range(data_length - sequence_length):
                    sequence = client_data[i, j:j + sequence_length, :-1]
                    label = client_data[i, j + sequence_length, -1] 

                    if j == data_length - sequence_length - 1: 
                        valid_data.append(sequence)
                        valid_labels.append(label)
                    else:
                        train_data.append(sequence)
                        train_labels.append(label)

            train_x = np.array(train_data)
            valid_x = np.array(valid_data)
            train_y = np.array(train_labels).reshape(-1, 1)
            valid_y = np.array(valid_labels).reshape(-1, 1)
            padding_length_train = train_x.shape[0]
            train_masks = torch.ones((padding_length_train, sequence_length))
            padding_length_valid = valid_x.shape[0]
            valid_masks = torch.ones((padding_length_valid, sequence_length))

            if is_data_enhancement:
                train_x, train_y, train_masks = self.data_enhancement(train_x, train_y, num_copies=5)

            return train_x, valid_x, train_y, valid_y, train_masks, valid_masks

        elif method == "predict":
            for i in tqdm(range(num_clients), desc="Processing Clients"):
                for j in range(data_length - sequence_length):
                    sequence = client_data[i, j:j + sequence_length, :-1]
                    label = client_data[i, j + sequence_length, -1] 
                    pre_data.append(sequence)
                    pre_label.append(label)

            predict_x = np.array(pre_data)
            predict_y = np.array(pre_label).reshape(-1, 1)
            padding_length_predict = predict_x.shape[0]
            predict_masks = torch.ones((padding_length_predict, sequence_length))

            return predict_x, predict_y, predict_masks
        self.data = data

    def data_enhancement(self, original_data, original_data_y, num_copies):

        augmented_data = np.tile(original_data, (num_copies, 1, 1))
        augmented_data_y = np.tile(original_data_y, (num_copies, 1))
        
        masks = []
        for _ in range(num_copies):
            mask = np.random.choice([0, 1], size=original_data.shape[:2], p=[0.5, 0.5])
            masks.append(mask)

        original_mask = np.ones_like(original_data[:, :, 0])[:, :, np.newaxis]

        full_data = np.concatenate([original_data, augmented_data], axis=0)
        full_data_y = np.concatenate([original_data_y, augmented_data_y], axis=0)
        full_masks = np.concatenate([original_mask] + [mask[:, :, np.newaxis] for mask in masks], axis=0)

        full_data = np.array(full_data)
        full_masks = np.squeeze(np.array(full_masks), axis=-1)

        self.data = full_data

        return full_data, full_data_y, full_masks


if __name__ == '__main__':
    pass
