import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


class DataPreprocess1(object):
    def __init__(self):
        self.data_length = 12000
        self.data_x = None
        self.data_y = None

    def data_process(self, data: pd.DataFrame, timestep: int, date_length: int, feature_size: int, method: str):
        if method == "train":
            data = data
        else:
            data = data
        data_x = []
        data_y = []

        user_count = data['cust_dk'].nunique()

        # 将整个窗口的数据保存到X中，将未来一天保存到Y
        index = 0
        j = 0
        tqdm_total = (date_length - timestep) * user_count
        pbar = tqdm(total=tqdm_total, desc="Data Processing")
        while index < len(data) - timestep:
            if index == j + (date_length - timestep):
                index = index + timestep
                j = j + date_length
            else:
                features = data.iloc[index: index + timestep, 1:feature_size+1].copy()
                label = data.iloc[index + timestep:index + timestep + 1, -1].copy()
                data_x.append(features)
                data_y.append(label)
                index = index + 1
            pbar.update(1)

        data_x = np.array(data_x)
        data_y = np.array(data_y)

        self.data_x = data_x
        self.data_y = data_y
        padding_length = data_x.shape[0]
        padding_masks = torch.ones((padding_length, timestep))
        return data_x, data_y, padding_masks


class DataPreprocess(object):
    def __init__(self):
        self.data = pd.DataFrame()

    def data_process(self, data: pd.DataFrame, data_length, sequence_length, num_features, method, is_data_enhancement):
        # 创建训练集和验证集
        train_data = []
        valid_data = []
        train_labels = []
        valid_labels = []
        pre_data = []
        pre_label = []

        num_clients = data['cust_dk'].nunique()
        # 将数据转换为三维数组，每个客户的数据作为一个二维数组
        client_data = data.iloc[:num_clients * data_length, 1:].values.reshape((num_clients, data_length, num_features+1))

        if method == "train":
            for i in tqdm(range(num_clients), desc="Processing Clients"):
                for j in range(data_length - sequence_length):
                    sequence = client_data[i, j:j + sequence_length, :-1]
                    label = client_data[i, j + sequence_length, -1]  # 最后一列作为标签

                    if j == data_length - sequence_length - 1:  # 最后一条序列作为验证集
                        valid_data.append(sequence)
                        valid_labels.append(label)
                    else:
                        train_data.append(sequence)
                        train_labels.append(label)

                # 转换为 NumPy 数组
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
                    label = client_data[i, j + sequence_length, -1]  # 最后一列作为标签
                    pre_data.append(sequence)
                    pre_label.append(label)

            predict_x = np.array(pre_data)
            predict_y = np.array(pre_label).reshape(-1, 1)
            padding_length_predict = predict_x.shape[0]
            predict_masks = torch.ones((padding_length_predict, sequence_length))

            return predict_x, predict_y, predict_masks
        self.data = data

    def data_enhancement(self, original_data, original_data_y, num_copies):
        # 1. 复制原始样本n份
        augmented_data = np.tile(original_data, (num_copies, 1, 1))
        augmented_data_y = np.tile(original_data_y, (num_copies, 1))

        # 2. 为每份样本生成不同的 mask
        masks = []
        for _ in range(num_copies):
            mask = np.random.choice([0, 1], size=original_data.shape[:2], p=[0.5, 0.5])
            masks.append(mask)

        # 3. 将原始样本对应的 mask 设为全1
        original_mask = np.ones_like(original_data[:, :, 0])[:, :, np.newaxis]

        # 4. 将原始样本和5份样本拼接在一起，考虑 mask
        full_data = np.concatenate([original_data, augmented_data], axis=0)
        full_data_y = np.concatenate([original_data_y, augmented_data_y], axis=0)
        full_masks = np.concatenate([original_mask] + [mask[:, :, np.newaxis] for mask in masks], axis=0)

        # 5. 将样本和 mask 转换为 NumPy 数组的形式
        full_data = np.array(full_data)
        full_masks = np.squeeze(np.array(full_masks), axis=-1)

        self.data = full_data

        return full_data, full_data_y, full_masks


if __name__ == '__main__':
    # option_feature = ['date', 'cust_dk', 'day_of_month', 'attendance_mode', 'interval_from_last_purchase',
    #                   'avg_interval_last_7_days', 'target']
    #
    # obj = DataPreprocess()
    # df = pd.read_csv(f"../data/interim/train_transform_data_cluster_1.csv")
    # # df = pd.read_csv(f"../data/interim/predict_transform_data_cluster_0.csv")
    # df = df[option_feature]
    # #
    # # obj.data_process(df, data_length=8, sequence_length=7, num_features=5, method="predict")
    # obj.data_process(df, data_length=60, sequence_length=7, num_features=5, method="train", is_data_enhancement=True)

    # obj1 = DataPreprocess1()
    # obj1.data_process(df, timestep=7, date_length=60, feature_size=5, method="train")
    pass
