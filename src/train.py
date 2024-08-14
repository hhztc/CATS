import os
import torch
import random
import mlflow
import logging
import numpy as np
import pandas as pd
import torch.nn as nn
import config as config

from model.Gru import GRU
from model.AttentionGru import AttentionGRU
from model.Lstm import LSTM
from model.LstNet import LSTNet
from model.DLinear import DLinear
from model.TimesNet import TimesNet
from model.Informer import Informer
from model.PatchTST import PatchTST
from model.Transformer import Transformer
from model.Crossformer import Crossformer

from tqdm import tqdm
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset
from dataclasses_json import dataclass_json


from dataset.data_preprocess import DataPreprocess
from dataset.feature_connect_dataset import FeatureConnect
from feature.raw_data_transform import DatasetTransformPipeline
from dataset.creat_sample_dataset_main import AutoEncodeMode, TrainSampleDataset

logging.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
                    level=logging.INFO, datefmt='%a, %d %b %Y %H:%M:%S')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed_value = 2024
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)  

torch.manual_seed(seed_value)  
torch.cuda.manual_seed(seed_value)  


@dataclass_json
@dataclass
class TrainModule(object):
    def __init__(self, args):

        self.model_type = args.model_type  

        self.params = config.model_type_dict[f'{self.model_type}']

        self.params['hidden_size'] = args.hidden_size
        self.params['d_model'] = args.d_model
        self.params['seq_len'] = args.seq_length  

        self.train_sample_data = pd.DataFrame()

        self.train_index_data = pd.DataFrame()

        self.train_feature_data = pd.DataFrame()

        self.train_transform_data = pd.DataFrame()

        self.option_feature = ['date', 'cust_dk', 'day_of_month', 'attendance_mode',
                               'interval_from_last_purchase', 'avg_interval_last_7_days',
                               'target']

        self.train_loader = None

        self.test_loader = None

        self.predict_loader = None  
        self.dataset_id = args.dataset_id

        self.seq_length = args.seq_length  

        self.train_interval = args.train_interval  

        self.n_clusters = args.n_clusters

        self.user_length = args.user_length

        self.data_length = None  
  
        self.epochs = args.epochs  
        self.learning_rate = args.learning_rate  
        self.batch_size = args.batch_size  
        self.step_size = args.step_size  
        self.gamma = args.gamma

        self.weight_decay = args.weight_decay  
        self.loss_function = None

        self.optimizer = None

        self.model = None

        self.model_save_path = args.model_output_path + f'dataset_{self.dataset_id}/'

        self.is_cluster = args.is_cluster

        self.is_dataEnhancement = args.is_dataEnhancement

        self.tans_obj = DatasetTransformPipeline()

        logging.info('-----是否使用聚类：{}'.format(self.is_cluster))
        logging.info('-----是否启动数据增强：{}'.format(self.is_dataEnhancement))

    def create_train_index_dataset(self):
        obj = TrainSampleDataset(dataset_id=self.dataset_id, user_length=self.user_length, train_interval=self.train_interval)
        self.train_sample_data = obj.main()

        obj = AutoEncodeMode(self.dataset_id)
        self.train_index_data = obj.encode_mode_train_data(data=self.train_sample_data)

        train_index_data_path = f"../data/interim/dataset_{self.dataset_id}/train_index_data.csv"
        self.train_index_data.to_csv(train_index_data_path, index=False)

    def connect_feature_engineer(self):
        obj = FeatureConnect(dataset_id=self.dataset_id, index_data=self.train_index_data)
        self.train_feature_data = obj.main()
        cluster_data = pd.read_csv(f"../data/cluster_data/dataset_{self.dataset_id}/cluster_{self.n_clusters}_output_data.csv")
        self.train_feature_data = pd.merge(self.train_feature_data, cluster_data, on='cust_dk', how='left')
        self.train_feature_data = self.train_feature_data.drop(['sequence_value'], axis=1)
        self.train_feature_data.to_csv(f"../data/interim/dataset_{self.dataset_id}/train_feature_data.csv")

    def _train_data_fit_transform(self):
        self.train_transform_data = self.tans_obj.fit_transform(input_dataset=self.train_feature_data)
        self.train_transform_data.to_csv(f"../data/interim/dataset_{self.dataset_id}/train_transform_data.csv", index=False)
        with open(f"../data/interim/dataset_{self.dataset_id}/transform.json", "w+") as dump_file:
            dump_file.write(self.tans_obj.to_json())

    def _cluster_group(self, is_cluster: bool):
        train_transform_data = pd.read_csv(f"../data/interim/dataset_{self.dataset_id}/train_transform_data.csv")
        if is_cluster:
            for cluster in range(0, self.n_clusters):
                data = train_transform_data[train_transform_data['cluster'] == cluster]
                data.to_csv(f"../data/interim/dataset_{self.dataset_id}/train_transform_data_cluster_{cluster}.csv", index=False)
        else:
            logging.warning('------当前模式为不加入聚类模块-----')

    def _data_preprocess(self, cluster: int, is_cluster: bool):
        obj = DataPreprocess()
        if is_cluster:
            data = pd.read_csv(f"../data/interim/dataset_{self.dataset_id}/train_transform_data_cluster_{cluster}.csv")
            data = data[self.option_feature]
            train_x, valid_x, train_y, valid_y, train_masks, valid_masks = obj.data_process(data, data_length=self.train_interval, sequence_length=self.seq_length, num_features=5, method="train", is_data_enhancement=self.is_dataEnhancement)
            np.savez(f"../data/interim/dataset_{self.dataset_id}/TimesSeries_train_cluster_{cluster}",
                     train_x=train_x, valid_x=valid_x, train_y=train_y, valid_y=valid_y, train_masks=train_masks, valid_masks=valid_masks)
        else:
            data = pd.read_csv(f"../data/interim/dataset_{self.dataset_id}/train_transform_data.csv")
            data = data[self.option_feature]
            train_x, valid_x, train_y, valid_y, train_masks, valid_masks = obj.data_process(data, data_length=self.train_interval, sequence_length=self.seq_length, num_features=5, method="train", is_data_enhancement=False)
            np.savez(f"../data/interim/dataset_{self.dataset_id}/TimesSeries_train_cluster_None",
                     train_x=train_x, valid_x=valid_x, train_y=train_y, valid_y=valid_y, train_masks=train_masks, valid_masks=valid_masks)

    def _split_dataset(self, cluster: int, is_cluster: bool):
        if is_cluster:
            processed_data = np.load(f"../data/interim/dataset_{self.dataset_id}/TimesSeries_train_cluster_{cluster}.npz")
        else:
            processed_data = np.load(f"../data/interim/dataset_{self.dataset_id}/TimesSeries_train_cluster_None.npz")
        train_x = processed_data['train_x']
        valid_x = processed_data['valid_x']
        train_y = processed_data['train_y']
        valid_y = processed_data['valid_y']
        train_masks = processed_data['train_masks']
        valid_masks = processed_data['valid_masks']

        logging.info("train_x.shape: {}".format(train_x.shape))
        logging.info("valid_x.shape: {}".format(valid_x.shape))
        logging.info("train_y.shape: {}".format(train_y.shape))
        logging.info("valid_y.shape: {}".format(valid_y.shape))
        logging.info("train_masks.shape: {}".format(train_masks.shape))
        logging.info("valid_masks.shape: {}".format(valid_masks.shape))

        x_train_tensor = torch.from_numpy(train_x).to(torch.float32)
        y_train_tensor = torch.from_numpy(train_y).to(torch.float32)
        padding_mask_train = torch.from_numpy(train_masks).to(torch.int16)

        x_val_tensor = torch.from_numpy(valid_x).to(torch.float32)
        y_val_tensor = torch.from_numpy(valid_y).to(torch.float32)
        padding_mask_val = torch.from_numpy(valid_masks).to(torch.int16)  
        train_data = TensorDataset(x_train_tensor, y_train_tensor, padding_mask_train)
        val_data = TensorDataset(x_val_tensor, y_val_tensor, padding_mask_val)  
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size, shuffle=False)

    def _init_model(self, option_model: str):

        logging.info("----model type is : {}".format(self.model_type))

        if option_model == "gru":  
            self.model = GRU(self.params)

        elif option_model == "lstm":
            self.model = LSTM(self.params)

        elif option_model == "LstNet":
            self.model = LSTNet(self.params)

        elif option_model == "DLinear":
            self.model = DLinear(self.params)

        elif option_model == "TimesNet":
            self.model = TimesNet(self.params)

        elif option_model == "Transformer":
            self.model = Transformer(self.params)

        elif option_model == "Informer":
            self.model = Informer(self.params)

        elif option_model == "PatchTST":
            self.model = PatchTST(self.params)

        elif option_model == "AttentionGRU":
            self.model = AttentionGRU(self.params)
            logging.info("hidden_size: {}".format(self.params['hidden_size']))
            logging.info("d_model: {}".format(self.params['d_model']))
            logging.info("seq_length: {}".format(self.seq_length))

        self.model.to(device)
        self.loss_function = nn.BCELoss()
        self.loss_function = self.loss_function.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)  
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)

    def train(self, cluster: int, is_cluster: bool):
        best_auc = 0.0  
  
        mlflow.set_experiment(f"test-v1.0.1")
        run_name = f"model:{self.model_type}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("params", self.params)
            for epoch in range(1, self.epochs + 1):
                self.model.train()
                train_loss = []
                train_bar = tqdm(self.train_loader)  
                for data in train_bar:
                    x_train, y_train, padding_mask = data  
                    x_train, y_train, padding_mask = x_train.to(device), y_train.to(device), padding_mask.to(device)
                    y_train_pred = self.model(x_train, padding_mask)
                    loss = self.loss_function(y_train_pred, y_train)

                    train_loss.append(loss.item())
                    loss.backward()

                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    train_bar.desc = "     epoch[{}/{}]".format(epoch, self.epochs, loss)

                train_loss = np.average(train_loss)
                lr = self.scheduler.get_last_lr()[0]
                self.scheduler.step()

                logging.info("train epoch[{}/{}] loss:{:.4f} learning_rate:{}".format(epoch, self.epochs, train_loss, lr))
                mlflow.log_metric("loss", train_loss, step=epoch)
                if epoch % 2 == 0:  
                    self.model.eval()
                    val_bar = tqdm(self.val_loader)
                    y_test_pred_list = []
                    y_test_label_list = []
                    test_loss = []
                    with torch.no_grad():
                        for data in val_bar:
                            x_test, y_test, padding_mask = data
                            x_test, y_test, padding_mask = x_test.to(device), y_test.to(device), padding_mask.to(device)
                            y_test_pred = self.model(x_test, padding_mask)
                            y_test_pred_list.append(y_test_pred)
                            y_test_label_list.append(y_test)
                            loss = self.loss_function(y_test_pred, y_test)
                            test_loss.append(loss.item())
                            val_bar.desc = "     valid at Epoch:{}".format(epoch)  
                        y_test_pred_array = np.concatenate([tensor.cpu().numpy() for tensor in y_test_pred_list])
                        y_test_label_array = np.concatenate([tensor.cpu().numpy() for tensor in y_test_label_list])
                        val_auc = roc_auc_score(y_test_label_array, y_test_pred_array)
                        test_loss = np.average(test_loss)
                        mlflow.log_metric("val_auc", val_auc, step=epoch)
                        logging.info("test_loss: {:.4f} best_auc: {:.4f}  val_auc :{:.4f} ".format(test_loss, best_auc, val_auc))

                    if best_auc < val_auc:
                        best_auc = val_auc
                        if is_cluster:
                            model_save_path = self.model_save_path + "best_" + self.model_type + f"_model_cluster_{cluster}_dataset_id_{self.dataset_id}.pth"
                        else:
                            model_save_path = self.model_save_path + "best_" + self.model_type + f"_model_cluster_None_dataset_id_{self.dataset_id}.pth"
                        logging.info('-----Best model save as：{}'.format(model_save_path))
                        torch.save(self.model.state_dict(), model_save_path)

        logging.info('-----Done, Best model save as：{}'.format(model_save_path))

    def main(self):
        self.create_train_index_dataset()
        self.connect_feature_engineer()
        self._train_data_fit_transform()
        self._cluster_group(is_cluster=self.is_cluster)
        if self.is_cluster is False:
            self.n_clusters = 1
        for cluster in range(0, self.n_clusters):
            self._data_preprocess(cluster, is_cluster=self.is_cluster)
            self._split_dataset(cluster, is_cluster=self.is_cluster)
            self._init_model(option_model=self.model_type)
            self.train(cluster, is_cluster=self.is_cluster)


if __name__ == '__main__':  
  
    pass
