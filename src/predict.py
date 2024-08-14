import os
import sys
import torch
import mlflow
import logging
import numpy as np
import pandas as pd
import config as config

from tqdm import tqdm
from model.Gru import GRU
from model.Lstm import LSTM
from model.LstNet import LSTNet
from model.DLinear import DLinear
from model.TimesNet import TimesNet
from model.Informer import Informer
from model.Crossformer import Crossformer
from model.PatchTST import PatchTST
from model.Transformer import Transformer
from model.AttentionGru import AttentionGRU

from torch.utils.data import TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

from dataset.data_preprocess import DataPreprocess
from dataset.feature_connect_dataset import FeatureConnect
from feature.raw_data_transform import DatasetTransformPipeline
from dataset.creat_sample_dataset_main import AutoEncodeMode,  PredictSampleDataset

from train import TrainModule

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')
logger.root.setLevel(level=logging.INFO)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

local_url = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(local_url)


def _load_transform(transformer_path: str) -> DatasetTransformPipeline:
    with open(transformer_path, "r+") as dump_file:
        transformer = DatasetTransformPipeline.from_json(dump_file.read())
    return transformer


class Predict(TrainModule):
    def __init__(self, args):
        super(Predict, self).__init__(args)

        self.params['hidden_size'] = args.hidden_size
        self.params['d_model'] = args.d_model
        self.params['seq_len'] = args.seq_length

        self.predict_interval = args.seq_length 

        self.predict_length = args.predict_length 

        self.predict_start_date = "2023-10-1" 

        self.predict_sample_data = pd.DataFrame()

        self.predict_index_data = pd.DataFrame()

        self.predict_encode_data = pd.DataFrame()  

        self.predict_feature_data = pd.DataFrame() 

        self.predict_transform_data = pd.DataFrame() 

        self.option_feature = config.option_feature

        self.output_data = pd.DataFrame() 

        self.predict_loader = None

        self.batch_size = args.batch_size

        self.model = None

        self.avg_auc = 0

        self.avg_acc = 0

        self.avg_precision = 0

        self.avg_recall = 0

        self.avg_f1score = 0

        self.top_value = args.top_N

        self.threshold_value = 0

        self.is_cluster = args.is_cluster

        self.predict_output_path = args.predict_output_path + f"dataset_{self.dataset_id}/"

    def create_predict_index_dataset(self):
        obj = PredictSampleDataset(dataset_id=self.dataset_id, predict_date=self.predict_start_date, predict_interval=self.seq_length, predict_length=self.predict_length, user_length=self.user_length)
        self.predict_sample_data, self.output_data = obj.main()

        obj = AutoEncodeMode(self.dataset_id)
        self.predict_index_data = obj.encode_mode_predict_data(self.predict_sample_data)

        predict_index_data_path = f"../data/interim/dataset_{self.dataset_id}/predict_encode_data.csv"
        self.predict_index_data.to_csv(predict_index_data_path, index=False)

    def connect_feature_engineer(self):
        obj = FeatureConnect(dataset_id=self.dataset_id, index_data=self.predict_index_data)
        self.predict_feature_data = obj.main()
        cluster_data = pd.read_csv(f"../data/cluster_data/dataset_{self.dataset_id}/cluster_{self.n_clusters}_output_data.csv")
        self.predict_feature_data = pd.merge(self.predict_feature_data, cluster_data, on='cust_dk', how='left')
        self.predict_feature_data = self.predict_feature_data.drop(['sequence_value'], axis=1)
        self.predict_feature_data.to_csv(f"../data/interim/dataset_{self.dataset_id}/predict_feature_data.csv")

    def transform(self):
        transform = _load_transform(transformer_path=f"../data/interim/dataset_{self.dataset_id}/transform.json")
        self.predict_transform_data = transform.transform(input_dataset=self.predict_feature_data)
        self.predict_transform_data.to_csv(f"../data/interim/dataset_{self.dataset_id}/predict_transform_data.csv", index=False)

    def _cluster_group(self, is_cluster: bool):
        predict_transform_data = pd.read_csv(f"../data/interim/dataset_{self.dataset_id}/predict_transform_data.csv")
        if is_cluster:
            for i in range(0, self.n_clusters):
                data = predict_transform_data[predict_transform_data['cluster'] == i]
                data.to_csv(f"../data/interim/dataset_{self.dataset_id}/predict_transform_data_cluster_{i}.csv", index=False)
                self.output_data = data[data['date'] == '2023-10-01']
                self.output_data = self.output_data[['date', 'cust_dk']]
                self.output_data.to_csv(f"../data/interim/dataset_{self.dataset_id}/output_index_cluster_{i}_data.csv")
        else:
            logging.warning('------当前模式为不加入聚类模块-----')

    def data_preprocess(self, cluster: int, is_cluster: bool):
        obj = DataPreprocess()
        if is_cluster:
            data = pd.read_csv(f"../data/interim/dataset_{self.dataset_id}/predict_transform_data_cluster_{cluster}.csv")
            date_length = self.seq_length + self.predict_length
            data = data[self.option_feature]
            predict_x, predict_y, predict_mask = obj.data_process(data=data,  data_length=date_length, sequence_length=self.seq_length, num_features=5, method="predict", is_data_enhancement=False)
            np.savez(f"../data/interim/dataset_{self.dataset_id}/TimesSeries_predict_cluster_{cluster}", data_x=predict_x, data_y=predict_y, predict_mask=predict_mask)
        else:
            data = pd.read_csv(f"../data/interim/dataset_{self.dataset_id}/predict_transform_data.csv")
            date_length = self.seq_length + self.predict_length
            data = data[self.option_feature]
            predict_x, predict_y, predict_mask = obj.data_process(data=data, data_length=date_length, sequence_length=self.seq_length, num_features=5, method="predict", is_data_enhancement=False)
            np.savez(f"../data/interim/dataset_{self.dataset_id}/TimesSeries_predict_cluster_None", data_x=predict_x, data_y=predict_y, predict_mask=predict_mask)

    def dataloader(self, cluster: int, is_cluster: bool):
        if is_cluster:
            processed_data = np.load(f"../data/interim/dataset_{self.dataset_id}/TimesSeries_predict_cluster_{cluster}.npz")
        else:
            processed_data = np.load(f"../data/interim/dataset_{self.dataset_id}/TimesSeries_predict_cluster_None.npz")

        predict_x = processed_data['data_x']
        predict_y = processed_data['data_y']
        padding_mask = processed_data['predict_mask']

        logging.info("predict_x.shape: {}".format(predict_x.shape))
        logging.info("predict_y.shape: {}".format(predict_y.shape))
        logging.info("padding_mask.shape: {}".format(padding_mask.shape))

        x_predict_tensor = torch.from_numpy(predict_x).to(torch.float32)
        y_predict_tensor = torch.from_numpy(predict_y).to(torch.float32)
        padding_mask = torch.from_numpy(padding_mask).to(torch.int16)

        predict_data = TensorDataset(x_predict_tensor, y_predict_tensor, padding_mask)

        self.predict_loader = torch.utils.data.DataLoader(predict_data, batch_size=self.batch_size, shuffle=False)

    def load_model(self, option_model: str, cluster: int, is_cluster: bool):

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

        elif option_model == "Crossformer":
            self.model = Crossformer(self.params)

        elif option_model == "AttentionGRU":
            self.model = AttentionGRU(self.params)
            logging.info("hidden_size: {}".format(self.params['hidden_size']))
            logging.info("d_model: {}".format(self.params['d_model']))
            logging.info("seq_length: {}".format(self.seq_length))


        self.model.to(device)

        if is_cluster:
            model_save_path = self.model_save_path + "best_" + self.model_type + f"_model_cluster_{cluster}_dataset_id_{self.dataset_id}.pth"
        else:
            model_save_path = self.model_save_path + "best_" + self.model_type + f"_model_cluster_None_dataset_id_{self.dataset_id}.pth"

        logging.info("-----加载模型权重：{} ".format(model_save_path))
        self.model.load_state_dict(torch.load(model_save_path))

    def predict(self, cluster: int):
        pre_bar = tqdm(self.predict_loader)
        y_predict_pred_list = []
        y_predict_label_list = []
        mlflow.log_metric("dataset_id", self.dataset_id)
        mlflow.log_param("params", self.params)
        with torch.no_grad():
            for data in pre_bar:
                x_predict, y_predict, padding_mask = data
                x_predict, y_predict, padding_mask = x_predict.to(device), y_predict.to(device), padding_mask.to(device)
                y_predict_pred = self.model(x_predict, padding_mask)
                y_predict_pred_list.append(y_predict_pred)
                y_predict_label_list.append(y_predict)
                pre_bar.desc = "Predicting" 
            y_predict_pred_array = np.concatenate([tensor.cpu().numpy() for tensor in y_predict_pred_list])
            y_predict_label_array = np.concatenate([tensor.cpu().numpy() for tensor in y_predict_label_list])

            if self.is_cluster:
                output_data = pd.read_csv(f"../data/interim/dataset_{self.dataset_id}/output_index_cluster_{cluster}_data.csv")
                output_data['pre_order_prob'] = y_predict_pred_array
                output_data['target'] = y_predict_label_array
                predict_output_path = self.predict_output_path + "predict_output_" + self.model_type + f"_cluster_{cluster}" + ".csv"
            else:
                output_data = pd.read_csv(f"../data/interim/dataset_{self.dataset_id}/output_index_data.csv")
                output_data['pre_order_prob'] = y_predict_pred_array
                output_data['target'] = y_predict_label_array
                predict_output_path = self.predict_output_path + "predict_output_" + self.model_type + f"_cluster_None" + ".csv"
                output_data = self.evaluation_indicators_top(output_data)

            output_data.to_csv(predict_output_path, index=False)
            logging.info("----output file save as :{}".format(predict_output_path))

    def evaluation_indicators_without_cluster(self):
        data = pd.read_csv(f"../output/predict_file/dataset_{self.dataset_id}/predict_output_{self.model_type}_cluster_None.csv")
        predict_auc = roc_auc_score(data['target'], data['pre_order_prob'])
        logger.info("Auc:{} ".format(predict_auc))

        acc = accuracy_score(data['target'], data['decision'])
        logger.info("Acc:{} ".format(acc))

        precision = precision_score(data['target'], data['decision'])
        recall = recall_score(data['target'], data['decision'])
        f1 = f1_score(data['target'], data['decision'])

        logger.info("Precision:{} ".format(precision))
        logger.info("Recall:{} ".format(recall))
        logger.info("F1:{} ".format(f1))

        mlflow.log_metric("Auc", round(predict_auc, 4))
        mlflow.log_metric("Acc", round(acc, 4))
        mlflow.log_metric("Precision", round(precision, 4))
        mlflow.log_metric("Recall", round(recall, 4))
        mlflow.log_metric("f1", round(f1, 4))
        mlflow.log_metric("threshold", self.threshold_value)

    def evaluation_indicators_cluster(self, n_clusters: int):
        output_data = pd.DataFrame()
        for i in range(0, n_clusters):
            data = pd.read_csv(f"../output/predict_file/dataset_{self.dataset_id}/predict_output_{self.model_type}_cluster_{i}.csv")
            output_data = pd.concat([output_data, data], ignore_index=True, axis=0)
        output_data = self.evaluation_indicators_top(output_data)
        output_data.to_csv(f"../output/predict_file/dataset_{self.dataset_id}/output_{self.model_type}_all_clusters.csv")
        predict_auc = roc_auc_score(output_data['target'], output_data['pre_order_prob'])
        logger.info("Auc:{} ".format(predict_auc))

        acc = accuracy_score(output_data['target'], output_data['decision'])
        logger.info("Acc:{} ".format(acc))

        precision = precision_score(output_data['target'], output_data['decision'])
        recall = recall_score(output_data['target'], output_data['decision'])
        f1 = f1_score(output_data['target'], output_data['decision'])

        logger.info("Precision:{} ".format(precision))
        logger.info("Recall:{} ".format(recall))
        logger.info("F1:{} ".format(f1))

        mlflow.log_metric("Auc", round(predict_auc, 4))
        mlflow.log_metric("Acc", round(acc, 4))
        mlflow.log_metric("Precision", round(precision, 4))
        mlflow.log_metric("Recall", round(recall, 4))
        mlflow.log_metric("f1", round(f1, 4))
        mlflow.log_metric("top_N", self.top_value)
        mlflow.log_metric("n_clusters", self.n_clusters)
        mlflow.log_metric("d_model", self.params['d_model'])
        mlflow.log_metric("hidden_size", self.params['hidden_size'])
        mlflow.log_metric("seq_length", self.seq_length)
        mlflow.log_metric("is_data_enhancement", self.is_dataEnhancement) 

    def evaluation_indicators_top(self, data):
        df = data.sort_values(by="pre_order_prob", ascending=False).reset_index(drop=True)

        print("len:", len(df))
        top_num = int(len(df) * self.top_value)

        print(top_num)

        self.threshold_value = df.loc[top_num]['pre_order_prob']

        logger.info("threshold_value:{} ".format(self.threshold_value))

        data['decision'] = data['pre_order_prob'].apply(lambda x: 1 if x >= self.threshold_value else 0)

        return data

    def main(self):
        self.create_predict_index_dataset()
        self.connect_feature_engineer()
        self.transform()
        self._cluster_group(is_cluster=self.is_cluster)
        mlflow.set_experiment("Predict_test")
        if self.is_cluster is False:
            self.n_clusters = 1
            run_name = f"predict_{self.model_type}_all"
        else:
            run_name = f"predict_{self.model_type}_cluster"
        with mlflow.start_run(run_name=run_name):
            for cluster in range(0, self.n_clusters):
                self.data_preprocess(cluster=cluster, is_cluster=self.is_cluster)
                self.dataloader(cluster=cluster, is_cluster=self.is_cluster)
                self.load_model(option_model=self.model_type, cluster=cluster, is_cluster=self.is_cluster)
                self.predict(cluster)
                logger.info('-------Done!  You are very good!-------')
            if self.is_cluster:
                self.evaluation_indicators_cluster(self.n_clusters)
            else:
                self.evaluation_indicators_without_cluster()


if __name__ == '__main__': 
 
    pass
