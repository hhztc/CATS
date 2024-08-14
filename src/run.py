import os
import torch
import mlflow
import random
import argparse
import numpy as np
import config as config

from predict import Predict
from train import TrainModule
from feature_engineer_main import FeatureEngineerModule
from cluster import ClusterModule

from datetime import datetime

log_time = datetime.now().strftime("%m-%d-%H-%M-%S")
log_file = f'../log/{log_time}.txt'

# 检查日志文件是否存在，如果不存在则创建
if not os.path.isfile(log_file):
    open(log_file, 'w').close()

local_url = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(local_url)


if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    parser = argparse.ArgumentParser(description='gru')
    ## 训练命令参数
    parser.add_argument('--epochs', type=int, default=10, help='epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch_size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning_rate')
    parser.add_argument('--step_size', type=int, default=1, help='step_size')
    parser.add_argument('--gamma', type=float, default=0.25, help='gamma')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay')
    ## 聚类模块参数
    parser.add_argument('--is_cluster', type=bool, default=False, help='')
    parser.add_argument('--n_clusters', type=int, default=4, help='')
    parser.add_argument('--cluster_date_length', type=int, default=60, help='')
    parser.add_argument('--cluster_data_id', type=int, default=1, help='')
    ## 特征工程参数
    parser.add_argument('--is_createFeature', type=bool, default=True, help='')
    ## 数据处理参数
    parser.add_argument('--train_interval', type=int, default=60, help='')
    parser.add_argument('--seq_length', type=int, default=7, help='')
    parser.add_argument('--feature_size', type=int, default=5, help='')
    parser.add_argument('--user_length', type=int, default=60, help='')
    ## 预测参数
    parser.add_argument('--predict_length', type=int, default=1, help='')
    parser.add_argument('--top_N', type=float, default=0.301, help='')
    parser.add_argument('--predict_output_path', type=str, default="../output/predict_file/", help='预测输出路径')
    ## 其他
    parser.add_argument('--model_output_path', type=str, default="../output/model/", help='模型权重输出路径')
    parser.add_argument('--is_dataEnhancement', type=bool, default=False, help='')
    ## 部分模型超参
    parser.add_argument('--hidden_size', type=float, default=32, help='')
    parser.add_argument('--d_model', type=float, default=16, help='embedding_size')
    # parser.add_argument('--top_k', type=float, default=3, help='')
    ## 主命令
    parser.add_argument('--dataset_id', type=int, default=3, help='')
    parser.add_argument('--model_type', type=str, default='AttentionGRU', help='options:[AttentionGRU, gru, lstm, LstNet, TimesNet, DLinear, PatchTST, Transformer, Informer]')
    parser.add_argument('--task_id', type=int, default=1, help='[0: cluster, 1: train, 2:predict]任务选择')

    args = parser.parse_args()

    exp_feature = FeatureEngineerModule(args)
    exp_feature.main()
    exp_train = TrainModule(args)
    exp_train.main()
    exp_predict = Predict(args)
    exp_predict.main()

