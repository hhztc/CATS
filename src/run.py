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
    parser.add_argument('--is_cluster', type=bool, default=False, help='是否加入聚类模块')
    parser.add_argument('--n_clusters', type=int, default=4, help='聚类簇数量')
    parser.add_argument('--cluster_date_length', type=int, default=60, help='聚类样本时间长度')
    parser.add_argument('--cluster_data_id', type=int, default=2, help='option:[1: user_l4_times, 2: attendance_mode]')
    ## 特征工程参数
    parser.add_argument('--is_createFeature', type=bool, default=True, help='是否启动特征工程')
    ## 数据处理参数
    parser.add_argument('--train_interval', type=int, default=60, help='训练集天数')
    parser.add_argument('--seq_length', type=int, default=7, help='时间序列切片长度')
    parser.add_argument('--feature_size', type=int, default=5, help='特征维度')
    parser.add_argument('--user_length', type=int, default=60, help='活跃用户集采样天数')
    ## 预测参数
    parser.add_argument('--predict_length', type=int, default=1, help='预测天数')
    # parser.add_argument('--predict_interval', type=int, default=1, help='预测的间隔，等同于算法切片长度')
    # parser.add_argument('--threshold', type=float, default=0.7, help='设置固定阈值')
    parser.add_argument('--top_N', type=float, default=0.301, help='设置top_N阈值')
    parser.add_argument('--predict_output_path', type=str, default="../output/predict_file/", help='预测输出路径')
    ## 其他
    parser.add_argument('--model_output_path', type=str, default="../output/model/", help='模型权重输出路径')
    parser.add_argument('--is_dataEnhancement', type=bool, default=False, help='是否启动数据增强')
    ## 部分模型超参
    parser.add_argument('--hidden_size', type=float, default=32, help='隐藏层个数')
    parser.add_argument('--d_model', type=float, default=16, help='embedding_size')
    # parser.add_argument('--top_k', type=float, default=3, help='')
    ## 主命令
    parser.add_argument('--dataset_id', type=int, default=3, help='[1:广东, 2:江苏, 3:浙江, 4:福建, 5:广西]')
    parser.add_argument('--model_type', type=str, default='MICN', help='options:[AttentionGRU, SegRnn, gru, iGru, lstm, LstNet, TimesNet, DLinear, PatchTST, Transformer, Informer, iTransformer]')
    parser.add_argument('--task_id', type=int, default=1, help='[0: cluster, 1: train, 2:predict]任务选择')

    args = parser.parse_args()

    # exp_cluster = ClusterModule(args)
    # exp_cluster.main()
    exp_feature = FeatureEngineerModule(args)
    exp_feature.main()
    exp_train = TrainModule(args)
    exp_train.main()
    exp_predict = Predict(args)
    exp_predict.main()
    #
    # for d in [16, 32, 48, 64]:
    #     for h in [16, 32, 48, 64]:
    #         args.d_model = d
    #         args.hidden_size = h
    #         if args.model_type == "AttentionGRU":
    #             args.is_dataEnhancement = True
    #             args.is_cluster = True
    #         else:
    #             args.is_dataEnhancement = False
    #             args.is_cluster = False
    # for d in [16, 32, 48, 64]:
    #     for h in [16, 32, 48, 64]:
    #         exp_train = TrainModule(args)
    #         exp_train.main()
    #         exp_predict = Predict(args)
    #         exp_predict.main()
    """
    # 超参实验

    ## 聚类样本数----------------------------------------
    for ids in range(1, 6):
        args.dataset_id = 3

    ## stage1: 聚类
    for n_cluster in range(2, 6):
        args.n_clusters = n_cluster
        exp_cluster = ClusterModule(args)
        exp_cluster.main()

    ## stage2: 特征生成
    exp_feature = FeatureEngineerModule(args)
    exp_feature.main()

    ## stage3: 训练预测
    for n_cluster in range(2, 6):
        args.n_clusters = n_cluster
        exp_train = TrainModule(args)
        exp_train.main()
        exp_predict = Predict(args)
        exp_predict.main()
    """
    ## Embedding实验--------------------------------------
    # for i in range(1, 6):
    #     args.dataset_id = i
    #     ### stage1: 聚类-每个数据集固定一个最好聚类簇数
    #     if i == 1:
    #         args.n_clusters = 2
    #     elif i == 2:
    #         args.n_clusters = 3
    #     elif i == 3:
    #         args.n_clusters = 2
    #     elif i == 4:
    #         args.n_clusters = 5
    #     elif i == 5:
    #         args.n_clusters = 2
    #
    #     ## stage2: 训练
    #     # args.n_clusters = 2
    #     for d_model in [64, 48, 32, 16, 8]:
    #         # args.hidden_size = hidden_size
    #         args.d_model = d_model
    #         exp_train = TrainModule(args)
    #         exp_train.main()
    #         exp_predict = Predict(args)
    #         exp_predict.main()

    """
    ## 序列切片长度实验
    for i in range(1, 6):
        args.dataset_id = i
        ### stage1: 聚类-每个数据集固定一个最好聚类簇数
        if i == 1:
            args.n_clusters = 2
            args.d_model = 32
        elif i == 2:
            args.n_clusters = 3
            args.d_model = 16
        elif i == 3:
            args.n_clusters = 4
            args.d_model = 16
        elif i == 4:
            args.n_clusters = 5
            args.d_model = 16
        elif i == 5:
            args.n_clusters = 2
            args.d_model = 32

        ### stage2: 训练
        for seq_length in [35, 28, 21, 14, 7]:
            args.seq_length = seq_length
            # exp_feature = FeatureEngineerModule(args)
            # exp_feature.main()
            exp_train = TrainModule(args)
            exp_train.main()
            exp_predict = Predict(args)
            exp_predict.main()
 """
    #  消融实验
    # for i in range(1, 6):
    #     args.dataset_id = i
    #     ### stage1: 聚类-每个数据集固定最好超参
    #     if i == 1:
    #         args.n_clusters = 2
    #         args.d_model = 32
    #     elif i == 2:
    #         args.n_clusters = 3
    #         args.d_model = 16
    #     elif i == 3:
    #         args.n_clusters = 4
    #         args.d_model = 16
    #     elif i == 4:
    #         args.n_clusters = 5
    #         args.d_model = 16
    #     elif i == 5:
    #         args.n_clusters = 2
    #         args.d_model = 32
    #
    #     args.is_cluster = True
    #     args.is_dataEnhancement = True
    #     exp_train = TrainModule(args)
    #     exp_train.main()
    #     exp_predict = Predict(args)
    #     exp_predict.main()

    # if args.task_id == 0:
    #     exp_cluster = ClusterModule(args)
    #     exp_cluster.main()
    # elif args.task_id == 1:
    #     exp_train = TrainModule(args)
    #     exp_train.main()
    # elif args.task_id == 2:
    #     exp_predict = Predict(args)
    #     exp_predict.main()
    # for ids in range(1, 6):
    #     args.dataset_id = ids
    #     exp_feature = FeatureEngineerModule(args)
    #     exp_feature.main()
    #     for model_type in ['iGru', 'PatchTST', 'iTransformer', 'DLinear', 'TimesNet', 'Informer', 'LstNet', 'Transformer', 'gru', 'lstm']:
    #         args.model_type = model_type
    #         if args.model_type == "iGru":
    #             args.is_cluster = True
    #             args.is_dataEnhancement = True
    #             exp_cluster = ClusterModule(args)
    #             exp_cluster.main()
    #             exp_train = TrainModule(args)
    #             exp_train.main()
    #         else:
    #             args.is_cluster = False
    #             args.is_dataEnhancement = False
    #
    #         exp_predict = Predict(args)
    #         exp_predict.main()
    # #
    #
    # exp_feature = FeatureEngineerModule(args)
    # exp_feature.main()
    # exp_cluster = ClusterModule(args)
    # exp_cluster.main()
    # exp_train = TrainModule(args)
    # exp_train.main()
    # exp_predict = Predict(args)
    # exp_predict.main()
    # #
    # for model_type in ['iGru', 'PatchTST', 'iTransformer', 'DLinear', 'TimesNet', 'Informer', 'LstNet', 'Transformer', 'gru', 'lstm']:
    #     args.model_type = model_type
    #     if args.model_type == "iGru":
    #         args.is_cluster = True
    #     else:
    #         args.is_cluster = False
    #     exp_predict = Predict(args)
    #     exp_predict.main()

    # for ids in range(1, 6):
    #     args.dataset_id = ids
    #     exp_feature = FeatureEngineerModule(args)
    #     exp_feature.main()
    #     exp_cluster = ClusterModule(args)
    #     exp_cluster.main()
    #     exp_train = TrainModule(args)
    #     exp_train.main()
    #     exp_predict = Predict(args)
    #     exp_predict.main()


