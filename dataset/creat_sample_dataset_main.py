import os
import sys
import pickle
import logging
import pandas as pd
import src.config as config

from datetime import datetime, timedelta

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.root.setLevel(level=logging.INFO)


class ListEncoder(object):
    def __init__(self, dataset_id):
        self.encoding_dicts = {}
        self.unique_index = 1
        self.dataset_id = dataset_id

    # 编码器
    def encode_list(self, input_list: str):
        input_str = ''.join(map(str, input_list))

        if input_str not in self.encoding_dicts:
            self.encoding_dicts[input_str] = self.unique_index
            self.unique_index += 1

        return self.encoding_dicts[input_str]

    def dump_dict(self):
        # 将字典保存到文件中
        with open(f'../data/interim/dataset_{self.dataset_id}/l4_mode_dict.pkl', 'wb') as file:
            pickle.dump(self.encoding_dicts, file)

    def load_dict(self):
        with open(f'../data/interim/dataset_{self.dataset_id}/l4_mode_dict.pkl', 'rb') as file:
            self.encoding_dicts = pickle.load(file)

    def decode_list(self, input_list):
        # 解码器
        input_str = ''.join(map(str, input_list))
        if input_str not in self.encoding_dicts:
            self.encoding_dicts[input_str] = 0

        return self.encoding_dicts[input_str]


class TrainSampleDataset(object):
    def __init__(self, dataset_id: int, user_length: int, train_interval: int):
        self.shipping_data = pd.read_pickle("../data/raw/ADS_AI_MRT_SALES_SHIPPING.pkl")  # 发货表
        self.org_data = pd.read_csv("../data/raw/ADS_AI_MRT_DIM_ORG_INV.csv")  # 生产组织对照表

        self.shipping_data = pd.merge(self.shipping_data, self.org_data, on='org_inv_dk', how='left')

        self.shipping_data = self.shipping_data[self.shipping_data['l3_org_inv_nm'] == config.dataset_id[dataset_id]]

        self.end_date = '2023-9-30'

        self.dataset_id = dataset_id

        self.date_length = train_interval

        self.user_length = user_length

        self.label_data = pd.DataFrame()  # 真实标签数据集

        self.raw_data = pd.DataFrame()  # 原始数据集

        self.data = pd.DataFrame()  # 最终数据集

        self.transform_data = pd.DataFrame()  # transform处理后数据集

    def _creat_label_dataset(self):
        shipping_data = self.shipping_data.copy()

        start_date = (datetime.strptime(self.end_date, "%Y-%m-%d") - timedelta(days=self.date_length - 1)).strftime("%Y-%m-%d")

        logging.info("start_date: {}".format(start_date))
        logging.info("end_date: {}".format(self.end_date))

        shipping_data = shipping_data[(shipping_data['shipping_dt'] >= start_date) & (shipping_data['shipping_dt'] <= self.end_date)]

        shipping_data = shipping_data[['shipping_dt', 'cust_dk', 'l4_org_inv_nm']].drop_duplicates()
        shipping_data.reset_index()

        shipping_data['label'] = 1

        shipping_data.rename(columns={'shipping_dt': 'date'}, inplace=True)

        # shipping_data.to_csv("../data/interim/train_label_data.csv")

        self.label_data = shipping_data

    def _create_index_sample_dataset(self):
        start_date = (datetime.strptime(self.end_date, "%Y-%m-%d") - timedelta(days=self.date_length-1)).strftime(
            "%Y-%m-%d")

        logging.info("start_date: {}".format(start_date))
        logging.info("end_date: {}".format(self.end_date))

        # 构造日期集
        date_index = pd.date_range(start_date, self.end_date, freq='D')
        date_data = pd.DataFrame(date_index, columns=['date'])

        # 构造用户集
        cust_data = self.shipping_data.copy()
        user_end_date = '2023-9-30'
        user_start_date = (datetime.strptime(user_end_date, "%Y-%m-%d") - timedelta(days=self.user_length)).strftime("%Y-%m-%d")
        cust_data = cust_data[(cust_data['shipping_dt'] >= user_start_date) & (cust_data['shipping_dt'] <= user_end_date)]
        cust_data = cust_data[['cust_dk']].drop_duplicates()
        cust_data.reset_index()
        num_cust = len(cust_data)
        logging.info('-----客户集时间间隔为：{}'.format(self.user_length))
        logging.info('-----客户数为：{}'.format(num_cust))

        org_temp_data = config.l4_dict[self.dataset_id]
        org_temp_data = org_temp_data[['l4_org_inv_nm']]
        # 构造样本集
        index_sample_data = pd.merge(date_data.assign(key=1), cust_data.assign(key=1), on='key').drop('key', axis=1)
        index_sample_data = pd.merge(index_sample_data.assign(key=1), org_temp_data.assign(key=1), on='key').drop('key', axis=1)

        index_sample_data.to_csv("../data/interim/index_sample_data.csv")

        self.index_sample_data = index_sample_data

    def _create_dataset(self):
        self.index_sample_data['date'] = pd.to_datetime(self.index_sample_data['date'])
        self.label_data['date'] = pd.to_datetime(self.label_data['date'])

        raw_data = pd.merge(self.index_sample_data, self.label_data, on=['date', 'cust_dk', 'l4_org_inv_nm'], how='left')

        l4_org_data = config.l4_dict[self.dataset_id]

        raw_data = pd.merge(raw_data, l4_org_data, on='l4_org_inv_nm', how='left')

        raw_data = raw_data[['date', 'cust_dk', 'l4_org_inv_nm', 'l4_org_inv_bk', 'label']]

        raw_data['label'].fillna(0, inplace=True)

        counts = raw_data['label'].value_counts()
        logger.info("---正样本：{}".format(counts[1]))
        logger.info("---负样本：{}".format(counts[0]))
        logger.info("---总样本量：{}".format(len(raw_data)))
        ratio = counts[0] / counts[1]
        logger.info("---正负样本比例为 1：{}".format(ratio))
        # raw_data.to_csv("../data/interim/train_index_data.csv")

        self.data = raw_data

    def main(self):
        logger.info("-----构建标签集-----")
        self._creat_label_dataset()
        logger.info("-----构建样本集-----")
        self._create_index_sample_dataset()
        logger.info("-----样本集关联真实标签生成原始数据集-----")
        self._create_dataset()
        logger.info("-----Done-----")
        return self.data


class PredictSampleDataset(object):
    def __init__(self, **params):
        self.shipping_data = pd.read_pickle("../data/raw/ADS_AI_MRT_SALES_SHIPPING.pkl")  # 发货表
        self.org_data = pd.read_csv("../data/raw/ADS_AI_MRT_DIM_ORG_INV.csv")  # 生产组织对照表

        self.shipping_data = pd.merge(self.shipping_data, self.org_data, on='org_inv_dk', how='left')

        self.predict_date = params.get("predict_date", '2023-10-01')  # 预测起始日期

        self.predict_length = params.get("predict_length", 1)  # 预测天数

        self.predict_interval = params.get("predict_interval", 7)  # 预测切片长度

        self.user_length = params.get("user_length", 60)

        self.dataset_id = params.get("dataset_id", 1)

        self.shipping_data = self.shipping_data[self.shipping_data['l3_org_inv_nm'] == config.dataset_id[self.dataset_id]]

        self.label_data = pd.DataFrame()  # 真实标签数据集

        self.raw_data = pd.DataFrame()  # 原始数据集

        self.cust_data = pd.DataFrame()  # 用户数据集

        self.data = pd.DataFrame()  # 最终数据集

        self.output_data = pd.DataFrame()  # 预测输出格式文件（用于存放模型输出）

        self.transform_data = pd.DataFrame()  # transform处理后数据集

        logging.info("-----init done----")

    def _creat_label_dataset(self):
        shipping_data = self.shipping_data.copy()

        start_date = (datetime.strptime(self.predict_date, "%Y-%m-%d") - timedelta(days=self.predict_interval)).strftime("%Y-%m-%d")
        end_date = (datetime.strptime(self.predict_date, "%Y-%m-%d") + timedelta(days=self.predict_length-1)).strftime("%Y-%m-%d")

        logging.info("start_date: {}".format(start_date))
        logging.info("end_date: {}".format(end_date))

        shipping_data = shipping_data[(shipping_data['shipping_dt'] >= start_date) & (shipping_data['shipping_dt'] <= end_date)]

        shipping_data = shipping_data[['shipping_dt', 'cust_dk', 'l4_org_inv_nm']].drop_duplicates()
        shipping_data.reset_index()

        shipping_data['label'] = 1

        shipping_data.rename(columns={'shipping_dt': 'date'}, inplace=True)

        # shipping_data.to_csv("../data/interim/predict_label_data.csv")

        self.label_data = shipping_data

    def _create_index_sample_dataset(self):
        start_date = (datetime.strptime(self.predict_date, "%Y-%m-%d") - timedelta(days=self.predict_interval)).strftime("%Y-%m-%d")
        end_date = (datetime.strptime(self.predict_date, "%Y-%m-%d") + timedelta(days=self.predict_length-1)).strftime("%Y-%m-%d")

        logging.info("start_date: {}".format(start_date))
        logging.info("end_date: {}".format(end_date))

        # 构造日期集
        date_index = pd.date_range(start_date, end_date, freq='D')
        date_data = pd.DataFrame(date_index, columns=['date'])

        # 构造用户集
        cust_data = self.shipping_data.copy()
        user_end_date = '2023-9-30'
        user_start_date = (datetime.strptime(user_end_date, "%Y-%m-%d") - timedelta(days=self.user_length)).strftime("%Y-%m-%d")
        cust_data = cust_data[(cust_data['shipping_dt'] >= user_start_date) & (cust_data['shipping_dt'] <= user_end_date)]
        cust_data = cust_data[['cust_dk']].drop_duplicates()
        cust_data.reset_index()
        self.cust_data = cust_data
        num_cust = len(cust_data)
        logging.info('-----客户集时间间隔为：{}'.format(self.user_length))
        logging.info('-----客户数为：{}'.format(num_cust))

        org_temp_data = config.l4_dict[self.dataset_id]
        org_temp_data = org_temp_data[['l4_org_inv_nm']]

        # 构造样本集
        index_sample_data = pd.merge(date_data.assign(key=1), cust_data.assign(key=1), on='key').drop('key', axis=1)
        index_sample_data = pd.merge(index_sample_data.assign(key=1), org_temp_data.assign(key=1), on='key').drop('key', axis=1)

        # index_sample_data.to_csv("../data/interim/predict_index_sample_data.csv")

        self.index_sample_data = index_sample_data

    def _create_dataset(self):
        self.index_sample_data['date'] = pd.to_datetime(self.index_sample_data['date'])
        self.label_data['date'] = pd.to_datetime(self.label_data['date'])

        raw_data = pd.merge(self.index_sample_data, self.label_data, on=['date', 'cust_dk', 'l4_org_inv_nm'], how='left')

        l4_org_data = config.l4_dict[self.dataset_id]

        raw_data = pd.merge(raw_data, l4_org_data, on='l4_org_inv_nm', how='left')

        raw_data = raw_data[['date', 'cust_dk', 'l4_org_inv_bk', 'label']]

        raw_data['label'].fillna(0, inplace=True)

        counts = raw_data['label'].value_counts()
        logger.info("---正样本：{}".format(counts[1]))
        logger.info("---负样本：{}".format(counts[0]))
        logger.info("---总样本量：{}".format(len(raw_data)))
        ratio = counts[0] / counts[1]
        logger.info("---正负样本比例为 1：{}".format(ratio))
        self.data = raw_data
        # self.data.to_csv("../data/interim/predict_index_data.csv")

    def _create_output_predict(self):
        start_date = self.predict_date
        end_date = (datetime.strptime(self.predict_date, "%Y-%m-%d") + timedelta(days=self.predict_length - 1)).strftime("%Y-%m-%d")

        logging.info("start_date: {}".format(start_date))
        logging.info("end_date: {}".format(end_date))

        # 构造日期集
        date_index = pd.date_range(start_date, end_date, freq='D')
        date_data = pd.DataFrame(date_index, columns=['date'])

        # 构造用户集
        output_data = pd.merge(date_data.assign(key=1), self.cust_data.assign(key=1), on='key').drop('key', axis=1)
        self.output_data = output_data[['date', 'cust_dk']]
        self.output_data.to_csv(f"../data/interim/dataset_{self.dataset_id}/output_index_data.csv")

    def main(self):
        logger.info("-----构建标签集-----")
        self._creat_label_dataset()
        logger.info("-----构建样本集-----")
        self._create_index_sample_dataset()
        logger.info("-----样本集关联真实标签生成原始数据集-----")
        self._create_dataset()
        logger.info("-----构建预测输出文件-----")
        self._create_output_predict()
        logger.info("-----Done-----")
        return self.data, self.output_data


class AutoEncodeMode(object):
    def __init__(self, dataset_id):
        self.dataset_id = dataset_id

        self.sequence_train_data = pd.DataFrame()
        self.sequence_predict_data = pd.DataFrame()

    def encode_mode_train_data(self, data: pd.DataFrame):
        # 统计每名客户每天去各四级公司的情况
        # eg:
        # user1 ：【0，1，0，0，0，0，0，0，1，1，0，0】 则当天去到第2，10，11家 3家四级公司

        sequence_train_data = data.groupby(['cust_dk', 'date'])['label'].apply(list).reset_index()

        # 对四级公司序列进行编码
        # eg:
        # 【0，1，0，0，0，0，0，0，1，1，0，0】 编码为 1 ，代表一种出勤模式
        # 【0，0，0，0，0，0，0，0，0，1，0，0】 编码为 2 ，代表一种出勤模式
        # 创建 ListEncoder 实例
        list_encoder = ListEncoder(self.dataset_id)

        # 应用编码器并新增一列编码值
        sequence_train_data['attendance_mode'] = sequence_train_data['label'].apply(list_encoder.encode_list)

        # sequence_train_data.to_csv("../data/interim/sequence_train_data.csv")

        list_encoder.dump_dict()

        # 删除原列名
        sequence_train_data = sequence_train_data.drop(columns=['label'])

        sequence_train_data['target'] = sequence_train_data['attendance_mode'].apply(lambda x: 1 if x > 1 else 0)

        sequence_train_data = sequence_train_data[['date', 'cust_dk', 'attendance_mode', 'target']]

        self.sequence_train_data = sequence_train_data

        return self.sequence_train_data

    def encode_mode_predict_data(self, data: pd.DataFrame):

        sequence_predict_data = data.groupby(['cust_dk', 'date'])['label'].apply(list).reset_index()

        list_encoder = ListEncoder(self.dataset_id)

        list_encoder.load_dict()

        sequence_predict_data['attendance_mode'] = sequence_predict_data['label'].apply(list_encoder.decode_list)

        # sequence_predict_data.to_csv("../data/interim/sequence_predict_data.csv")

        sequence_predict_data = sequence_predict_data.drop(columns=['label'])

        sequence_predict_data['target'] = sequence_predict_data['attendance_mode'].apply(lambda x: 1 if x > 1 else 0)

        sequence_predict_data = sequence_predict_data[['date', 'cust_dk', 'attendance_mode', 'target']]

        self.sequence_predict_data = sequence_predict_data

        return self.sequence_predict_data


if __name__ == '__main__':
    # obj = TrainSampleDataset()
    # obj.main()
    # obj = PredictSampleDataset()
    # obj.main()
    # obj = ListEncoder()
    # obj.encode_list(input_list="[1, 0, 0, 1]")
    # with open("../data/interim/l4_mode_encode.pickle", "w+") as dump_file:
    #     dump_file.write(obj.to_json())
    #
    # with open("../data/interim/l4_mode_encode.json", "r+") as dump_file:
    #     transformer = ListEncoder.from_json(dump_file.read())
    # print(transformer)
    #
    # x = transformer.decode_list(input_list="[1, 0, 0, 1]")
    # print(x)
    pass
