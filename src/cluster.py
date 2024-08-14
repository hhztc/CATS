import os
import ast
import sys
import logging
import pandas as pd

import config as config

from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from cluster_model.k_shape import k_shape


program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.root.setLevel(level=logging.INFO)


# 四级公司出勤序列模式编码器
class ListEncoder:
    def __init__(self):
        self.encoding_dicts = {}
        self.unique_index = 1

    def encode_list(self, input_list):
        input_str = ''.join(map(str, input_list))

        if input_str not in self.encoding_dicts:
            self.encoding_dicts[input_str] = self.unique_index
            self.unique_index += 1

        return self.encoding_dicts[input_str]


class ClusterModule(object):
    def __init__(self, args):
        self.shipping_data = pd.read_pickle("SHIPPING.pkl")
        self.org_data = pd.read_csv("ADS_AI_MRT_DIM_ORG_INV.csv")

        self.dataset_id = args.dataset_id

        self.shipping_data = pd.merge(self.shipping_data, self.org_data, on='org_inv_dk', how='left')

        self.shipping_data = self.shipping_data[self.shipping_data['l3_org_inv_nm'] == config.dataset_id[self.dataset_id]]

        self.end_date = '2023-9-30'

        self.n_clusters = args.n_clusters

        self.user_length = args.user_length

        self.cluster_data_id = args.cluster_data_id

        self.date_length = args.cluster_date_length

        self.start_date = (datetime.strptime(self.end_date, "%Y-%m-%d") - timedelta(days=self.date_length-1)).strftime("%Y-%m-%d")

        logging.info("start_date: {}".format(self.start_date))
        logging.info("end_date: {}".format(self.end_date))

        self.cluster_raw_data = pd.DataFrame()

        self.attendance_mode_data = pd.DataFrame()

        self.cluster_data = pd.DataFrame()

        self.user_l4_times_data = pd.DataFrame()

        logging.info("-----init done----")

    def create_cluster_dataset(self):

        temp_data = self.shipping_data.copy()
        temp_data['shipping_dt'] = pd.to_datetime(temp_data['shipping_dt'])

        date_index = pd.date_range(self.start_date, self.end_date, freq='D')
        date_data = pd.DataFrame(date_index, columns=['date'])

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

        # 作笛卡尔积
        cartesian_data = pd.merge(date_data.assign(key=1), cust_data.assign(key=1), on='key').drop('key', axis=1)
        cartesian_data = pd.merge(cartesian_data.assign(key=1), org_temp_data.assign(key=1), on='key').drop('key', axis=1)

        temp_data = temp_data[['cust_dk', 'shipping_dt', 'l4_org_inv_nm']].drop_duplicates()
        temp_data.reset_index()
        temp_data['is_come'] = 1
        cartesian_data = pd.merge(cartesian_data, temp_data, left_on=['date', 'cust_dk', 'l4_org_inv_nm'],
                                  right_on=['shipping_dt', 'cust_dk', 'l4_org_inv_nm'], how='left')

        cartesian_data['is_come'].fillna(0, inplace=True)

        cartesian_data = cartesian_data[['date', 'cust_dk', 'l4_org_inv_nm', 'is_come']]

        l4_org_data = config.l4_dict[self.dataset_id]

        cartesian_data = pd.merge(cartesian_data, l4_org_data, on='l4_org_inv_nm', how='left')

        cartesian_data = cartesian_data[['date', 'cust_dk', 'l4_org_inv_bk', 'is_come']]

        self.cluster_raw_data = cartesian_data

        return self.cluster_raw_data

    def create_user_l4_times(self):
        cluster_raw_data = self.cluster_raw_data.copy()
        temp_data = cluster_raw_data.groupby(['cust_dk', 'date']).agg({'is_come': 'sum'})
        temp_data = temp_data.groupby(['cust_dk'])['is_come'].apply(list).reset_index()
        temp_data = temp_data.rename(columns={'is_come': 'sequence_value'}, inplace=False)
        self.user_l4_times_data = temp_data

    def encode_attendance_sequence(self):

        cluster_raw_data = self.cluster_raw_data.copy()
        sequence_data = cluster_raw_data.groupby(['cust_dk', 'date'])['is_come'].apply(list).reset_index()

        # 创建 ListEncoder 实例
        list_encoder = ListEncoder()

        sequence_data['sequence_value'] = sequence_data['is_come'].apply(list_encoder.encode_list)

        sequence_data = sequence_data.groupby('cust_dk')['sequence_value'].apply(list).reset_index()

        self.attendance_mode_data = sequence_data

    def cluster(self, data: pd.DataFrame, column_name: str):
        logging.info('----聚类样本簇数： {}'.format(self.n_clusters))
        cluster_data = k_shape(data=data, column_name=column_name, n_clusters=self.n_clusters)
        self.cluster_data = cluster_data
        self.cluster_data.to_csv(f"../data/cluster_data/dataset_{self.dataset_id}/cluster_{self.n_clusters}_output_data.csv", index=False)

    def main(self):
        self.create_cluster_dataset()
        if self.cluster_data_id == 1:
            self.create_user_l4_times()
            self.cluster(self.user_l4_times_data, column_name="sequence_value")
        elif self.cluster_data_id == 2:
            self.encode_attendance_sequence()
            self.cluster(self.attendance_mode_data, column_name="sequence_value")









