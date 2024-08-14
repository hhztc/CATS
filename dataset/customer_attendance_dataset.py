import os
import sys
import logging

import pandas as pd

from .base_dataset import BaseDataSet
from datetime import timedelta, datetime
import src.config as config

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')
logging.root.setLevel(level=logging.INFO)


class CustomerAttendanceDataset(BaseDataSet):
    def __init__(self, dataset_id: int, running_dt_end: str, train_interval: int, file_type: str,  **param):
        super().__init__(param)
        logger.info('-----Loading data-----')
        self.sale_data = pd.read_csv("../data/raw/ADS_AI_MRT_SALES_SHIPPING.csv")
        self.org_data = pd.read_csv("../data/raw/ADS_AI_MRT_DIM_ORG_INV.csv")
        self.dataset_id = dataset_id
        self.running_dt_end = running_dt_end
        self.train_interval = train_interval
        self.file_type = file_type
        self.k = param.get('interval')
        self.end_date = (datetime.strptime(self.running_dt_end, "%Y-%m-%d")).strftime("%Y-%m-%d")
        self.start_date = (datetime.strptime(self.end_date, "%Y-%m-%d") - timedelta(days=self.train_interval)).strftime("%Y-%m-%d")
        logger.info('-----start_date: {}'.format(self.start_date))
        logger.info('-----end_date: {}'.format(self.end_date))
        self.data = pd.DataFrame()

    def _preprocessing_data(self):
        sale_data = self.sale_data.copy()
        org_data = self.org_data.copy()
        sale_data = pd.merge(sale_data, org_data, on='org_inv_dk', how='left')
        sale_data = sale_data[sale_data['l3_org_inv_nm'] == config.dataset_id[self.dataset_id]]
        # 仅包含[start_date, end_date]的发货数据
        sale_data = sale_data[sale_data['shipping_dt'] <= self.end_date]
        self.sale_data = sale_data.drop_duplicates(subset=['cust_dk', 'shipping_dt'], keep='first')[['cust_dk', 'shipping_dt']]
        self.sale_data['purchase_count'] = 1

    def _get_avg_interval_last_k_days(self, k: int):
        sale_data = self.sale_data.copy()
        sale_data['shipping_dt'] = pd.to_datetime(sale_data['shipping_dt'])

        # 生成每个日期×每个客户的笛卡尔积表，并关联发货记录。
        # 在发货记录中的行purchase_count为1,不在的为nan.填充后不在的为0.
        dates = pd.date_range(self.start_date, self.end_date)
        users = sale_data['cust_dk'].unique()
        index = pd.MultiIndex.from_product([dates, users], names=['shipping_dt', 'cust_dk'])
        full_index = pd.DataFrame(index=index).reset_index()
        result = pd.merge(full_index, sale_data, how='left')
        result = result.fillna(0)
        result['total_purchase_last_%d_days' % k] = result.groupby('cust_dk')['purchase_count'].apply(lambda x: x.rolling(k, 1).sum())
        result = result[result['total_purchase_last_%d_days' % k] != 0.0]

        result['avg_interval_last_%d_days' % k] = k / result['total_purchase_last_%d_days' % k]

        result.drop('total_purchase_last_%d_days' % k, axis=1, inplace=True)

        self.data = pd.concat([self.data, result]).reset_index(drop=True)

    def _post_processing_data(self):
        self.data.rename(columns={'shipping_dt': 'date'}, inplace=True)
        if self.data.isnull().any().any():
            logger.info("Warning: Null in customer_attendance.csv")
        self.file_name = "customer_attendance_last_" + str(self.k) + "_days." + self.file_type

    def build_dataset_all(self):
        logger.info("-----Preprocessing data----- ")
        self._preprocessing_data()
        logger.info("-----Calculating avg interval last {} days...".format(self.k))
        self._get_avg_interval_last_k_days(k=self.k)
        logger.info("-----Calculate avg interval last {} days done.".format(self.k))
        logger.info("-----Postprocessing data----- ")
        self._post_processing_data()
        self.data.to_csv(f"../data/feature_store/dataset_{self.dataset_id}/avg_purchase_data_7.csv")


if __name__ == "__main__":
    pass
