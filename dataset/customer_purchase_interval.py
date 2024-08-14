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


class CustomerPurchaseIntervalDataset(BaseDataSet):

    def __init__(self, dataset_id:int, running_dt_end: str, train_interval: int, file_type: str, **param):
        super().__init__(param)
        logger.info('-----Loading data-----')

        self.sale_data = pd.read_pickle("../data/raw/ADS_AI_MRT_SALES_SHIPPING.pkl")
        self.org_data = pd.read_csv("../data/raw/ADS_AI_MRT_DIM_ORG_INV.csv")
        self.dataset_id = dataset_id
        self.running_dt_end = running_dt_end
        self.train_interval = train_interval
        self.file_type = file_type

        self.end_date = (datetime.strptime(self.running_dt_end, "%Y-%m-%d")).strftime("%Y-%m-%d")
        self.start_date = (datetime.strptime(self.end_date, "%Y-%m-%d") - timedelta(days=self.train_interval)).strftime(
            "%Y-%m-%d")
        logger.info('-----start_date: {}'.format(self.start_date))
        logger.info('-----end_date: {}'.format(self.end_date))

        self.data = pd.DataFrame()
        self.file_name = None

    def _preprocessing_data(self):
        sale_data = self.sale_data.copy()
        org_data = self.org_data.copy()
        sale_data = pd.merge(sale_data, org_data, on='org_inv_dk', how='left')
        sale_data = sale_data[sale_data['l3_org_inv_nm'] == config.dataset_id[self.dataset_id]]

        # 仅包含[start_date, end_date]的发货数据
        sale_data = sale_data[(sale_data['shipping_dt'] >= self.start_date) &
                              (sale_data['shipping_dt'] <= self.end_date)]
        sale_data = sale_data.drop_duplicates(subset=['cust_dk', 'shipping_dt'], keep='first')[
            ['cust_dk', 'shipping_dt']]
        # 原始数据按发货日期和客户id排序，主要是保证每行的上一个发货日期就是上一次采购日期
        sale_data.sort_values(['shipping_dt', 'cust_dk'], inplace=True)
        self.sale_data = sale_data

    def _get_purchase_interval(self):
        sale_data = self.sale_data.copy()
        sale_data['shipping_dt'] = pd.to_datetime(sale_data['shipping_dt'])

        # 原始数据中发货日期下移一格就是上一次采购日期。第一次采购没有上一次，last_purchase_date为nan
        sale_data['last_purchase_date'] = sale_data.groupby('cust_dk')['shipping_dt'].shift(1)

        # 生成每个日期×每个客户的笛卡尔积表，并关联发货记录
        dates = pd.date_range(self.start_date, self.end_date)
        users = sale_data['cust_dk'].unique()
        index = pd.MultiIndex.from_product([dates, users], names=['shipping_dt', 'cust_dk'])
        full_index = pd.DataFrame(index=index).reset_index()
        result = pd.merge(full_index, sale_data, how='left')

        # 计算每个客户第一次和最后一次购买的日期，并关联到result中。关联后的字段为shipping_dt_first与shipping_dt_last
        first_purchase_date = sale_data.groupby('cust_dk')['shipping_dt'].min()
        last_purchase_date = sale_data.groupby('cust_dk')['shipping_dt'].max()
        result = result.merge(first_purchase_date, on="cust_dk", suffixes=('', '_first'), how='left')
        result = result.merge(last_purchase_date, on="cust_dk", suffixes=('', '_last'), how='left')

        # 在第一次购买的日期及之前的日期都没有“上一次”，直接去掉这些行
        result = result[(result['shipping_dt'] > result['shipping_dt_first'])]
        result = result.reset_index(drop=True)

        # bfill从后往前填充，能够填充[start_date, shipping_dt_last]之间的空值
        result['last_purchase_date'] = result.groupby('cust_dk')['last_purchase_date'].fillna(method='bfill')
        # 在(shipping_dt_last, end_date]之间的空值，都用shipping_dt_last来填充
        result['last_purchase_date'].fillna(result['shipping_dt_last'], inplace=True)

        # 距离上次间隔=当天日期-上次日期
        result['interval_from_last_purchase'] = result['shipping_dt'] - result['last_purchase_date']
        result['interval_from_last_purchase'] = result['interval_from_last_purchase'].apply(lambda x: x.days)

        result.drop(labels=['shipping_dt_first', 'shipping_dt_last'], axis=1, inplace=True)

        self.data = pd.concat([self.data, result]).reset_index(drop=True)

    def _post_processing_data(self):
        self.data.rename(columns={'shipping_dt': 'date'}, inplace=True)
        if self.data.isnull().any().any():
            logger.info("Warning: Null in customer_attendance.csv")
        self.file_name = "customer_interval_from_last_purchase." + self.file_type

    def build_dataset_all(self):
        logger.info("-----Preprocessing data----- ")
        self._preprocessing_data()
        logger.info("Calculating interval from last purchase...")
        self._get_purchase_interval()
        logger.info("-----Postprocessing data----- ")
        self._post_processing_data()
        self.data.to_csv(f"../data/feature_store/dataset_{self.dataset_id}/purchase_data.csv")


