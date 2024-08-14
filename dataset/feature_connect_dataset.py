import os
import sys
import logging
import pandas as pd

import src.config as config


base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)


class FeatureConnect(object):
    def __init__(self, dataset_id: int, index_data: pd.DataFrame):
        self.index_data = index_data
        self.dataset_id = dataset_id
        self.context_data = pd.read_csv(f"../data/feature_store/dataset_{self.dataset_id}/context.csv")
        self.purchase_interval_data = pd.read_csv(f"../data/feature_store/dataset_{self.dataset_id}/purchase_data.csv")
        self.avg_purchase_interval_data = pd.read_csv(f"../data/feature_store/dataset_{self.dataset_id}/avg_purchase_data_7.csv")

        self.context_data['date'] = pd.to_datetime(self.context_data['date'])
        self.purchase_interval_data['date'] = pd.to_datetime(self.purchase_interval_data['date'])
        self.avg_purchase_interval_data['date'] = pd.to_datetime(self.avg_purchase_interval_data['date'])

        self.index_columns = ['date', 'cust_dk', 'attendance_mode']

        self.feature_columns = ['month_of_year', 'quarter_of_year', 'day_of_week', 'is_weekday', 'week_of_year',
                                'day_of_month', 'is_holiday', 'interval_from_last_purchase', 'avg_interval_last_7_days']

        self.label = ['target']

        self.output_data = pd.DataFrame()

        self.data = pd.DataFrame()

    def connect_context_feature(self):
        index_data = self.index_data.copy()
        data = self.context_data.copy()

        output_data = pd.merge(index_data, data, on='date', how='left')
        self.index_data = output_data

    def connect_purchase_interval_feature(self):
        index_data = self.index_data.copy()
        purchase_data = self.purchase_interval_data.copy()
        avg_purchase_data = self.avg_purchase_interval_data.copy()

        purchase_data = purchase_data[['date', 'cust_dk', 'interval_from_last_purchase']]
        avg_purchase_data = avg_purchase_data[['date', 'cust_dk', 'avg_interval_last_7_days']]

        index_data = pd.merge(index_data, purchase_data, on=['date', 'cust_dk'], how='left')
        index_data = pd.merge(index_data, avg_purchase_data, on=['date', 'cust_dk'], how='left')

        self.index_data = index_data

    def _post_processing_data(self):
        index_data = self.index_data.copy()
        index_data = index_data[self.index_columns + self.feature_columns + self.label]
        null_counts = index_data.isnull().sum()
        logger.info("-----null value as follows: {}".format(null_counts))
        logger.info("-----Processed Dataset Size：{}".format(len(index_data)))
        index_data.fillna(0, inplace=True)
        self.data = index_data

    def main(self):
        logger.info("-----Connect context feature")
        self.connect_context_feature()
        logger.info("-----Connect purchase interval feature")
        self.connect_purchase_interval_feature()
        logger.info("-----Post process")
        self._post_processing_data()
        logger.info("-----Done----- ")
        return self.data


if __name__ == '__main__':
    pass




