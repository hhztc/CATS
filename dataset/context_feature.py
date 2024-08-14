"""
@author: 张天晨

@date: 2023-05-27
"""
import os
import sys
import logging
import pandas as pd
from .base_dataset import BaseDataSet
from datetime import timedelta, datetime

import holidays

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


class ContextFeatureDataSet(BaseDataSet):
    def __init__(self, running_dt_end: str, train_interval: int, file_type: str, dataset_id: int, **param):
        super().__init__(param)
        self.running_dt_end = running_dt_end
        self.train_interval = train_interval
        self.file_type = file_type
        self.dataset_id = dataset_id
        self.end_date = (datetime.strptime(self.running_dt_end, "%Y-%m-%d") + timedelta(days=7)).strftime("%Y-%m-%d")
        self.start_date = (
                    datetime.strptime(self.running_dt_end, "%Y-%m-%d") - timedelta(days=self.train_interval)).strftime(
            "%Y-%m-%d")
        logger.info('-----start_date: {}'.format(self.start_date))
        logger.info('-----end_date: {}'.format(self.end_date))

        self.file_name = None
        self.data = pd.DataFrame()

    def _check_data(self):
        pass

    def _preprocessing_data(self):
        pass

    def _get_context_feature(self):
        date_index = pd.date_range(self.start_date, self.end_date, freq='D')
        date_data = pd.DataFrame(date_index, columns=['date'])
        date_data['month_of_year'] = date_data['date'].dt.month  # 该年的第几个月
        date_data['quarter_of_year'] = date_data['date'].dt.quarter  # 该年的第几个季节
        date_data['day_of_month'] = date_data['date'].dt.day  # 该月的第几天
        date_data['day_of_week'] = date_data['date'].dt.dayofweek  # 该周的第几天
        date_data['week_of_year'] = date_data['date'].dt.isocalendar().week  # 该年的第几个周
        date_data['is_weekday'] = (date_data['date'].dt.dayofweek >= 5).astype(int)  # 是否为周末
        date_data['is_holiday'] = (date_data['date'].isin(holidays.China(years=date_data['date'].dt.year))).astype(int)  # 是否节假日
        self.data = date_data

    def _post_processing_data(self):
        self.file_name = "context_feature." + self.file_type

    def build_dataset_all(self):
        logger.info("-----Preprocessing data----- ")
        self._preprocessing_data()
        logger.info("-----Generation context_feature----- ")
        self._get_context_feature()
        logger.info("-----Postprocessing data----- ")
        self._post_processing_data()
        self.data.to_csv(f"../data/feature_store/dataset_{self.dataset_id}/context.csv")


if __name__ == '__main__':
    pass
