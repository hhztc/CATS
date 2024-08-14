import os
import sys
import logging
import config as config
import pandas as pd

from dataset.context_feature import ContextFeatureDataSet
from dataset.customer_purchase_interval import CustomerPurchaseIntervalDataset
from dataset.customer_attendance_dataset import CustomerAttendanceDataset

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

logging.root.setLevel(level=logging.INFO)


class FeatureEngineerModule(object):
    def __init__(self, args):
        self.dataset_id = args.dataset_id
        logger.info("-----当前数据集id:{}".format(self.dataset_id))

    def main(self):
        logger.info("-----载入数据-----")
        ck = ContextFeatureDataSet(running_dt_end='2023-10-7', train_interval=96, file_type='csv', dataset_id=self.dataset_id)
        ck.build_dataset_all()

        param = {"interval": 7}
        att = CustomerAttendanceDataset(dataset_id=self.dataset_id, running_dt_end='2023-10-7', train_interval=96, file_type="csv", **param)
        att.build_dataset_all()

        att = CustomerPurchaseIntervalDataset(dataset_id=self.dataset_id, running_dt_end='2023-10-07', train_interval=96, file_type='csv')
        att.build_dataset_all()


if __name__ == '__main__':
    pass
