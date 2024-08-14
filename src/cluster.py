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
        self.shipping_data = pd.read_pickle("../data/raw/ADS_AI_MRT_SALES_SHIPPING.pkl")  #
        self.org_data = pd.read_csv("../data/raw/ADS_AI_MRT_DIM_ORG_INV.csv")  # 

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

        self.cluster_raw_data = pd.DataFrame()  # 聚类原始样本

        self.attendance_mode_data = pd.DataFrame()  # 出勤模式样本

        self.cluster_data = pd.DataFrame()  # 聚类最终输出样本

        self.user_l4_times_data = pd.DataFrame()  # 客户-四级公司个数统计样本

        logging.info("-----init done----")

    def create_cluster_dataset(self):
        """
        构建聚类样本
        格式：
        date   cust_dk   attendance_sequence
        步骤：
        1. 构建 时间集 和 客户集 并作笛卡尔积
            生成格式：
                date cust_dk l4_org_inv_nm
        2. 关联 四级公司
            生成格式：
                date cust_ck l4_org_inv_bk  is_come
        3. 生成近n天客户四级公司序列
            生成格式：
                date cust_dk l4_list
        """
        temp_data = self.shipping_data.copy()
        temp_data['shipping_dt'] = pd.to_datetime(temp_data['shipping_dt'])

        # 生成 时间集、客户集、四级公司集 作 笛卡尔积
        #  时间集
        date_index = pd.date_range(self.start_date, self.end_date, freq='D')
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

        # 作笛卡尔积
        cartesian_data = pd.merge(date_data.assign(key=1), cust_data.assign(key=1), on='key').drop('key', axis=1)
        cartesian_data = pd.merge(cartesian_data.assign(key=1), org_temp_data.assign(key=1), on='key').drop('key', axis=1)

        # 关联真实集（关联成功为1，未关联到的填充为0）
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

        # cartesian_data.to_csv(f"../data/cluster_data/客户出勤样本统计-{self.date_length}天-stage1.csv")

        self.cluster_raw_data = cartesian_data

        return self.cluster_raw_data

    def create_user_l4_times(self):
        cluster_raw_data = self.cluster_raw_data.copy()
        temp_data = cluster_raw_data.groupby(['cust_dk', 'date']).agg({'is_come': 'sum'})
        temp_data = temp_data.groupby(['cust_dk'])['is_come'].apply(list).reset_index()
        temp_data = temp_data.rename(columns={'is_come': 'sequence_value'}, inplace=False)
        # temp_data.to_csv(f"../data/cluster_data/客户-四级公司出勤个数统计_{self.date_length}天.csv")
        self.user_l4_times_data = temp_data

    def encode_attendance_sequence(self):
        """
        生成客户-四级公司出勤序列样本
        1. 生成序列样本
            格式：
                    date        cust_dk         attendance_sequence
            eg:     2023/9/1       1    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0] # 1所在的列表位置i代码当天去过第i家公司
        2. 对序列进行编码
            格式：
                    date        cust_dk         attendance_sequence                 sequence_value
            eg:     2023/9/1       1    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]             1    # 编码值1代表一种出勤模式

        3. 对编码值按客户聚合， 生成客户近n天的出勤序列
            格式：
                    cust_dk             sequence_mode_list
            eg:       1         [1, 5, 9, 6, 1, 1, 1, 5, 5, 6, 9, …………]  # 序列长度代表n天 得到客户n天的出勤模式序列
        """

        cluster_raw_data = self.cluster_raw_data.copy()
        sequence_data = cluster_raw_data.groupby(['cust_dk', 'date'])['is_come'].apply(list).reset_index()

        # 对四级公司序列进行编码
        # eg:
        # 【0，1，0，0，0，0，0，0，1，1，0，0】 编码为 1 ，代表一种出勤模式
        # 【0，0，0，0，0，0，0，0，0，1，0，0】 编码为 2 ，代表一种出勤模式
        # 创建 ListEncoder 实例
        list_encoder = ListEncoder()

        # 应用编码器并新增一列编码值
        sequence_data['sequence_value'] = sequence_data['is_come'].apply(list_encoder.encode_list)

        # sequence_data.to_csv("../data/cluster_data/用户-四级公司出勤序列-stage2.csv")

        # 得到每名客户近n天的去四级公司的模式序列
        # eg:
        #  user1: [1,5,8,1,1,1,6,5,2,78,45], 代表该客户近n天去各四级公司的情况
        sequence_data = sequence_data.groupby('cust_dk')['sequence_value'].apply(list).reset_index()

        # sequence_data.rename(columns={"sequence_value": "sequence_mode_list"}, inplace=True)

        sequence_data.to_csv("../data/cluster_data/用户过去n天的四级公司出勤模式序列-stage3.csv")

        self.attendance_mode_data = sequence_data

    def cluster(self, data: pd.DataFrame, column_name: str):
        logging.info('----聚类样本簇数： {}'.format(self.n_clusters))
        cluster_data = k_shape(data=data, column_name=column_name, n_clusters=self.n_clusters)
        self.cluster_data = cluster_data
        self.cluster_data.to_csv(f"../data/cluster_data/dataset_{self.dataset_id}/cluster_{self.n_clusters}_output_data.csv", index=False)

    def sequence_diagram(self, data: pd.DataFrame, column_name: str, cluster: int):
        # 设置图形大小
        df = data.copy()
        # df[column_name] = data[column_name].apply(ast.literal_eval)

        df = df.sample(n=5, random_state=0).reset_index(drop=True)

        # 设置子图布局
        fig, axs = plt.subplots(nrows=len(df), figsize=(60, 6 * len(df) + 1))

        # 遍历每个客户，并在相应的子图中绘制方波图
        for index, row in df.iterrows():
            axs[index].step(range(len(row[column_name])), row[column_name], where='post', linewidth=5)  # 线条加粗
            # axs[index].set_title(row['cust_dk'])
            # axs[index].set_xlabel('时间')
            # axs[index].set_ylabel('出勤模式')
            axs[index].set_xticks(range(self.date_length))
            axs[index].set_yticks([0, 1])
            axs[index].set_xticklabels([])  # 取消横坐标标签
            axs[index].set_yticklabels([])  # 取消纵坐标标签

        plt.rcParams['font.sans-serif'] = ['SimHei']
        # 调整布局
        plt.tight_layout()

        # 保存图片
        image_name = f"客户出勤序列k-shape聚类样本结果-分类{cluster}.jpg"
        plt.savefig(f'../images/{image_name}')

        # 显示图形
        plt.show()

    def main(self):
        logging.info("------构造聚类原始样本------")
        self.create_cluster_dataset()
        if self.cluster_data_id == 1:
            logging.info("------统计客户当天去四级公司的个数------")
            self.create_user_l4_times()
            logging.info("------对客户-出勤模式样本聚类------")
            self.cluster(self.user_l4_times_data, column_name="sequence_value")
        elif self.cluster_data_id == 2:
            logging.info("------根据原始样本构造客户-出勤模式样本------")
            self.encode_attendance_sequence()
            self.cluster(self.attendance_mode_data, column_name="sequence_value")
        logging.info("------聚类效果可视化------")
        for cluster in range(0, self.n_clusters):
            data = self.cluster_data[self.cluster_data['cluster'] == cluster]
            self.sequence_diagram(data, "sequence_value", cluster)









