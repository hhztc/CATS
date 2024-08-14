import ast
import numpy as np
import pandas as pd

from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance


def k_shape(data: pd.DataFrame, column_name: str, n_clusters: int):
    # 将字符串表示的序列转为实际的列表
    # data[column_name] = data[column_name].apply(ast.literal_eval)
    x = np.array(data[column_name].tolist())
    # x = x.reshape(-1, 8, 1)
    # print(x.shape)
    # 将时间序列进行标准化
    # x_flat = TimeSeriesScalerMinMax().fit_transform(x)
    x_flat = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(x)
    # print(x_flat)
    # 创建并拟合 k-形状聚类模型
    model = KShape(n_clusters=n_clusters,  n_init=5, verbose=True, random_state=42)
    y_pred = model.fit_predict(x_flat)

    # 将聚类结果添加到 DataFrame
    data['cluster'] = y_pred

    return data
