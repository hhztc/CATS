from dataclasses import dataclass, field
from typing import List, OrderedDict, Optional, Tuple
from dataclasses_json import dataclass_json

from .features import FeatureDtype, Features, Feature, FeatureType
from .features import StringLookup, Normalization
from .base_transformer import BaseTransform

import pandas as pd
import logging
import os
import sys

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.root.setLevel(level=logging.INFO)


@dataclass_json
@dataclass
class DatasetTransform(BaseTransform):
    features: Features = None  # 必须有
    categorical_single_feature_names: List[str] = None
    continuous_single_feature_names: List[str] = None
    unchanged_feature_names: List[str] = None
    categorical_list_feature_set_dict: OrderedDict[str, set] = field(default_factory=OrderedDict)
    continuous_list_feature_set_dict: OrderedDict[str, list] = field(default_factory=OrderedDict)
    #  k: int = None

    def __post_init__(self):
        # maybe do something
        self.categorical_single_feature_names = ['cust_dk']

        self.continuous_single_feature_names = ['interval_from_last_purchase', 'avg_interval_last_7_days']

        self.unchanged_feature_names = ['date', 'attendance_mode', 'month_of_year', 'quarter_of_year', 'day_of_week',
                                        'is_weekday', 'week_of_year', 'day_of_month', 'is_holiday', 'target', 'cluster']

    def _fit_categorical_single_feature(self, feature_name: str, input_dataset: pd.DataFrame, column_name: str):
        feature = Feature(name=feature_name, feature_type=FeatureType.Categorical)
        # 注意编码要从1开始，0代表未见的id。获取长度的时候记得要len(feature.category_encode)+1
        feature.category_encode = StringLookup(name=feature_name, offset=1)
        feature.category_encode.fit(input_dataset[column_name].unique().tolist())
        self.features.add(feature)

    def _transform_categorical_single_feature(self, feature_name: str, input_dataset: pd.DataFrame, column_name: str, output_dataset: pd.DataFrame):
        feature = self.features[feature_name]
        transform_series = feature.category_encode.transform_series(input_dataset[column_name], default=0).rename(column_name)
        new_df = pd.DataFrame(transform_series)
        return pd.concat([output_dataset.reset_index(drop=True), new_df.reset_index(drop=True)], axis=1)

    def _fit_transform_categorical_single_feature(self, feature_name: str, input_dataset: pd.DataFrame, column_name: str, output_dataset: pd.DataFrame):
        self._fit_categorical_single_feature(feature_name, input_dataset, column_name)
        return self._transform_categorical_single_feature(feature_name, input_dataset, column_name,
                                                          output_dataset=output_dataset)

    def _fit_continuous_single_feature(self, feature_name: str, input_dataset: pd.DataFrame, column_name: str):
        feature = Feature(name=feature_name, feature_type=FeatureType.Continuous)
        feature.normalization = Normalization(name=feature_name)
        feature.normalization.fit(input_dataset[column_name])
        self.features.add(feature)

    def _transform_continuous_single_feature(self, feature_name: str, input_dataset: pd.DataFrame, column_name: str, output_dataset: pd.DataFrame):
        feature = self.features[feature_name]
        transform_series = feature.normalization.transform(input_dataset[column_name]).rename(column_name)
        new_df = pd.DataFrame(transform_series)
        return pd.concat([output_dataset.reset_index(drop=True), new_df.reset_index(drop=True)], axis=1)

    def _fit_transform_continuous_single_feature(self, feature_name: str, input_dataset: pd.DataFrame, column_name: str, output_dataset: pd.DataFrame):
        self._fit_continuous_single_feature(feature_name, input_dataset, column_name)
        return self._transform_continuous_single_feature(feature_name, input_dataset, column_name,
                                                         output_dataset=output_dataset)

    def _fit_categorical_list_feature(self, feature_name: str, input_dataset: pd.DataFrame, column_name: str):
        feature = Feature(name=feature_name, feature_type=FeatureType.CategoricalSequence)
        # 注意编码要从1开始，0代表未见的id。获取长度的时候记得要len(feature.category_encode)+1
        feature.category_encode = StringLookup(name=feature_name, offset=1)
        feature.category_encode.fit(list(self.categorical_list_feature_set_dict[feature_name]))
        self.features.add(feature)

    def _transform_categorical_list_feature(self, feature_name: str, input_dataset: pd.DataFrame, column_name: str,
                                            output_dataset: pd.DataFrame, mask: Optional[Tuple[float, float]] = None):
        # transform的时候要做好padding，返回的时候就应该是长度为k的list。
        feature = self.features[feature_name]
        transform_series = feature.category_encode.transform_list(input_dataset[column_name], default=0, padding=self.k, mask=mask).rename(column_name)
        new_df = pd.DataFrame(transform_series)
        return pd.concat([output_dataset.reset_index(drop=True), new_df.reset_index(drop=True)], axis=1)

    def _fit_transform_categorical_list_feature(self, feature_name: str, input_dataset: pd.DataFrame, column_name: str,
                                                output_dataset: pd.DataFrame,
                                                mask: Optional[Tuple[float, float]] = None):
        self._fit_categorical_list_feature(feature_name, input_dataset, column_name)
        return self._transform_categorical_list_feature(feature_name, input_dataset, column_name,
                                                        output_dataset=output_dataset, mask=mask)

    def _fit_continuous_list_feature(self, feature_name: str, input_dataset: pd.DataFrame, column_name: str):
        feature = Feature(name=feature_name, feature_type=FeatureType.ContinuousSequence)
        feature.normalization = Normalization(name=feature_name)
        feature.normalization.fit_list(self.continuous_list_feature_set_dict[feature_name])
        self.features.add(feature)

    def _transform_continuous_list_feature(self, feature_name: str, input_dataset: pd.DataFrame, column_name: str, output_dataset: pd.DataFrame):
        # transform的时候要做好padding，返回的时候就应该是长度为k的list。
        feature = self.features[feature_name]
        transform_series = feature.normalization.transform_list(input_dataset[column_name], padding=self.k).rename(column_name)
        new_df = pd.DataFrame(transform_series)
        return pd.concat([output_dataset.reset_index(drop=True), new_df.reset_index(drop=True)], axis=1)

    def _fit_transform_continuous_list_feature(self, feature_name: str, input_dataset: pd.DataFrame, column_name: str, output_dataset: pd.DataFrame):
        self._fit_continuous_list_feature(feature_name, input_dataset, column_name)
        return self._transform_continuous_list_feature(feature_name, input_dataset, column_name, output_dataset=output_dataset)

    def _fit_transform_list_feature(self, feature_name: str, input_dataset: pd.DataFrame):
        input_dataset[feature_name] = input_dataset[feature_name].tolist()
        expanded_data = input_dataset[feature_name].apply(pd.Seriese).add_prefix('new_column')
        expanded_data.columns = [f'purchase_day_{i}' for i in range(1, 8)]
        input_dataset = pd.concat([input_dataset, expanded_data], axis=1)
        input_dataset.drop(feature_name, axis=1, inplace=True)
        return input_dataset

    def fit_transform(self, input_dataset: pd.DataFrame):
        # input_dataset = self._flatten_item_feature(input_dataset=input_dataset)
        output_dataset = pd.DataFrame()

        logger.info('---Sparse feature fit transform-----')
        for name in self.categorical_single_feature_names:
            output_dataset = self._fit_transform_categorical_single_feature(feature_name=name,
                                                                            input_dataset=input_dataset,
                                                                            column_name=name,
                                                                            output_dataset=output_dataset)

        logger.info('---Continuous feature fit transform-----')
        for name in self.continuous_single_feature_names:
            output_dataset = self._fit_transform_continuous_single_feature(feature_name=name,
                                                                           input_dataset=input_dataset,
                                                                           column_name=name,
                                                                           output_dataset=output_dataset)

        logger.info('---Unchanged feature transform-----')
        for name in self.unchanged_feature_names:
            output_dataset[name] = input_dataset[name]

        output_dataset = output_dataset[['date', 'cust_dk', 'attendance_mode', 'month_of_year', 'quarter_of_year',
                                         'day_of_week', 'is_weekday', 'week_of_year', 'day_of_month', 'is_holiday',
                                         'interval_from_last_purchase', 'avg_interval_last_7_days', 'target', 'cluster']]

        return output_dataset

    def transform(self, input_dataset: pd.DataFrame):
        # input_dataset = self._flatten_item_feature(input_dataset=input_dataset)
        output_dataset = pd.DataFrame()

        logger.info('---Sparse feature transform-----')
        for name in self.categorical_single_feature_names:
            output_dataset = self._transform_categorical_single_feature(feature_name=name,
                                                                        input_dataset=input_dataset,
                                                                        column_name=name,
                                                                        output_dataset=output_dataset)

        logger.info('---Continuous feature transform-----')
        for name in self.continuous_single_feature_names:
            output_dataset = self._transform_continuous_single_feature(feature_name=name,
                                                                       input_dataset=input_dataset,
                                                                       column_name=name,
                                                                       output_dataset=output_dataset)
        logger.info('---Unchanged feature transform-----')
        for name in self.unchanged_feature_names:
            output_dataset[name] = input_dataset[name]

        output_dataset = output_dataset[['date', 'cust_dk', 'attendance_mode', 'month_of_year', 'quarter_of_year',
                                         'day_of_week', 'is_weekday', 'week_of_year', 'day_of_month', 'is_holiday',
                                         'interval_from_last_purchase', 'avg_interval_last_7_days', 'target', 'cluster']]

        return output_dataset


@dataclass_json
@dataclass
class DatasetTransformPipeline(BaseTransform):
    features: Features = None
    trans: DatasetTransform = None
    # k: int = None

    def __post_init__(self):
        if self.trans is None:
            self.trans = DatasetTransform(features=Features())

    def fit_transform(self, input_dataset: pd.DataFrame):
        x1: pd.DataFrame = self.trans.fit_transform(input_dataset)
        self.features = self.trans.features
        return x1

    def transform(self, input_dataset: pd.DataFrame):
        x1: pd.DataFrame = self.trans.transform(input_dataset)
        return x1
