import numpy as np
from pandas import DataFrame
from sklearn import preprocessing
from rex.preprocessing import StandardScaler
from tests.preprocessing_tests.dataframe_preprocessing_test.dataframe_preprocessing_function_test import \
    DataFramePreprocessingFunctionTest


class StandardScalerTest(DataFramePreprocessingFunctionTest):
    group_by_feature = 'categorical_repeated'
    group_by_features = [group_by_feature, 'numerical_repeated']
    feature = 'integer'
    features = [feature, 'float']

    def test_min_max_scaler_one_group_one_feature(self):
        self._transform(self.group_by_feature, self.feature)

    def test_standard_scaler_scaler_groups_feature(self):
        self._transform(self.group_by_features, self.feature)

    def test_standard_scaler_scaler_one_group_features(self):
        self._transform(self.group_by_feature, self.features)

    def test_standard_scaler_scaler_groups_features(self):
        self._transform(self.group_by_features, self.features)

    def test_inverse_transform(self):
        scaler = StandardScaler(self.group_by_features, self.features).fit(self.dataframe)
        computed = scaler.inverse_transform(scaler.transform(self.dataframe))
        self.assertEqual(self.dataframe, computed)

    def _transform(self, group_by, features):
        computed = self.transform_default_dataframe(StandardScaler(group_by, features))

        features = list(np.atleast_1d(features))
        group_by = list(np.atleast_1d(group_by))
        groups = self.dataframe.groupby(group_by)
        for feature in features:
            self.dataframe[feature] = groups[[feature]] \
                .apply(lambda x: DataFrame(preprocessing.StandardScaler().fit_transform(x), index=x.index))

        self.assertEqual(self.dataframe, computed)

    def _preprocess_function_class(self):
        return StandardScaler
