import pandas as pd
from scipy import sparse

from rex.preprocessing import ToCOOMatrix
from tests.preprocessing_tests.dataframe_preprocessing_test.dataframe_preprocessing_function_test import \
    DataFramePreprocessingFunctionTest


class ToCOOMatrixTest(DataFramePreprocessingFunctionTest):
    first_column_name = 'first'
    second_column_name = 'second'
    data_column_name = 'data'

    def test_to_coo_matrix(self):
        converter = ToCOOMatrix(self.first_column_name, self.second_column_name, self.data_column_name)
        computed = converter.fit_transform(self._sparse_dataframe)
        sparse_matrix = sparse.csr_matrix(self._sparse_dataframe.values)
        non_zero_row, non_zero_col = sparse_matrix.nonzero()
        coo_matrix = [
            (self._sparse_dataframe.index[r], self._sparse_dataframe.columns[c], self._sparse_dataframe.iloc[r][c])
            for r, c in zip(non_zero_row, non_zero_col)]
        expected = pd.DataFrame(coo_matrix,
                                columns=[self.first_column_name, self.second_column_name, self.data_column_name])
        self.assertEqual(computed, expected)

    def _preprocess_function_class(self):
        return ToCOOMatrix
