import pytest
from sklearn import preprocessing
from tests.test_utilities import check_attributes, fit_transform_and_compare, expected
from src.preprocessing import LabelEncoder
from pandas.testing import assert_frame_equal

feature = 'integer'
column_not_present = 'column not present'


def test_label_encoder(expected):
    expected[feature] = _fit_label_encoder(expected).transform(expected[feature])
    fit_transform_and_compare(expected, LabelEncoder(feature))


def test_label_encoding_reverse(expected):
    label_encoder = _fit_label_encoder(expected)
    expected[feature] = label_encoder.transform(expected[feature])
    expected[feature] = label_encoder.inverse_transform(expected[feature])

    encoder = LabelEncoder(feature).fit(expected)
    result = encoder.transform(expected)
    assert_frame_equal(expected, encoder.inverse_transform(result))


def test_select_unknown_column(expected):
    with pytest.raises(KeyError):
        LabelEncoder(column_not_present).fit_transform(expected)


def test_sklearn_compatibility():
    check_attributes(LabelEncoder)


def test_equals():
    assert LabelEncoder(column_not_present) != LabelEncoder(feature)
    assert LabelEncoder(feature) == LabelEncoder(feature)


def _fit_label_encoder(expected):
    return preprocessing.LabelEncoder().fit(expected[feature])
