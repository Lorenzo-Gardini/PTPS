import numpy as np
import pandas as pd
import pytest

from src.prepydf.preprocessing import Update
from tests.test_utilities import fit_transform_and_compare, check_attributes, full_dataframe, expected


update_dictionary = {
    'new_integer': np.arange(0, len(full_dataframe())),
    'new_float': np.arange(0., len(full_dataframe()))
}
update_series = pd.Series(update_dictionary['new_integer'], name='new_integer')
update_dataframe = pd.DataFrame(update_dictionary)


def test_update_dictionary(expected):
    for key, val in update_dictionary.items():
        expected[key] = val
    fit_transform_and_compare(expected, Update(update_dictionary))


def test_update_series(expected):
    expected[update_series.name] = update_series.values
    fit_transform_and_compare(expected, Update(update_series))


def test_update_dataframe(expected):
    fit_transform_and_compare(pd.concat([expected, update_dataframe], axis=1),
                              Update(update_dataframe))


def test_sklearn_compatibility():
    check_attributes(Update)


@pytest.mark.parametrize('update', [update_series, update_dictionary, update_dataframe])
def test_equals(update):
    for elem in [update_series, update_dictionary, update_dataframe]:
        if type(elem) != type(update):
            assert Update(update) != Update(elem)
    assert Update(update) == Update(update)
