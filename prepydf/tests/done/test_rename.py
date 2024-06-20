from prepydf.preprocessing import Rename
from prepydf.tests.test_utilities import fit_transform_and_compare, check_attributes

renames = {'integer': 'new_integer', 'float': 'new_float'}
wrong_renames = {'not_existing': 'still_not_existing'}


def test_filter(expected):
    fit_transform_and_compare(expected.rename(columns=renames), Rename(renames))


def test_invariant_column_not_present(expected):
    fit_transform_and_compare(expected, Rename(wrong_renames))


def test_sklearn_compatibility():
    check_attributes(Rename)


def test_equals():
    assert Rename(renames) != Rename(wrong_renames)
    assert Rename(renames) == Rename(renames)
