import pytest

from prepydf.preprocessing import PreprocessPipeline, Select, Drop, Bin
from prepydf.tests.test_utilities import check_attributes, fit_transform_and_compare, expected

preprocess_functions = [Drop('float'), Bin('integer', bins=2)]
breaking_preprocess_functions = [Select('non present column')] + preprocess_functions


def test_ok_no_functions(expected):
    fit_transform_and_compare(expected, PreprocessPipeline([]))


def test_should_raise_error_if_function_raises(expected):
    with pytest.raises(Exception):
        PreprocessPipeline(breaking_preprocess_functions).fit_transform(expected)


def test_can_access_functions_with_index():
    for i in range(len(preprocess_functions)):
        expected = preprocess_functions[i]
        computed = PreprocessPipeline(preprocess_functions)[i]
        assert expected == computed


def test_equals_transformations(expected):
    expected = _manual_preprocess_functions_application(expected)
    fit_transform_and_compare(expected, PreprocessPipeline(preprocess_functions))


def test_equals():
    assert PreprocessPipeline(preprocess_functions) != PreprocessPipeline(breaking_preprocess_functions)
    assert PreprocessPipeline(preprocess_functions) == PreprocessPipeline(preprocess_functions)


def test_sklearn_compatibility():
    return check_attributes(PreprocessPipeline)


def _manual_preprocess_functions_application(expected):
    for f in preprocess_functions:
        expected = f.fit_transform(expected)
    return expected
