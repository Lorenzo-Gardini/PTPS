import pytest
from sklearn import preprocessing

from src.rex.commons import at_least_list, opt_at_least_list
from src.rex.preprocessing import Normalizer
from prepydf.tests import check_attributes, fit_transform_and_compare

group_by_feature = 'categorical_repeated'
group_by_features = [group_by_feature, 'numerical_repeated']
single_feature = 'repeated_integers'
multiple_features = [single_feature, 'float']


@pytest.mark.parametrize('group_by', [group_by_feature, group_by_features, None])
@pytest.mark.parametrize('feature', [single_feature, multiple_features])
@pytest.mark.parametrize('norm', ['l1', 'l2', 'max'])
def test_features_with_group_by(group_by, feature, norm, expected):
    preprocess_fn = Normalizer(feature, group_by, norm)
    features = at_least_list(feature)
    group_by = opt_at_least_list(group_by)
    if group_by is not None:
        groups = expected.groupby(group_by)
        for feature in features:
            expected[feature] = groups[feature].transform(lambda x: preprocessing.Normalizer(norm).fit_transform(x.values.reshape(1, -1)).flatten())
    else:
        for feature in features:
            expected[feature] = preprocessing.Normalizer(norm).fit_transform(expected[feature].values.reshape(1, -1)).T

    fit_transform_and_compare(expected, preprocess_fn)


def test_sklearn_compatibility():
    check_attributes(Normalizer)


@pytest.mark.parametrize('feature', [single_feature, multiple_features])
@pytest.mark.parametrize('feature2', [single_feature, multiple_features])
@pytest.mark.parametrize('group_by', [group_by_feature, group_by_features, None])
@pytest.mark.parametrize('group_by2', [group_by_feature, group_by_features, None])
@pytest.mark.parametrize('norm', ['l1', 'l2', 'max'])
@pytest.mark.parametrize('norm2', ['l1', 'l2', 'max'])
def test_equals(feature, feature2, group_by, group_by2, norm, norm2):
    if feature == feature2 and group_by == group_by2 and norm == norm2:
        assert Normalizer(feature, group_by, norm) == Normalizer(feature2, group_by2, norm2)
    else:
        assert Normalizer(feature, group_by, norm) != Normalizer(feature2, group_by2, norm2)
