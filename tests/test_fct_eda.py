import pandas as pd
from Notebooks import fct_eda


def test_describe_variables_light_check_return_right_columns():
    # given
    df_sample = pd.read_csv('/OLD/df.csv', nrows=3)
    expected = ['Variable name', 'Variable type', 'Example', 'Rows', 'Distinct',
                '% distinct', 'Not NaN', '% Not NaN', 'NaN', '% NaN']
    # when
    dataframe = fct_eda.describe_variables_light(df_sample)
    actual = list(dataframe.columns)
    # then
    assert expected == actual


# python -m pytest
# pytest tests
# pytest test_fct_eda.py

