from Notebooks import fct_model

def test_get_data_check_return_right_columns():
    # given
    expected = ['SK_ID_CURR', 'AGE', 'CODE_GENDER', 'NAME_EDUCATION_TYPE_Lower Secondary & Secondary',
                'YEARS_EMPLOYED', 'YEARS_ID_PUBLISH', 'YEARS_LAST_PHONE_CHANGE', 'REGION_POPULATION_RELATIVE',
                'AMT_CREDIT', 'AMT_GOODS_PRICE', 'CREDIT_GOODS_PERC', 'CREDIT_DURATION', 'AMT_ANNUITY', 'DEBT_RATIO',
                'PAYMENT_RATE', 'EXT_SOURCE_2', 'PREV_YEARS_DECISION_MEAN', 'PREV_PAYMENT_RATE_MEAN',
                'INSTAL_DAYS_BEFORE_DUE_MEAN', 'INSTAL_PAYMENT_DIFF_MEAN', 'INSTAL_DAYS_PAST_DUE_MEAN',
                'POS_MONTHS_BALANCE_MEAN', 'POS_CNT_INSTALMENT_FUTURE_MEAN', 'POS_NB_CREDIT',
                'BURO_AMT_CREDIT_SUM_SUM', 'BURO_YEARS_CREDIT_ENDDATE_MAX', 'BURO_AMT_CREDIT_SUM_DEBT_SUM',
                'BURO_YEARS_CREDIT_ENDDATE_MEAN', 'BURO_AMT_CREDIT_SUM_MEAN', 'BURO_CREDIT_ACTIVE_Active_SUM',
                'BURO_AMT_CREDIT_SUM_DEBT_MEAN']
    # when
    dataframe = fct_model.get_data(nrows=2)
    actual = list(dataframe.columns)
    # then
    assert expected == actual

def test_score_metier_check_return_calculation():
    # given
    ytest = [0, 0, 1, 1, 0, 1]
    y_pred = [0, 1, 1, 0, 0, 1]
    expected = 11
    # when
    actual = fct_model.score_metier(ytest=ytest,
                                    y_pred=y_pred)
    # then
    assert expected == actual

# python -m pytest
