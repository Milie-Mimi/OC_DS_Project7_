''' Fonctions de preprocessing des datasets du projet 7
'''

import pandas as pd
import numpy as np
import time
import fct_eda
import gc
from contextlib import contextmanager


# --------------------------------------------------------------------------------
# ---------------------- PREPROCESSING DATASETS ----------------------------------
# --------------------------------------------------------------------------------


def preprocess_app_train_test():
    '''Fonction de preprocessing des datasets application_train et application_test:
    merge, traitement des données manquantes, traitement des outliers et des modalités,
    feature engineering
    
    Arguments:
    --------------------------------
    
    
    return:
    --------------------------------
    Dataframe preprocessé'''
    
    # Chargement des datasets et regroupement
    app_train = pd.read_csv('data/application_train.csv', sep=',', encoding='utf-8')
    
    # Traitement des données manquantes
        # Suppression des variables avec 30% ou plus de NaN
    app_train = fct_eda.tx_rempl_min(app_train, 70)
        # iterative imputer pour les variables corrélées entre-elles
    app_train['OBS_30_CNT_SOCIAL_CIRCLE'] = fct_eda.iterative_imputer_function(
        app_train[['OBS_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE']])[:,0]
    app_train['OBS_60_CNT_SOCIAL_CIRCLE'] = fct_eda.iterative_imputer_function(
        app_train[['OBS_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE']])[:,1]
    app_train['AMT_GOODS_PRICE'] = fct_eda.iterative_imputer_function(
        app_train[['AMT_GOODS_PRICE', 'AMT_CREDIT']])[:,0]
    app_train['CNT_FAM_MEMBERS'] = fct_eda.iterative_imputer_function(
        app_train[['CNT_FAM_MEMBERS', 'CNT_CHILDREN']])[:,0]
    app_train['DEF_60_CNT_SOCIAL_CIRCLE'] = fct_eda.iterative_imputer_function(
        app_train[['DEF_60_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE']])[:,0]
    app_train['DEF_30_CNT_SOCIAL_CIRCLE'] = fct_eda.iterative_imputer_function(
        app_train[['DEF_60_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE']])[:,1]
    app_train['AMT_GOODS_PRICE'] = fct_eda.iterative_imputer_function(
        app_train[['AMT_GOODS_PRICE', 'AMT_ANNUITY']])[:,0]
    app_train['AMT_ANNUITY'] = fct_eda.iterative_imputer_function(
        app_train[['AMT_GOODS_PRICE', 'AMT_ANNUITY']])[:,1]
    app_train['AMT_ANNUITY'] = fct_eda.iterative_imputer_function(
        app_train[['AMT_CREDIT', 'AMT_ANNUITY']])[:,1]
        # Médiane et mode pour variables avec taux de NaN > 80% (hors variable TARGET)
    df_nan = fct_eda.nan_a_retraiter(app_train)
    col_med = df_nan[(df_nan['Tx_rempl']>80) & 
                     (df_nan['dtypes'] == 'float64') &
                     (~ df_nan.index.str.contains('TARGET'))].index.tolist()
    for col in col_med:
        app_train[col] = app_train[col].fillna(app_train[col].median())
    del app_train['AMT_REQ_CREDIT_BUREAU_YEAR']
    del app_train['EXT_SOURCE_3']
    app_train['NAME_TYPE_SUITE'] = app_train['NAME_TYPE_SUITE'].fillna(app_train['NAME_TYPE_SUITE'].mode()[0])
    
    # Outliers et traitement des modalités
    app_train.drop(app_train[app_train['DAYS_EMPLOYED'] == 365243].index, inplace=True)    
    app_train = app_train[app_train['CODE_GENDER'].isin(["F", "M"])]
    
    # Feature engineering
    app_train['DAYS_EMPLOYED_PERC'] = app_train['DAYS_EMPLOYED'] / app_train['DAYS_BIRTH']
    app_train['CREDIT_GOODS_PERC'] = app_train['AMT_CREDIT'] / app_train['AMT_GOODS_PRICE']
    app_train['INCOME_CREDIT_PERC'] = app_train['AMT_INCOME_TOTAL'] / app_train['AMT_CREDIT']
    app_train['INCOME_PER_PERSON'] = app_train['AMT_INCOME_TOTAL'] / app_train['CNT_FAM_MEMBERS']
    app_train['DEBT_RATIO'] = app_train['AMT_ANNUITY'] / app_train['AMT_INCOME_TOTAL']
    app_train['PAYMENT_RATE'] = app_train['AMT_ANNUITY'] / app_train['AMT_CREDIT']
    
    new_list_var_quanti = ['DAYS_EMPLOYED_PERC', 'CREDIT_GOODS_PERC', 'INCOME_CREDIT_PERC',
                           'INCOME_PER_PERSON', 'DEBT_RATIO', 'PAYMENT_RATE']
    
    new_list_var_quali = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'NAME_TYPE_SUITE',
                          'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
                          'NAME_HOUSING_TYPE', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL',
                          'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'WEEKDAY_APPR_PROCESS_START', 
                          'REG_REGION_NOT_LIVE_REGION', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY',
                          'LIVE_CITY_NOT_WORK_CITY', 'ORGANIZATION_TYPE', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3',
                          'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_11',
                          'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 
                          'FLAG_DOCUMENT_18']
    
    app_train = app_train.set_index('SK_ID_CURR')
    app_train = app_train[new_list_var_quanti + new_list_var_quali + ['TARGET']]
    app_train, app_train_cat = fct_eda.categories_encoder(app_train, nan_as_category = False)
    
    gc.collect()
    
    return app_train


def preprocess_bureau_balance():
    '''Fonction de preprocessing des datasets bureau et bureau_balance:
    merge, traitement des données manquantes, traitement des outliers et des modalités,
    feature engineering
    
    Arguments:
    --------------------------------
    
    
    return:
    --------------------------------
    Dataframe preprocessé'''
    
    
    # Chargement des datasets
    bureau = pd.read_csv('data/bureau.csv', sep = ',', encoding='utf-8')
    bureau_balance = pd.read_csv('data/bureau_balance.csv', sep=',', encoding='utf-8')
    
    # Preprocessing bureau_balance
    bureau_balance, bureau_balance_cat = fct_eda.categories_encoder(bureau_balance, nan_as_category = False)
        # Dictionnaire des agrégations
    bureau_balance_aggreg = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bureau_balance_cat:
        bureau_balance_aggreg[col] = ['mean']
       # Groupe par SK_ID_BUREAU avec application du dictionnaire des agrégations
    bureau_balance = bureau_balance.groupby('SK_ID_BUREAU').agg(bureau_balance_aggreg)
        # Renommage des colonnes pour éviter les niveaux multiples
    bureau_balance.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bureau_balance.columns.tolist()])
        # Fusion de bureau et bureau balance et suppression de la variable SK_ID_BUREAU
    bureau_bb = bureau.join(bureau_balance, how='left', on='SK_ID_BUREAU')
    bureau_bb.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    
    # Traitement des données manquantes
        # Moyenne pour les variables STATUS
    for col in bureau_bb.columns[bureau_bb.columns.str.contains('STATUS')]:
        bureau_bb[col] = bureau_bb[col].fillna(bureau_bb[col].mean())
    del bureau_bb['STATUS_X_MEAN']
    del bureau_bb['STATUS_C_MEAN']
    del bureau_bb['STATUS_0_MEAN']
        # Suppression des variables avec 30% ou plus de NaN
    bureau_bb_nan = fct_eda.tx_rempl_min(bureau_bb, 70)
        # Médiane pour variables avec taux de remplissage > 70%
    df_nan = fct_eda.nan_a_retraiter(bureau_bb)
    col_med = df_nan[(df_nan['Tx_rempl']>70) & 
                 (df_nan['dtypes'] == 'float64')].index.tolist()
    for col in col_med:
        bureau_bb[col] = bureau_bb[col].fillna(bureau_bb[col].median())

    # Traitement des outliers
    bureau_bb = bureau_bb[~(bureau_bb['DAYS_CREDIT_ENDDATE'] < -20000) & 
                          ~(bureau_bb['DAYS_CREDIT_UPDATE'] < -20000)]

    # Suppression de variables
    del bureau_bb['DAYS_CREDIT_UPDATE']
    del bureau_bb['AMT_CREDIT_SUM']
    del bureau_bb['STATUS_3_MEAN']   

    # Feature engineering
    bureau_bb_feat, bureau_bb_feat_cat = fct_eda.categories_encoder(bureau_bb, nan_as_category = False)
        # Actualisation de la liste des variables catégorielles de bureau_balance
    bureau_balance_cat.remove('STATUS_3')
    bureau_balance_cat.remove('STATUS_0')
    bureau_balance_cat.remove('STATUS_C')
    bureau_balance_cat.remove('STATUS_X')
        # Dictionnaire des agrégations
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
    }
    cat_aggregations = {}
    for cat in bureau_bb_feat_cat:
        cat_aggregations[cat] = ['mean']
    for cat in bureau_balance_cat:
        cat_aggregations[cat + "_MEAN"] = ['mean']
        # Groupe par SK_ID_CURR avec application du dictionnaire des agrégations
    bureau_bb_feat_gby = bureau_bb_feat.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
        # Renommage des colonnes pour éviter les niveaux multiples
    bureau_bb_feat_gby.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() 
                                           for e in bureau_bb_feat_gby.columns.tolist()])
        # Indicateurs agrégés crédits actifs
    cred_actifs = bureau_bb_feat[bureau_bb_feat['CREDIT_ACTIVE_Active'] == 1]
    cred_actifs_agg = cred_actifs.groupby('SK_ID_CURR').agg(num_aggregations)
    cred_actifs_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() 
                                        for e in cred_actifs_agg.columns.tolist()])
    bureau_agg = bureau_bb_feat_gby.join(cred_actifs_agg, how = 'left', on = 'SK_ID_CURR')
        # Indicateurs agrégés crédits fermés
    cred_closed = bureau_bb_feat[bureau_bb_feat['CREDIT_ACTIVE_Closed'] == 1]
    cred_closed_agg = cred_closed.groupby('SK_ID_CURR').agg(num_aggregations)
    cred_closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() 
                                        for e in cred_closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(cred_closed_agg, how = 'left', on = 'SK_ID_CURR')
        # Imputation des variables par la constante 0
    for col in cred_actifs_agg:
        bureau_agg[col] = bureau_agg[col].fillna(0)
    for col in cred_closed_agg:
        bureau_agg[col] = bureau_agg[col].fillna(0)
    
    del bureau, bureau_balance, bureau_bb, bureau_bb_feat, bureau_bb_feat_gby, cred_actifs_agg, cred_closed_agg
    
    gc.collect()
    
    return bureau_agg


def preprocess_previous_app():
    '''Fonction de preprocessing du dataset previous application:
    traitement des données manquantes, feature engineering et 
    sélection de variables
    
    Arguments:
    --------------------------------
    
    
    return:
    --------------------------------
    Dataframe preprocessé'''
    
    
    # Chargement du dataset
    previous_app = pd.read_csv('data/previous_application.csv',
                           sep=',',
                           encoding='utf-8')
    
    # Traitement des données manquantes
        
        # Suppression des variables avec 30% ou plus de NaN
    previous_app = fct_eda.tx_rempl_min(previous_app, 70)
        
        # Iterative imputer pour les variables corrélées entre-elles
    previous_app['AMT_CREDIT'] = fct_eda.iterative_imputer_function(
        previous_app[['AMT_CREDIT', 'AMT_APPLICATION']])[:,0]
    previous_app['AMT_ANNUITY'] = fct_eda.iterative_imputer_function(
        previous_app[['AMT_ANNUITY', 'AMT_CREDIT', 'AMT_GOODS_PRICE',
                      'AMT_APPLICATION', 'CNT_PAYMENT']])[:,0]

        # Médiane et mode pour variables avec taux de NaN > 70% (hors variable TARGET)
    df_nan = fct_eda.nan_a_retraiter(previous_app)
    col_med = df_nan[(df_nan['Tx_rempl']>70) & 
                     (df_nan['dtypes'] == 'float64')].index.tolist()
    for col in col_med:
        previous_app[col] = previous_app[col].fillna(previous_app[col].median())
        
        # Imputation de la variable NAME_TYPE_SUITE par le mode
    previous_app['PRODUCT_COMBINATION'] = previous_app['PRODUCT_COMBINATION'].fillna(
        previous_app['PRODUCT_COMBINATION'].mode()[0])
    
    # Feature engineering
    previous_app_feat = previous_app.copy()
    previous_app_feat['CREDIT_GOODS_PERC'] = previous_app_feat['AMT_CREDIT'] / previous_app_feat['AMT_GOODS_PRICE']
    previous_app_feat = previous_app_feat[~previous_app_feat['CREDIT_GOODS_PERC'].isna()]
    
        # Suppression des variables non conservées
    del previous_app_feat['AMT_APPLICATION']
    del previous_app_feat['AMT_CREDIT']
    del previous_app_feat['AMT_GOODS_PRICE']
    del previous_app_feat['AMT_ANNUITY']
        
        # One hot encoder
    previous_app_feat, previous_app_cat = fct_eda.categories_encoder(previous_app_feat, nan_as_category = False)
        
        # Dictionnaire des agrégations pour les variables numériques
    num_aggregations = {
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'FLAG_LAST_APPL_PER_CONTRACT' : ['mean'],
        'NFLAG_LAST_APPL_IN_DAY': ['mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'SELLERPLACE_AREA' : ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
        'CREDIT_GOODS_PERC' : ['mean'],
    }
        
        # Dictionnaire des agrégations pour les variables catégorielles
    cat_aggregations = {}
    for cat in previous_app_cat:
        cat_aggregations[cat] = ['mean']
        
        # Groupe par SK_ID_CURR + agrégation
    previous_app_feat_gby = previous_app_feat.groupby('SK_ID_CURR').agg({**num_aggregations,
                                                                         **cat_aggregations})
        
        # On renomme les colonnes pour supprimer le multiindex
    previous_app_feat_gby.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() 
                                              for e in previous_app_feat_gby.columns.tolist()])
    previous_app_feat_gby = previous_app_feat_gby[~previous_app_feat_gby.isin([np.inf, -np.inf]).any(1)]
        
        # Imputation des valeurs manquantes par la moyenne
    for col in previous_app_feat_gby.columns:
        previous_app_feat_gby[col] = previous_app_feat_gby[col].fillna(previous_app_feat_gby[col].mean())
        
        # Indicateurs agrégés crédits approuvés
    cred_approuv = previous_app_feat[previous_app_feat['NAME_CONTRACT_STATUS_Approved'] == 1]
    cred_approuv_agg = cred_approuv.groupby('SK_ID_CURR').agg(num_aggregations)
    cred_approuv_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() 
                                         for e in cred_approuv_agg.columns.tolist()])
    cred_approuv_agg = cred_approuv_agg[~cred_approuv_agg.isin([np.inf, -np.inf]).any(1)]
    previous_app_agg = previous_app_feat_gby.join(cred_approuv_agg, how = 'left', on = 'SK_ID_CURR')

        # Indicateurs agrégés crédits refusés
    cred_refus = previous_app_feat[previous_app_feat['NAME_CONTRACT_STATUS_Refused'] == 1]
    cred_refus_agg = cred_refus.groupby('SK_ID_CURR').agg(num_aggregations)
    cred_refus_agg.columns = pd.Index(['REFUSED' + e[0] + "_" + e[1].upper() 
                                       for e in cred_refus_agg.columns.tolist()])
    cred_refus_agg = cred_refus_agg[~cred_refus_agg.isin([np.inf, -np.inf]).any(1)]
    previous_app_agg = previous_app_agg.join(cred_refus_agg, how = 'left', on = 'SK_ID_CURR')

        # Imputation des valeurs manquantes par la constante 0
    for col in cred_approuv_agg.columns:
        previous_app_agg[col] = previous_app_agg[col].fillna(0)

    for col in cred_refus_agg.columns:
        previous_app_agg[col] = previous_app_agg[col].fillna(0)
    
    del previous_app, previous_app_feat, previous_app_feat_gby, cred_approuv_agg, cred_refus_agg
    
    gc.collect()
    
    return previous_app_agg


def preprocess_POS_CASH_balance():
    '''Fonction de preprocessing du dataset POS_CASH_balance:
    traitement des données manquantes, feature engineering et 
    sélection de variables
    
    Arguments:
    --------------------------------
    
    
    return:
    --------------------------------
    Dataframe preprocessé'''
    
    
    # Chargement du dataset
    pos_cash = pd.read_csv('data/POS_CASH_balance.csv',
                           sep=',',
                           encoding='utf-8')
    
    # Traitement des données manquantes (médiane)
    for col in ['CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE']:
        pos_cash[col] = pos_cash[col].fillna(pos_cash[col].median())

    # Feature engineering
    del pos_cash['CNT_INSTALMENT']
    pos_cash_feat, pos_cash_cat = fct_eda.categories_encoder(pos_cash, nan_as_category = False)
        # Dictionnaire des agrégations pour les variables numériques
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    
        # Dictionnaire des agrégations pour les variables catégorielles
    for cat in pos_cash_cat:
        aggregations[cat] = ['mean']

        # Groupe par SK_ID_CURR + agrégation
    pos_cash_feat_gby = pos_cash_feat.groupby('SK_ID_CURR').agg(aggregations)

        # On renomme les colonnes pour supprimer le multi-index
    pos_cash_feat_gby.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() 
                                          for e in pos_cash_feat_gby.columns.tolist()])
    
        # Nombre de soldes mensuels de crédits antérieurs par SK_ID_CURR
    pos_cash_feat_gby['POS_COUNT'] = pos_cash_feat.groupby('SK_ID_CURR').size()

    del pos_cash, pos_cash_feat

    gc.collect()
    
    return pos_cash_feat_gby


def preprocess_installments_payments():
    '''Fonction de preprocessing du dataset installments_payments:
    traitement des données manquantes, feature engineering et 
    sélection de variables
    
    Arguments:
    --------------------------------
    
    
    return:
    --------------------------------
    Dataframe preprocessé'''
    
    
    # Chargement du dataset
    install_pay = pd.read_csv('data/installments_payments.csv',
                              sep=',',
                              encoding='utf-8')
    
    # Traitement des données manquantes (médiane)
    for col in ['DAYS_ENTRY_PAYMENT', 'AMT_PAYMENT']:
        install_pay[col] = install_pay[col].fillna(install_pay[col].median())

    # Feature engineering
    install_pay['PAYMENT_DIFF'] = install_pay['AMT_INSTALMENT'] - install_pay['AMT_PAYMENT']
    install_pay['DAYS_PAST_DUE'] = install_pay['DAYS_ENTRY_PAYMENT'] - install_pay['DAYS_INSTALMENT']
    install_pay['DAYS_PAST_DUE'] = install_pay['DAYS_PAST_DUE'].apply(lambda x: x if x > 0 else 0)
    install_pay['DAYS_BEFORE_DUE'] = install_pay['DAYS_INSTALMENT'] - install_pay['DAYS_ENTRY_PAYMENT']
    install_pay['DAYS_BEFORE_DUE'] = install_pay['DAYS_BEFORE_DUE'].apply(lambda x: x if x > 0 else 0)
        
        # Dictionnaire des agrégations pour les variables numériques
    aggregations = {
        'DAYS_PAST_DUE': ['max', 'mean', 'sum'],
        'DAYS_BEFORE_DUE': ['max', 'mean', 'sum'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum'],
    }
   
        # Groupe par SK_ID_CURR + agrégation
    install_pay_feat_gby = install_pay.groupby('SK_ID_CURR').agg(aggregations)

        # On renomme les colonnes pour supprimer le multi-index
    install_pay_feat_gby.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() 
                                             for e in install_pay_feat_gby.columns.tolist()])

        # Calcul du nombre de versements par SK_ID_CURR
    install_pay_feat_gby['INSTAL_COUNT'] = install_pay.groupby('SK_ID_CURR').size()

        # Traitement des valeurs manquantes
    #for col in ['INSTAL_PAYMENT_PERC_MEAN']:
    #    install_pay_feat_gby[col] = install_pay_feat_gby[col].fillna(install_pay_feat_gby[col].mean())

    del install_pay
    gc.collect()
    
    return install_pay_feat_gby


def preprocess_credit_card_balance():
    '''Fonction de preprocessing du dataset credit_card_balance:
    traitement des données manquantes, feature engineering et 
    sélection de variables
    
    Arguments:
    --------------------------------
    
    
    return:
    --------------------------------
    Dataframe preprocessé'''
    
    
    # Chargement du dataset
    cc_balance = pd.read_csv('data/credit_card_balance.csv',
                             sep=',',
                             encoding='utf-8')
    
    # Traitement des données manquantes (médiane)
    df_nan = fct_eda.nan_a_retraiter(cc_balance)
    col_med = df_nan[(df_nan['Tx_rempl']>=80)].index.tolist()
    for col in col_med:
        cc_balance[col] = cc_balance[col].fillna(cc_balance[col].median())

    # Suppression variables suite analyses
    cc_balance.drop(['AMT_DRAWINGS_ATM_CURRENT', 'AMT_DRAWINGS_POS_CURRENT',
                      'AMT_INST_MIN_REGULARITY', 'AMT_PAYMENT_TOTAL_CURRENT',
                      'AMT_RECEIVABLE_PRINCIPAL', 'AMT_RECIVABLE',
                      'AMT_TOTAL_RECEIVABLE', 'CNT_DRAWINGS_ATM_CURRENT',
                      'CNT_DRAWINGS_OTHER_CURRENT', 'CNT_DRAWINGS_POS_CURRENT'],
                    axis=1, inplace = True)

    # Feature engineering

        # One hot encoder de la variable catégorielle
    cc_balance_feat, cc_balance_cat = fct_eda.categories_encoder(cc_balance, nan_as_category = False)

        # Groupe par SK_ID_PREV + agrégations
    cc_balance_feat.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_balance_feat_gby = cc_balance_feat.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum'])

        # On renomme les colonnes pour supprimer le multi-index
    cc_balance_feat_gby.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() 
                                            for e in cc_balance_feat_gby.columns.tolist()])

        # Calcul du nombre d'historique
    cc_balance_feat_gby['CC_COUNT'] = cc_balance_feat.groupby('SK_ID_CURR').size()
   
    del cc_balance, cc_balance_feat
    gc.collect()
    
    return cc_balance_feat_gby


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print(f"{title} fait en {np.round(time.time() - t0, 0)}s ")



# --------------------------------------------------------------------------------
# ----------------------- PREPROCESSING GLOBAL -----------------------------------
# --------------------------------------------------------------------------------


def main():
    with timer("Processing application_train"):
        df = preprocess_app_train_test()
        print("Application_train shape:", df.shape)
    with timer("Processing bureau et bureau_balance"):
        bureau = preprocess_bureau_balance()
        print("Bureau shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()
    with timer("Processing previous_applications"):
        prev = preprocess_previous_app()
        print("Previous applications shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()
    with timer("Processing POS-CASH balance"):
        pos = preprocess_POS_CASH_balance()
        print("Pos-cash balance shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer("Processing installments payments"):
        ins = preprocess_installments_payments()
        print("Installments payments shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    with timer("Processing credit card balance"):
        cc = preprocess_credit_card_balance()
        print("Credit card balance shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        df = fct_eda.tx_rempl_min(df, 80)
        df_nan = fct_eda.nan_a_retraiter(df)
        col_med = df_nan[(df_nan['Tx_rempl']>80) & 
                 (df_nan['dtypes'] == 'float64') &
                 (~ df_nan.index.str.contains('TARGET'))].index.tolist()
        for col in col_med:
            df[col] = df[col].fillna(df[col].median())
        del cc, df_nan, col_med
        gc.collect()
    return df