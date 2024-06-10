from fastapi import FastAPI
from sklearn.preprocessing import LabelEncoder
from pickle import load
from pydantic import BaseModel
import pandas as pd


# --------------------------------------------------------------------------------
# ---------------------------- FONCTIONS UTILES  ---------------------------------
# --------------------------------------------------------------------------------

def categories_encoder(df, nan_as_category=True):
    """Fonction de preprocessing des variables catégorielles. Applique un
    One Hot Encoder sur les variables catégorielles non binaires et un Label
    Encoder pour les variables catégorielles binaires.

    Arguments:
    --------------------------------
    df: dataframe: tableau en entrée, obligatoire
    nan_as_category : bool, considère les valeurs manquantes comme une catégorie
    à part entière. Vrai par défaut.

    return:
    --------------------------------
    None
    """

    df_columns = list(df.columns)
    # Colonnes pour OHE (modalités > 2)
    categ_columns_ohe = [col for col in df.columns if df[col].dtype == 'object']
    df_ohe = df[categ_columns_ohe]
    categ_columns_ohe = [col for col in df_ohe.columns if len(list(df_ohe[col].unique())) > 2]
    # Colonnes pour Label Encoder (modalités <= 2)
    categ_columns_le = [col for col in df.columns if df[col].dtype == 'object']
    df_le = df[categ_columns_le]
    categ_columns_le = [col for col in df_le.columns if len(list(df_ohe[col].unique())) <= 2]

    # Label encoder quand modalités <= 2
    le = LabelEncoder()
    for col in df[categ_columns_le]:
        le.fit(df[col])
        df[col] = le.transform(df[col])

    # One Hot Encoder quand modalités > 2
    df = pd.get_dummies(df, columns=categ_columns_ohe, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in df_columns] + categ_columns_le
    return df, new_columns


# --------------------------------------------------------------------------------
# -------------------------------------- API -------------------------------------
# --------------------------------------------------------------------------------

# Création de la requête avec BaseModel de pydantic pour la validation de type
# et envoi des arguments à la fonction scoring
class User_input(BaseModel):
    loan_ID: str


app = FastAPI()


@app.get("/get_glossary/")
def get_glossary():
    """Fonction qui retourne la définition des variables au format.

    Arguments:
    --------------------------------

    return:
    --------------------------------
    glossary:json: Lexique"""
    glossary = pd.read_excel('Lexique.xlsx')
    return glossary.to_json(orient='index')


@app.get("/get_loans/")
def get_loans():
    """Fonction qui retourne la liste des demandes de prêts.

    Arguments:
    --------------------------------

    return:
    --------------------------------
    df:json: tableau des demandes de prêts"""

    # Lecture des données preprocessées
    df = pd.read_csv('df_light.csv', nrows=None)
    # Définition des features et de la target
    feat_lgb30 = ['SK_ID_CURR', 'TARGET', 'AGE', 'CODE_GENDER', 'NAME_EDUCATION_TYPE',
                  'YEARS_EMPLOYED', 'YEARS_ID_PUBLISH', 'YEARS_LAST_PHONE_CHANGE', 'REGION_POPULATION_RELATIVE',
                  'AMT_CREDIT', 'AMT_GOODS_PRICE', 'CREDIT_GOODS_PERC', 'CREDIT_DURATION', 'AMT_ANNUITY', 'DEBT_RATIO',
                  'PAYMENT_RATE', 'EXT_SOURCE_2', 'PREV_YEARS_DECISION_MEAN', 'PREV_PAYMENT_RATE_MEAN',
                  'INSTAL_DAYS_BEFORE_DUE_MEAN', 'INSTAL_PAYMENT_DIFF_MEAN', 'INSTAL_DAYS_PAST_DUE_MEAN',
                  'POS_MONTHS_BALANCE_MEAN', 'POS_CNT_INSTALMENT_FUTURE_MEAN', 'POS_NB_CREDIT',
                  'BURO_AMT_CREDIT_SUM_SUM', 'BURO_YEARS_CREDIT_ENDDATE_MAX', 'BURO_AMT_CREDIT_SUM_DEBT_SUM',
                  'BURO_YEARS_CREDIT_ENDDATE_MEAN', 'BURO_AMT_CREDIT_SUM_MEAN', 'BURO_CREDIT_ACTIVE_Active_SUM',
                  'BURO_AMT_CREDIT_SUM_DEBT_MEAN']
    df = df[feat_lgb30]
    df = df[df['NAME_EDUCATION_TYPE'] == 'Lower Secondary & Secondary']
    # OneHotEncoder sur nos variables catégorielles
    df, categ_feat = categories_encoder(df, nan_as_category=False)
    df.rename(columns={'NAME_EDUCATION_TYPE': 'NAME_EDUCATION_TYPE_Lower Secondary & Secondary'}, inplace=True)
    df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(str)
    df = df.reset_index(drop=True)
    df.index = df.index.map(str)
    return df.to_json(orient='index')


@app.get("/get_proba/{loan_ID}")
def get_proba(loan_ID):
    """Fonction qui charge la liste des demandes de prêts, effectue des
    transformations sur les features pour qu'elles correspondent à celles
    en entrée du modèle, filtre sur l'ID du prêt sélectionné par l'utilisateur
    dans le dashboard, standardise les données puis effectue la prédiction
    en appliquant le LightGBM.

    Arguments:
    --------------------------------
    loan_ID: str : ID du prêt

    return:
    --------------------------------
    proba: float: probabilité de faillite du client"""

    # Lecture des données preprocessées
    df = pd.read_csv('df_light.csv', nrows=None)
    # Définition des features et de la target
    feat_lgb30 = ['SK_ID_CURR', 'TARGET', 'CREDIT_DURATION', 'EXT_SOURCE_2', 'INSTAL_DAYS_PAST_DUE_MEAN',
                  'PAYMENT_RATE', 'POS_CNT_INSTALMENT_FUTURE_MEAN', 'CREDIT_GOODS_PERC',
                  'AGE', 'POS_NB_CREDIT', 'BURO_CREDIT_ACTIVE_Active_SUM', 'BURO_AMT_CREDIT_SUM_DEBT_MEAN',
                  'YEARS_EMPLOYED', 'YEARS_ID_PUBLISH', 'INSTAL_PAYMENT_DIFF_MEAN', 'BURO_AMT_CREDIT_SUM_MEAN',
                  'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'BURO_YEARS_CREDIT_ENDDATE_MEAN', 'AMT_CREDIT',
                  'YEARS_LAST_PHONE_CHANGE', 'POS_MONTHS_BALANCE_MEAN', 'INSTAL_DAYS_BEFORE_DUE_MEAN',
                  'BURO_AMT_CREDIT_SUM_DEBT_SUM', 'CODE_GENDER', 'PREV_YEARS_DECISION_MEAN',
                  'REGION_POPULATION_RELATIVE', 'DEBT_RATIO', 'BURO_AMT_CREDIT_SUM_SUM',
                  'BURO_YEARS_CREDIT_ENDDATE_MAX', 'NAME_EDUCATION_TYPE',
                  'PREV_PAYMENT_RATE_MEAN']
    df = df[feat_lgb30]
    df = df[df['NAME_EDUCATION_TYPE'] == 'Lower Secondary & Secondary']
    # OneHotEncoder sur nos variables catégorielles
    df, categ_feat = categories_encoder(df, nan_as_category=False)
    df.rename(columns={'NAME_EDUCATION_TYPE': 'NAME_EDUCATION_TYPE_Lower Secondary & Secondary'}, inplace=True)
    df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(str)
    df = df.reset_index(drop=True)
    df = df.drop(['TARGET'], axis=1)
    # Filtre sur l'ID du prêt
    ID_row = df[df['SK_ID_CURR'] == loan_ID]
    # Chargement du modèle et du scaler
    credit_score_model = load(open('credit_score_model_SHAP.sav', 'rb'))
    scaler = load(open('credit_score_model_scaler.sav', 'rb'))
    # Standardisation
    ID_row_scaled = scaler.transform(ID_row.drop(['SK_ID_CURR'], axis=1))
    # Prédiction
    proba = credit_score_model.predict_proba(ID_row_scaled)[:, 1]
    return proba.tolist()


@app.get("/get_feature_importance/")
def get_feature_importance():
    """Fonction qui retourne les scores d'importance
    des variables du modèle.

    Arguments:
    --------------------------------

    return:
    --------------------------------
    df_credit_score_model: json: tableau des features et score associé"""

    credit_score_model = load(open('credit_score_model_SHAP.sav', 'rb'))
    feat_imp_credit_score_model = credit_score_model.feature_importances_
    dic_credit_score_model = {'Features': ['CREDIT_DURATION', 'EXT_SOURCE_2', 'INSTAL_DAYS_PAST_DUE_MEAN',
                                           'PAYMENT_RATE', 'POS_CNT_INSTALMENT_FUTURE_MEAN', 'CREDIT_GOODS_PERC',
                                           'AGE', 'POS_NB_CREDIT', 'BURO_CREDIT_ACTIVE_Active_SUM',
                                           'BURO_AMT_CREDIT_SUM_DEBT_MEAN',
                                           'YEARS_EMPLOYED', 'YEARS_ID_PUBLISH', 'INSTAL_PAYMENT_DIFF_MEAN',
                                           'BURO_AMT_CREDIT_SUM_MEAN',
                                           'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'BURO_YEARS_CREDIT_ENDDATE_MEAN',
                                           'AMT_CREDIT',
                                           'YEARS_LAST_PHONE_CHANGE', 'POS_MONTHS_BALANCE_MEAN',
                                           'INSTAL_DAYS_BEFORE_DUE_MEAN',
                                           'BURO_AMT_CREDIT_SUM_DEBT_SUM', 'CODE_GENDER', 'PREV_YEARS_DECISION_MEAN',
                                           'REGION_POPULATION_RELATIVE', 'DEBT_RATIO', 'BURO_AMT_CREDIT_SUM_SUM',
                                           'BURO_YEARS_CREDIT_ENDDATE_MAX',
                                           'NAME_EDUCATION_TYPE_Lower Secondary & Secondary',
                                           'PREV_PAYMENT_RATE_MEAN'],
                              'Score': feat_imp_credit_score_model}
    df_credit_score_model = pd.DataFrame(data=dic_credit_score_model)
    df_credit_score_model = df_credit_score_model.sort_values(by='Score', ascending=False)
    return df_credit_score_model.to_json(orient='index')


# cd API_Heroku
# uvicorn main:app --reload  # pour récupérer la réponse de l'API (local)
# http://127.0.0.1:8000/docs  # pour tester l'API
