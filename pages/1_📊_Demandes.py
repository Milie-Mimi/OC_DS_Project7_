import pandas as pd
import streamlit as st
from PIL import Image
from fct_model import data_app

# --------------------------------------------------------------------------------
# --------------------- Configuration de la page ---------------------------------
# --------------------------------------------------------------------------------

# Configuration de la page
st.set_page_config(page_title="Liste_demandeurs", page_icon="📊", layout="wide", )

# --------------------------------------------------------------------------------
# ------------------------ Configuration texte -----------------------------------
# --------------------------------------------------------------------------------

st.markdown("""
                <style>
                .text-font {
                    font-size:20px;
                    text-align: justify;
                }
                </style>
                """, unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# ------------------------- Titre de la page -------------------------------------
# --------------------------------------------------------------------------------

# Titre de la page
st.title("Liste des demandes de prêt")

# --------------------------------------------------------------------------------
# ---------------------------- Définition ----------------------------------------
# --------------------------------------------------------------------------------

@st.cache_data
def upload_glossary():
    glossary = pd.read_excel('data/Lexique.xlsx')
    return glossary

# Définition
checkbox_app = st.sidebar.checkbox("Afficher les définitions")
if checkbox_app:
    st.divider()
    st.markdown("<p class='text-font'>Liste des demandes de prêt issues de l'outil <strong>Application</strong>. "
                "Une ligne représente un prêt."
                "<br><br>"
                "<strong>Liste des caractéristiques:</strong></p></p></p>", unsafe_allow_html=True)
    glossary = upload_glossary()
    st.dataframe(glossary, use_container_width=True)

# --------------------------------------------------------------------------------
# -------------------------------- Logo ------------------------------------------
# --------------------------------------------------------------------------------

logo = Image.open('Logo.jpg')
st.sidebar.image(logo, width=200)


# --------------------------------------------------------------------------------
# ------------------------------- KPIs -------------------------------------------
# --------------------------------------------------------------------------------

# Df en cache pour n'être chargé qu'une fois
@st.cache_data
def get_data(nrows):
    df = data_app(nrows)
    df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(str)
    df = df.reset_index(drop=True)
    df.index = df.index.map(str)
    df = df[['SK_ID_CURR', 'TARGET', 'AGE', 'CODE_GENDER', 'NAME_EDUCATION_TYPE_Lower Secondary & Secondary',
             'YEARS_EMPLOYED', 'YEARS_ID_PUBLISH', 'YEARS_LAST_PHONE_CHANGE', 'REGION_POPULATION_RELATIVE',
             'AMT_CREDIT', 'AMT_GOODS_PRICE', 'CREDIT_GOODS_PERC', 'CREDIT_DURATION', 'AMT_ANNUITY', 'DEBT_RATIO',
             'PAYMENT_RATE', 'EXT_SOURCE_2', 'PREV_YEARS_DECISION_MEAN', 'PREV_PAYMENT_RATE_MEAN',
             'INSTAL_DAYS_BEFORE_DUE_MEAN', 'INSTAL_PAYMENT_DIFF_MEAN', 'INSTAL_DAYS_PAST_DUE_MEAN',
             'POS_MONTHS_BALANCE_MEAN', 'POS_CNT_INSTALMENT_FUTURE_MEAN', 'POS_NB_CREDIT', 'BURO_AMT_CREDIT_SUM_SUM',
             'BURO_YEARS_CREDIT_ENDDATE_MAX', 'BURO_AMT_CREDIT_SUM_DEBT_SUM', 'BURO_YEARS_CREDIT_ENDDATE_MEAN',
             'BURO_AMT_CREDIT_SUM_MEAN', 'BURO_CREDIT_ACTIVE_Active_SUM', 'BURO_AMT_CREDIT_SUM_DEBT_MEAN']]
    return df


# Chargement des données
df = get_data(nrows=None)

# Calcul des KPIs
demandes = df['SK_ID_CURR'].count()
demandes = f"{demandes:,}"
montant = round(df['AMT_CREDIT'].mean(), 0)
montant = f"{montant:,}"
annuite = round(df['AMT_ANNUITY'].mean(), 0)
annuite = f"{annuite:,}"
endettement = df['DEBT_RATIO'].mean()
endettement = f"{endettement:.2%}"
duree = round(df['CREDIT_DURATION'].mean(), 0)

# Affichage des KPIs
st.divider()
st.subheader("KPIs sur les demandes de prêt")
st.markdown('<p <br><br> </p>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Nombre de demandes", demandes, help='Nombre total de demandes de prêt')
col2.metric("Montant moyen", montant, help='Montant moyen des prêts')
col3.metric("Annuité moyenne", annuite, help='Annuité moyenne des prêts')
col4.metric("Taux d'endettement moyen", endettement, help='Taux d endettement des prêts')
col5.metric("Durée moyenne en année", duree, help='Durée moyenne des prêts')
st.divider()

# Affichage des demandes de prêt
st.dataframe(df.drop('TARGET', axis=1))
