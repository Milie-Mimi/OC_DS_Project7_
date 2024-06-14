import pandas as pd
import streamlit as st
from PIL import Image
import requests

host = "https://api-scoring.onrender.com"

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
    """Fonction qui appelle l'API et retourne la définition des variables au format
    dataframe.

    Arguments:
    --------------------------------

    return:
    --------------------------------
    res:dataframe: Lexique"""

    res = requests.get(host + f"/get_glossary/")
    res = res.json()
    res = pd.read_json(res, orient='index')
    return res


@st.cache_data
def get_data():
    """Fonction qui appelle l'API et retourne la liste des demandes de
    prêts au format dataframe.

    Arguments:
    --------------------------------

    return:
    --------------------------------
    df:dataframe: tableau des demandes de prêts"""

    res = requests.get(host + f"/get_loans/")
    response = res.json()
    df = pd.read_json(response, orient='index')
    df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(str)
    df = df.drop('TARGET', axis=1)
    return df


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

#logo = Image.open('Logo.jpg')
#st.sidebar.image(logo, width=200)


# --------------------------------------------------------------------------------
# ------------------------------- KPIs -------------------------------------------
# --------------------------------------------------------------------------------

# Chargement des données
df = get_data()

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
st.dataframe(df)
