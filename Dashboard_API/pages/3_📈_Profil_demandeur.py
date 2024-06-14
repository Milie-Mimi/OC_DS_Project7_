from urllib.error import URLError

import pandas as pd
import streamlit as st
from PIL import Image
from matplotlib import pyplot as plt
import seaborn as sns
import requests

host = "https://api-scoring.onrender.com"

# --------------------------------------------------------------------------------
# --------------------- Configuration de la page ---------------------------------
# --------------------------------------------------------------------------------

st.set_page_config(page_title="Profil_demandeur", page_icon="📈", layout="wide", )

# --------------------------------------------------------------------------------
# ------------------------- Titre de la page -------------------------------------
# --------------------------------------------------------------------------------

st.title("Visualiser le profil du demandeur")

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
# ---------------------- Afficher la définition ----------------------------------
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


checkbox_profil = st.sidebar.checkbox("Afficher la définition")
if checkbox_profil:
    st.divider()
    st.markdown("<p class='text-font'><strong>Informations descriptives</strong> relatives au client sélectionné et "
                "comparaison aux clients défaillants et non défaillants. <br>"
                "<strong>Graphique de gauche:</strong> sélectionner la caractéristique à afficher pour une analyse "
                "univariée entre le client vs les clients défaillants et non défaillants. <br>"
                "<strong>Graphique de droite:</strong> sélectionner la 2ème caractéristique à afficher sur l'axe des "
                "abscisses pour une analyse bivariée entre le client vs les clients défaillants et non défaillants. "
                "<br><br>"
                "<strong>Liste des caractéristiques:</strong></p>",
                unsafe_allow_html=True)

    glossary = upload_glossary()
    st.dataframe(glossary, use_container_width=True)

# --------------------------------------------------------------------------------
# -------------------------------- Logo ------------------------------------------
# --------------------------------------------------------------------------------

#logo = Image.open('Logo.jpg')
#st.sidebar.image(logo, width=200)


# --------------------------------------------------------------------------------
# ----------------------- Chargement des données ---------------------------------
# --------------------------------------------------------------------------------


# Df en cache pour n'être chargé qu'une fois
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
    return df


try:
    # Chargement des données
    df = get_data()

    # --------------------------------------------------------------------------------
    # ----------------------------- Sélection ID -------------------------------------
    # --------------------------------------------------------------------------------

    select_ID = st.selectbox(
        "Sélectionner l'ID dans la liste pour accéder au profil du demandeur", list(df['SK_ID_CURR']))

    if not select_ID:

        st.error("Merci de rentrer un ID.")

    else:

        ID_row = df[df['SK_ID_CURR'] == select_ID]

        # --------------------------------------------------------------------------------
        # ----------------------- Informations du client ---------------------------------
        # --------------------------------------------------------------------------------

        st.divider()
        st.write(ID_row)
        st.divider()

        # --------------------------------------------------------------------------------
        # -------------------------- Choix des filtres -----------------------------------
        # --------------------------------------------------------------------------------

        # Séparation en 2 colonnes
        col1, col2 = st.columns(2)

        # Sélection d'une feature
        select_info = col1.selectbox(
            "Sélectionner la caractéristique à afficher", list(ID_row.drop(['SK_ID_CURR', 'TARGET', 'CODE_GENDER',
                                                                            'NAME_EDUCATION_TYPE_Lower Secondary & Secondary',
                                                                            'INSTAL_PAYMENT_DIFF_MEAN'],
                                                                           axis=1).columns))
        # Sélection d'une deuxième feature
        select_info_2 = col2.selectbox(
            "Sélectionner la 2ème caractéristique à afficher", list(ID_row.drop(['SK_ID_CURR', 'TARGET', 'CODE_GENDER',
                                                                                 'NAME_EDUCATION_TYPE_Lower Secondary & '
                                                                                 'Secondary',
                                                                                 'INSTAL_PAYMENT_DIFF_MEAN',
                                                                                 select_info],
                                                                                axis=1).columns))

        # --------------------------------------------------------------------------------
        # -------------------------- Analyse univariée -----------------------------------
        # --------------------------------------------------------------------------------

        # Dataframe pour le plot
        df_plot = {'ID_Client': [int(ID_row.loc[:, select_info]), int(ID_row.loc[:, select_info_2])],
                   'Défaillants': [df[df['TARGET'] == 1][select_info].mean(),
                                   df[df['TARGET'] == 1][select_info_2].mean()],
                   'Non défaillants': [df[df['TARGET'] == 0][select_info].mean(),
                                       df[df['TARGET'] == 0][select_info_2].mean()]}
        df_plot = pd.DataFrame(df_plot).T.reset_index()
        df_plot.rename(columns={'index': 'Clients_type',
                                0: select_info,
                                1: select_info_2}, inplace=True)

        # Barplot
        fig = plt.figure(figsize=(6, 3))
        sns.barplot(x='Clients_type', y=select_info, data=df_plot, palette=['#4c1130', 'tomato', 'powderblue'])
        plt.xlabel('', fontsize=12)
        plt.ylabel(select_info, fontsize=8)
        plt.title(f"{select_info} \n Comparaison du client à la moyenne par type de clients", fontsize=10)

        # Affichage barplot
        col1.pyplot(fig)

        # Distribution
        fig = plt.figure(figsize=(6, 3))
        sns.kdeplot(data=df, x=select_info, hue='TARGET', multiple="stack", palette=['powderblue', 'tomato'])
        plt.xlabel(select_info, fontsize=12)
        plt.ylabel("", fontsize=8)
        plt.title(f"{select_info} \n Distribution de la variable par type de clients", fontsize=10)
        plt.gca().legend().set_visible(False)

        # Affichage kdeplot
        col1.pyplot(fig)

        # Affichage données
        st.dataframe(df_plot, use_container_width=True)

        # --------------------------------------------------------------------------------
        # -------------------------- Analyse bivariée -----------------------------------
        # --------------------------------------------------------------------------------

        # Mapping pour sélection des clients à afficher
        df['TARGET'] = df['TARGET'].map({0: 'Non défaillants',
                                         1: 'Défaillants'})

        # Scatterplot
        color_dict = dict({'Défaillants': '#ff4040',
                           'Non défaillants': '#b0e0e6'})

        fig = plt.figure(figsize=(6, 3))
        sns.scatterplot(x=select_info_2, y=select_info, data=df,
                        hue='TARGET', palette=color_dict)
        plt.title('Affichage du client parmi les types de clients', fontsize=10)
        plt.xlabel(select_info_2, fontsize=8)
        plt.ylabel(select_info, fontsize=8)
        plt.scatter(df_plot[select_info_2][0], df_plot[select_info][0], c='#4c1130', marker='p', s=60,
                    label='ID sélectionné')
        plt.legend(title='Type de clients')
        col2.pyplot(fig)

        # Distribution
        fig = plt.figure(figsize=(6, 3))
        sns.kdeplot(data=df, x=select_info_2, hue='TARGET', multiple="stack", palette=['powderblue', 'tomato'])
        plt.xlabel(select_info_2, fontsize=12)
        plt.ylabel("", fontsize=8)
        plt.title(f"{select_info_2} \n Distribution de la variable par type de clients", fontsize=10)
        plt.gca().legend().set_visible(False)

        # Affichage kdeplot
        col2.pyplot(fig)

except URLError as e:
    st.error(
        """
        **This demo requires internet access.**
        Connection error: %s
    """
        % e.reason
    )
