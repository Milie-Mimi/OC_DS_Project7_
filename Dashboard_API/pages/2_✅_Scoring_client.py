from pickle import load
import streamlit as st
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import seaborn as sns
from urllib.error import URLError
import plotly.graph_objects as go
import shap
from streamlit_shap import st_shap
import requests

host = "https://api-scoring.onrender.com"
# host = "https://api-scoring-app.herokuapp.com"

# --------------------------------------------------------------------------------
# --------------------- Configuration de la page ---------------------------------
# --------------------------------------------------------------------------------

st.set_page_config(page_title="Scoring_Client", page_icon="✅", layout="wide", )

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

st.markdown("""
                <style>
                .text-font_red {
                    color:#ff0000;
                    font-size:20px;
                    text-align: justify;
                }
                </style>
                """, unsafe_allow_html=True)

st.markdown("""
                <style>
                .text-font_green {
                    color:#116813;
                    font-size:20px;
                    text-align: justify;
                }
                </style>
                """, unsafe_allow_html=True)

st.markdown("""
                <style>
                .text-font_title {
                    font-size:30px;
                    text-align:center
                }
                </style>
                """, unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# ------------------------- Titre de la page -------------------------------------
# --------------------------------------------------------------------------------

st.markdown("# Scoring client")


# --------------------------------------------------------------------------------
# ---------------------- Afficher  la définition ---------------------------------
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


checkbox_scoring = st.sidebar.checkbox("Afficher la définition")
if checkbox_scoring:
    st.divider()
    st.markdown("<p class='text-font'>Outil de scoring crédit pour calculer la <strong>probabilité</strong> de faillite"
                " du client et classifier sa demande de prêt en <strong>crédit <span "
                "class='text-font_green'>accordé</span> ou <span class='text-font_red'>refusé</span></strong>.<br>"
                "Divers graphiques viennent aider à la compréhension du score en montrant comment les "
                "caractéristiques du client contribuent à la prédiction finale."
                "<br><br>"
                "<strong>Liste des caractéristiques:</strong></p></p>",
                unsafe_allow_html=True)
    glossary = upload_glossary()
    st.dataframe(glossary, use_container_width=True)

# --------------------------------------------------------------------------------
# -------------------------------- Logo ------------------------------------------
# --------------------------------------------------------------------------------

#logo = Image.open('Logo.jpg')
#st.sidebar.image(logo, width=200)


# --------------------------------------------------------------------------------
# ---------------- Chargement des données et sélection ID-------------------------
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
    df = df.drop(['TARGET'], axis=1)
    return df


try:
    st.divider()
    # Chargement des données avec features ayant servi à la modélisation
    df = get_data()

    # Sélection de l'ID du prêt
    loan_ID = st.text_input("🆔 Renseigner l'ID du prêt", '', key="select_id", max_chars=6)

    if not loan_ID:
        st.error("Merci de rentrer un ID")

    # --------------------------------------------------------------------------------
    # ---------------- Filtre sur ID et affichage du dataframe------------------------
    # --------------------------------------------------------------------------------

    else:

        # Filtre sur l'ID du prêt
        ID_row = df[df['SK_ID_CURR'] == loan_ID]
        ID_row_index = ID_row.index
        # Affichage de la ligne
        st.divider()
        st.write("### Informations utilisées dans le calcul du scoring", ID_row)
        st.divider()

        # --------------------------------------------------------------------------------
        # ---------------- Calcul probabilité de faillite du client-----------------------
        # --------------------------------------------------------------------------------

        # Lors du click sur le bouton, appel à l'API de prédiction
        if st.button('Lancer la prédiction'):
            # res = requests.get(url="http://127.0.0.1:8000/get_proba", data=id_client)
            res = requests.get(host + f"/get_proba/{loan_ID}")
            response = res.json()[0]

            # Index du prêt choisi
            idx = ID_row.index[0]

            # Séparation page en 2 colonnes
            col1, col2 = st.columns(2)

            # --------------------------------------------------------------------------------
            # --------------------------- Jauge et explications ------------------------------
            # --------------------------------------------------------------------------------

            # Si la probabilité de défaillance est supérieure au seuil de 0.47, classement défaillant
            if response > 0.47:
                # Jauge
                gauge = go.Figure(go.Indicator(mode="gauge+number+delta",
                                               value=float(response) * 100,
                                               domain={'x': [0, 1], 'y': [0, 1]},
                                               delta={'reference': 47, 'increasing': {'color': "red"}},
                                               gauge={'axis': {'range': [None, 100]},
                                                      'bar': {'color': "red"},
                                                      'steps': [
                                                          {'range': [0, 40], 'color': "lightgrey"},
                                                          {'range': [40, 60], 'color': "gray"},
                                                          {'range': [60, 100], 'color': "dimgray"}],
                                                      'threshold': {
                                                          'line': {'color': "red", 'width': 4},
                                                          'thickness': 0.75,
                                                          'value': 47}}))
                gauge.update_layout(font={'size': 18})

                # Affichage jauge
                col1.markdown(
                    "<p class='text-font_title'><strong>Probabilité de faillite du client (%)</strong></p>",
                    unsafe_allow_html=True)
                col1.plotly_chart(gauge, use_container_width=True)

                # Classement du client et explications
                col2.markdown("<p class='text-font'>Risque élevé de défaillance. "
                              "<br>Proposition de classement du client: <span "
                              "class='text-font_red'> <strong>Défaillant</strong> </span> <br>Demande de prêt <span "
                              "class='text-font_red'> <strong>Refusée</strong> </span>.<br><br>La demande de prêt n° "
                              "{id} a une probabilité de {prob}% d'être défaillante. Le seuil de classement en client "
                              "défaillant se situe à 47%. <br><br>Ce seuil a été défini pour que le modèle de "
                              "prédiction soit capable de détecter efficacement les vrais positifs (clients "
                              "défaillants prédits défaillants) mais sans trop augmenter les faux positifs (clients "
                              "non défaillants) prédits défaillants.</p>".format(id=loan_ID,
                                                                                 prob=round(float(response) * 100, 2)),
                              unsafe_allow_html=True)

            # Si la probabilité de défaillance est inférieure au seuil de 0.47, classement non défaillant
            else:

                # Jauge
                gauge = go.Figure(go.Indicator(mode="gauge+number+delta",
                                               value=float(response) * 100,
                                               domain={'x': [0, 1], 'y': [0, 1]},
                                               delta={'reference': 47, 'decreasing': {'color': "#116813"}},
                                               gauge={'axis': {'range': [None, 100]},
                                                      'bar': {'color': "#116813"},
                                                      'steps': [
                                                          {'range': [0, 40], 'color': "lightgrey"},
                                                          {'range': [40, 60], 'color': "gray"},
                                                          {'range': [60, 100], 'color': "dimgray"}],
                                                      'threshold': {
                                                          'line': {'color': "red", 'width': 4},
                                                          'thickness': 0.75,
                                                          'value': 47}}))
                gauge.update_layout(font={'size': 18})

                # Affichage jauge
                col1.markdown(
                    "<p class='text-font_title'><strong>Probabilité de faillite du client (%)</strong></p>",
                    unsafe_allow_html=True)
                col1.plotly_chart(gauge, use_container_width=True)

                # Classement du client et explications
                col2.markdown("<p class='text-font'>Risque faible de défaillance. "
                              "<br>Proposition de classement du client: <span class='text-font_green'> <strong>Non "
                              "défaillant</strong> </span> <br>Demande de prêt :<span class='text-font_green'> "
                              "<strong>Acceptée</strong> </span>.<br><br>La demande de prêt n° {id} a une probabilité "
                              "de {prob}% d'être défaillante. Le seuil de classement en client défaillant se situe à "
                              "47%. <br><br>Ce seuil a été défini pour que le modèle de prédiction soit capable de "
                              "détecter efficacement les vrais positifs (clients défaillants prédits défaillants) "
                              "mais sans trop augmenter les faux positifs (clients non défaillants) prédits "
                              "défaillants.</p>".format(id=loan_ID, prob=round(float(response) * 100, 2)),
                              unsafe_allow_html=True)

            col2.markdown("<p class='text-font'> <em>❗ Attention, il s'agit d'une proposition de classement "
                          "automatique basée sur le risque de défaillance du client mais qui n'exclut en rien l'étude "
                          "attentive du dossier par le conseiller clientèle</em>.</p>",
                          unsafe_allow_html=True)

            st.divider()

            # --------------------------------------------------------------------------------
            # ------------------------- Features importance globales--------------------------
            # --------------------------------------------------------------------------------

            st.write("### Compréhension du modèle")

            st.markdown("<p class='text-font'>Visualiser l'importance des caractéristiques permet de mieux comprendre "
                        "quelles sont les <strong>features en entrée du modèle qui ont le plus d'utilité dans la "
                        "prédiction de la variable cible. </strong>"
                        "Dans un modèle de prédiction, l'importance de features permet d'avoir une meilleure "
                        "compréhension des données et du modèle et offre une base pour la réduction de "
                        "dimension, permettant la sélection de caractéristiques qui peuvent améliorer le "
                        "modèle.</p>",
                        unsafe_allow_html=True)

            # Appel à l'API pour obtenir les features importance globales
            res = requests.get(host + f"/get_feature_importance/")
            response = res.json()
            df_credit_score_model = pd.read_json(response, orient='index')

            fig = plt.figure(figsize=(15, 12))
            plt.title('Variables du modèle et leur importance', fontsize=18)
            plt.xticks(rotation=45, ha='right')
            sns.barplot(data=df_credit_score_model, x='Score', y='Features', palette='rocket')

            # Affichage barplot
            st.pyplot(fig, clear_figure=True)

            st.divider()

            # --------------------------------------------------------------------------------
            # ------------------------ Features importance locales ---------------------------
            # --------------------------------------------------------------------------------

            st.write("### Compréhension du score de l'ID sélectionné")

            # Dataframe pour shap (ordre des features correspond à l'ordre donné en entrée du modèle)
            df_shap = df[['CREDIT_DURATION', 'EXT_SOURCE_2', 'INSTAL_DAYS_PAST_DUE_MEAN',
                          'PAYMENT_RATE', 'POS_CNT_INSTALMENT_FUTURE_MEAN', 'CREDIT_GOODS_PERC',
                          'AGE', 'POS_NB_CREDIT', 'BURO_CREDIT_ACTIVE_Active_SUM', 'BURO_AMT_CREDIT_SUM_DEBT_MEAN',
                          'YEARS_EMPLOYED', 'YEARS_ID_PUBLISH', 'INSTAL_PAYMENT_DIFF_MEAN', 'BURO_AMT_CREDIT_SUM_MEAN',
                          'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'BURO_YEARS_CREDIT_ENDDATE_MEAN', 'AMT_CREDIT',
                          'YEARS_LAST_PHONE_CHANGE', 'POS_MONTHS_BALANCE_MEAN', 'INSTAL_DAYS_BEFORE_DUE_MEAN',
                          'BURO_AMT_CREDIT_SUM_DEBT_SUM', 'CODE_GENDER', 'PREV_YEARS_DECISION_MEAN',
                          'REGION_POPULATION_RELATIVE', 'DEBT_RATIO', 'BURO_AMT_CREDIT_SUM_SUM',
                          'BURO_YEARS_CREDIT_ENDDATE_MAX', 'NAME_EDUCATION_TYPE_Lower Secondary & Secondary',
                          'PREV_PAYMENT_RATE_MEAN']]

            # Chargement du scaler et du modèle LightGBM
            scaler = load(open('credit_score_model_scaler.sav', 'rb'))
            credit_score_model = load(open('credit_score_model_SHAP.sav', 'rb'))

            # Standardisation des données
            scaled_df_shap = scaler.transform(df_shap)

            # Sélection de l'index de la ligne du df correspondant au prêt sélectionné
            subsampled_scaled_df_shap = scaled_df_shap[idx].reshape(1, -1)

            # Création de l'explainer
            explainer = shap.TreeExplainer(credit_score_model)

            # Calcul des shap_values de la ligne sélectionnée
            shap_values = explainer.shap_values(subsampled_scaled_df_shap)

            # Décision plot
            # https://shap-lrjball.readthedocs.io/en/latest/generated/shap.decision_plot.html
            fig1 = plt.figure()
            plt.title("Chemin suivi pour l'obtention du score", fontsize=18)
            st_shap(shap.decision_plot(base_value=explainer.expected_value[1],
                                       shap_values=shap_values[1],
                                       features=subsampled_scaled_df_shap,
                                       feature_names=df_shap.columns.tolist(),
                                       link='logit',
                                       feature_display_range=slice(-1, -31, -1)),
                    width=1200,
                    height=800)

            expander = st.expander("Interprétation du decision plot")
            expander.markdown("<p class='text-font'>Le graphique ci-dessus est appelé un decision plot. Il permet de "
                              "visualiser <strong>comment un modèle complexe arrive à sa prédiction.</strong> "
                              "<br>L'axe des x représente la prédiction et l'axe y les caractéristiques par ordre "
                              "décroissant d'importance. En se déplaçant du bas vers le haut, les valeurs de "
                              "chaque caractéristique sont ajoutées à la valeur de base du modèle. Cela montre"
                              " comment chaque caractéristiques contribue à la prédiction.<br><br>Ce diagramme a "
                              "l'avantage de montrer clairement un grand nombre de caractéristiques (les 30 utilisées "
                              "par le modèle) et de visualiser l'effet cumulatif des intéractions.</p>",
                              unsafe_allow_html=True)

            # Force plot
            # https: // shap.readthedocs.io / en / latest / generated / shap.plots.force.html
            st_shap(shap.force_plot(base_value=explainer.expected_value[1],
                                    shap_values=shap_values[1],
                                    features=subsampled_scaled_df_shap,
                                    feature_names=df_shap.columns,
                                    out_names='Prob Défaillance',
                                    link='logit'))  # pour avoir les probabilités

            expander = st.expander("Interprétation du force plot")
            expander.markdown("<p class='text-font'>Le graphique ci-dessus est appelé un force plot. Il montre les "
                              "caractéristiques qui contribuent à pousser la prédiction à partir de la valeur de "
                              "base (sortie moyenne du modèle sur l'ensemble des données d'entrainement). <br>Les "
                              "caractéristiques qui poussent la prédiction vers le haut sont indiquées en rouge / à "
                              "gauche. Les caractéristiques qui la font baisser apparaissent en bleu / à droite.</p>",
                              unsafe_allow_html=True)

except URLError as e:
    st.error(
        """
        **This demo requires internet access.**
        Connection error: %s
    """
        % e.reason
    )
