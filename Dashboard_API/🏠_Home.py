import streamlit as st
from PIL import Image

# --------------------------------------------------------------------------------
# --------------------- Configuration de la page ---------------------------------
# --------------------------------------------------------------------------------

st.set_page_config(page_title="PAD_Scoring_Client", page_icon="🏠", layout="wide", )

# --------------------------------------------------------------------------------
# ------------------------ Configuration texte -----------------------------------
# --------------------------------------------------------------------------------

st.markdown("""
                <style>
                .image_credit-font {
                    font-size:14px;
                }
                </style>
                """, unsafe_allow_html=True)

st.markdown("""
                <style>
                .text-font {
                    font-size:20px;
                    text-align: justify;
                }
                </style>
                """, unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# -------------------------------- Logo ------------------------------------------
# --------------------------------------------------------------------------------

#logo = Image.open('Logo.jpg')
#st.sidebar.image(logo, width=200)

# --------------------------------------------------------------------------------
# -------------------------------- Image -----------------------------------------
# --------------------------------------------------------------------------------

#col1 = st.columns(1)

#col1.image("Home.jpg")
#col1.markdown("<p class='image_credit-font'><em>Image credit: <br>Creator: User Noah Windler (@noahwindler) from "
#              "Unsplash <br>License: Do whatever you want. https://unsplash.com/license <br>URL: "
#              "https://unsplash.com/photos/gQI8BOaL69o</em></p>", unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# ---------------------------- Introduction --------------------------------------
# --------------------------------------------------------------------------------

st.title("Outil de scoring crédit")
st.markdown("<p class='text-font'>Prêt à dépenser propose des crédits à la consommation pour les personnes ayant "
              "peu ou pas du tout d'historique de prêt. <br><br>"
              "Cette  application permet de calculer la <strong>probabilité qu'un client rembourse son "
              "crédit</strong> puis classifie la demande en <strong>crédit accordé ou refusé.</strong> "
              "Il permet donc une <strong>totale transparence</strong> quant aux décisions d'octroi de "
              "crédit.</strong> <br><br>"
              "<strong>Fonctionnalités:</strong> <br>"
              "- Liste des demandes de prêts<br>"
              "- Calcul de la probabilité de faillite d un client<br>"
              "- Connaissance du client"
              "</p>",
              unsafe_allow_html=True)

# cd Dashboard_API
# streamlit run 🏠_Home.py
