# Projet 7 : Implémentez un modèle de scoring

## Découpage du dossier "API_Heroku"

![Fast_API](img_fast_API.PNG)

Ce dossier contient les fichiers ayant permis la réalisation de l'API via Fast_API (partie back_end) et les différentes 
données à charger dans le dashboard Streamlit.

**API**
- main.py

**Données à afficher et servant au calcul de la probabilité**
- Lexique.xlsx
- df_light.csv

**Modèles sauvegardés pour calcul de la prédiction**
- credit_score_model_SHAP.sav
- credit_score_model_scaler.sav

**Versions des packages utilisés**
- requirements.txt

**Commandes pour démarrer l'application sur Heroku**
- Procfile → fichier qui spécifie les commandes exécutées par l'application Heroku au démarrage