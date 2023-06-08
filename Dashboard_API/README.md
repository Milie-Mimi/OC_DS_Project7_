# Projet 7 : Implémentez un modèle de scoring

## Découpage du dossier "Dashboard_API"

![Streamlit](streamlit_logo.PNG)

Ce dossier contient les fichiers ayant permis la réalisation du dashboard sur Streamlit (partie front-end).

**Pages du dashboard**
- 🏠_Home.py
- pages/1_📊_Demandes.py
- pages/2_✅_Scoring_client.py
- pages/3_📈_Profil_demandeur.py

**Configuration du dashboard**
- dossier .streamlit

**Images utilisées dans le dashboard**
- Home.jpg
- Logo.jpg

**Modèles sauvegardés pour affichage des features importance (globales et locales)**
- credit_score_model_SHAP.sav
- credit_score_model_SHAP_explainer.sav (non visible sur GitHub, trop volumineux)

**Versions des packages utilisés**
- requirements.txt

**Commandes pour démarrer l'application sur Heroku**
- setup.sh
- Procfile => fichier qui spécifie les commandes exécutées par l'application Heroku au démarrage