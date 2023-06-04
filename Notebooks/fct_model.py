# Fonctions pour l'entrainement et l'évaluation des modèles

import time
from pickle import dump
import dill
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from lime import lime_tabular
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from Notebooks import fct_eda as eda
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate, train_test_split, KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, fbeta_score, make_scorer
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as pipe
import shap


# --------------------------------------------------------------------
# ------------------------- ENTRAINEMENT -----------------------------
# --------------------------------------------------------------------

def pipeline_model(model, preprocessor):
    # Définition de la pipeline du modèle : étapes de preprocessing + classifier
    pipeline_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)])

    return pipeline_model


def pipeline_model_balanced(model, preprocessor,
                            oversampling_strategy, undersampling_strategy):
    # Sur échantillonnage de la classe minoritaire (10% de la classe majoritaire ~= 23000)
    oversampler = SMOTE(sampling_strategy=oversampling_strategy, random_state=42)

    # Sous échantillonnage pour réduire la classe majoritaire (50% de plus que la classe minoritaire ~= 46000
    undersampler = RandomUnderSampler(sampling_strategy=undersampling_strategy, random_state=42)

    # Transformations à effectuer sur nos variables
    # preprocessor = ColumnTransformer(transformers=[
    #    ('num', numeric_transformer, numeric_features)],
    #                                 remainder='passthrough')

    # Définition de la pipeline du modèle : étapes de preprocessing + classifier
    pipeline_model_balanced = pipe(steps=[
        ('over', oversampler),
        ('under', undersampler),
        ('preprocessor', preprocessor),
        ('classifier', model)])

    return pipeline_model_balanced


def compar_train_val_scores(model, cv, xtrain, ytrain, scoring):
    # Cross validation
    cv_results = cross_validate(model,
                                xtrain, ytrain,
                                cv=cv,
                                scoring=scoring,
                                return_train_score=True,
                                return_estimator=True, n_jobs=2)
    cv_results = pd.DataFrame(cv_results)

    # Distribution des erreurs d'entrainement et de test via cross-validation
    scores = pd.DataFrame()
    scores[["train_error", "validation_error"]] = cv_results[["train_score", "test_score"]]

    scores.plot.hist(bins=50, edgecolor="black", figsize=(8, 4))
    plt.xlabel("Betascore")
    _ = plt.title("Distribution du fbeta_score sur les données d'entrainement et de \n validation via cross-validation")
    plt.show()

    # Moyenne et écart type moyen des erreurs sur les différentes cross validation
    mean_score_train = scores['train_error'].mean()
    std_score_train = scores['train_error'].std()
    mean_score_validation = scores['validation_error'].mean()
    std_score_validation = scores['validation_error'].std()

    print(f"Betascore train set: {mean_score_train:.4f} +/- {std_score_train:.4f}")
    print(f"Betascore validation set: {mean_score_validation:.4f} +/- {std_score_validation:.4f}")


def optimize_and_train_model(pipeline_model, xtrain, ytrain, params, scoring, cv):
    _ = pipeline_model.fit(xtrain, ytrain)

    # Réglage automatique des meilleurs hyperparamètres avec GridSearchCV
    model_grid_cv = GridSearchCV(pipeline_model,
                                 param_grid=params,
                                 cv=cv,
                                 scoring=scoring,
                                 refit=True)

    model_grid_cv.fit(xtrain, ytrain)

    # Outer cross-validation
    # outer_cv = KFold(n_splits = 10, shuffle = True, random_state = 42)
    # compar_train_val_scores(model= model_grid_cv,
    #                        cv = outer_cv, 
    #                        xtrain = xtrain, 
    #                        ytrain = ytrain,
    #                        scoring = scoring)

    best_model = model_grid_cv.best_estimator_
    best_params = model_grid_cv.best_params_

    return best_model, best_params


def optimize_and_train_model_RSCV(pipeline_model, xtrain, ytrain, params, scoring, cv):
    _ = pipeline_model.fit(xtrain, ytrain)

    # Réglage automatique des meilleurs hyperparamètres avec GridSearchCV
    model_grid_cv = RandomizedSearchCV(pipeline_model,
                                       param_distributions=params,
                                       cv=cv,
                                       scoring=scoring,
                                       refit=True)

    model_grid_cv.fit(xtrain, ytrain)

    # Outer cross-validation
    # outer_cv = KFold(n_splits = 10, shuffle = True, random_state = 42)
    # compar_train_val_scores(model= model_grid_cv,
    #                        cv = outer_cv, 
    #                        xtrain = xtrain, 
    #                        ytrain = ytrain,
    #                        scoring = scoring)

    best_model = model_grid_cv.best_estimator_
    best_params = model_grid_cv.best_params_

    return best_model, best_params


def best_model(model_name, model, cv,
               xtrain, ytrain,
               preprocessor, params,
               scoring, xtest, ytest,
               oversampling_strategy=0.1, undersampling_strategy=0.5, balanced=False,
               Randomized=False):
    if not balanced:
        start = time.time()
        model = pipeline_model(model=model,
                               preprocessor=preprocessor)

        if not RandomizedSearchCV:
            # Optimisation via cross validation & GridSearch
            best_model, best_params = optimize_and_train_model(pipeline_model=model,
                                                               xtrain=xtrain,
                                                               ytrain=ytrain,
                                                               params=params,
                                                               scoring=scoring,
                                                               cv=cv)

        else:
            # Optimisation via cross validation & RandomizedSearch
            best_model, best_params = optimize_and_train_model_RSCV(pipeline_model=model,
                                                                    xtrain=xtrain,
                                                                    ytrain=ytrain,
                                                                    params=params,
                                                                    scoring=scoring,
                                                                    cv=cv)

        duration = time.time() - start

    else:

        start = time.time()
        model = pipeline_model_balanced(model=model,
                                        preprocessor=preprocessor,
                                        oversampling_strategy=oversampling_strategy,
                                        undersampling_strategy=undersampling_strategy)

        if not RandomizedSearchCV:
            # Optimisation via cross validation & GridSearch
            best_model, best_params = optimize_and_train_model(pipeline_model=model,
                                                               xtrain=xtrain,
                                                               ytrain=ytrain,
                                                               params=params,
                                                               scoring=scoring,
                                                               cv=cv)

        else:
            best_model, best_params = optimize_and_train_model_RSCV(pipeline_model=model,
                                                                    xtrain=xtrain,
                                                                    ytrain=ytrain,
                                                                    params=params,
                                                                    scoring=scoring,
                                                                    cv=cv)

        duration = time.time() - start

    return best_model, best_params, model_name, duration


# --------------------------------------------------------------------
# -------------------------- EVALUATION ------------------------------
# --------------------------------------------------------------------            

def score_metier(ytest, y_pred):
    # Matrice de confusion transformée en array avec affectation aux bonnes catégories
    (vn, fp, fn, vp) = confusion_matrix(ytest, y_pred).ravel()

    # Rappel avec action fp → à minimiser
    score_metier = 10 * fn + fp

    return score_metier


def eval_metrics(best_model, xtest, ytest, beta_value):
    y_pred = best_model.predict(xtest)

    score_biz = score_metier(ytest, y_pred)
    betascore = fbeta_score(ytest, y_pred, beta=beta_value)
    recall = recall_score(ytest, y_pred)
    precision = precision_score(ytest, y_pred, zero_division=0)
    accuracy = accuracy_score(ytest, y_pred)
    auc = roc_auc_score(ytest, y_pred)

    print(f'Score métier: {score_biz}')
    print(f'Beta score: {betascore}')
    print(f'Recall: {recall}')
    print(f'Precision: {precision}')
    print(f'Accuracy: {accuracy}')
    print(f'AUC: {auc}')

    return score_biz, betascore, recall, precision, accuracy, auc, y_pred


def matrice_confusion(ytest, ypred, model_name):
    plt.figure(figsize=(5, 5))
    cm = confusion_matrix(ytest, ypred)
    sns.heatmap(cm,
                xticklabels=['Y=0 (Non défaillant)', 'Y=1 (Défaillant)'],
                yticklabels=['Y=0 (Non défaillant)', 'Y=1 (Défaillant)'],
                annot=True,
                fmt='d',
                linewidth=.5,
                cmap=sns.cubehelix_palette(as_cmap=True), cbar=False)
    plt.title(f'Matrice de confusion: {model_name}')
    plt.ylabel('Réalité')
    plt.xlabel('Prédiction')
    plt.show()


# --------------------------------------------------------------------
# ---------- DONNEES POUR ENTRAINEMENT ET EVALUATION -----------------
# --------------------------------------------------------------------
def data_train_test(df):
    """Fonction qui applique le OneHotEncoder sur les données catégorielles du dataframe en entrée,
    le divise en jeux d'entrainement et de test et applique une première sélection de features (variance
    threshold + SelectKBest). Ce sont ces données qui serviront à l'entrainement des modèles de scoring.

    Arguments:
    --------------------------------
    df: dataframe: tableau issu de la consolidation des différents datasets ne contenant
    pas de valeurs manquantes, obligatoire

    return:
    --------------------------------
    X_train: données pour entrainer le modèle de scoring
    X_test : données pour tester le modèle de scoring
    y_train : variable à prédire (défaillant, non défaillant) des données d'entrainement
    y_test : variable à prédire (défaillant, non défaillant) des données de test"""

    # Split des données en jeux de train et test

    # Définition des features et de la target
    col_X = [f for f in df.columns if f not in ['TARGET']]
    X = df[col_X]
    y = df['TARGET']

    # Liste des variables quantitatives
    num_feat = X.select_dtypes(exclude='object').columns.tolist()

    # OneHotEncoder sur les variables catégorielles
    X, categ_feat = eda.categories_encoder(X, nan_as_category=False)

    # Jeu d'entrainement (80%) et de validation (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    # Features selection

    # VarianceThreshold (suppression variables sans variance)
    transform = VarianceThreshold(0)
    X_train_trans = transform.fit_transform(X_train)
    X_test_trans = transform.fit_transform(X_test)
    mask = transform.get_support()
    feat_suppr = X.columns[~mask].tolist()
    # Liste des variables catégorielles actualisée
    categ_feat = [elem for elem in categ_feat if elem not in feat_suppr]
    # Liste des variables numériques actualisée
    num_feat = [elem for elem in num_feat if elem not in feat_suppr]
    # Nouveaux df
    X_train = X_train[num_feat + categ_feat]
    X_test = X_test[num_feat + categ_feat]

    # SelectKBest
    # Caractéristiques qualitatives
    fs_categ = SelectKBest(score_func=chi2, k=41)
    fs_categ.fit(X_train[categ_feat], y_train)
    col_quali_keep = fs_categ.get_feature_names_out().tolist()
    # Liste des variables catégorielles actualisée
    categ_feat = col_quali_keep
    # Nouveaux df
    X_train = X_train[num_feat + categ_feat]
    X_test = X_test[num_feat + categ_feat]

    # Caractéristiques quantitatives
    fs_num = SelectKBest(score_func=f_classif, k=203)
    fs_num.fit(X_train[num_feat], y_train)
    col_quanti_keep = fs_num.get_feature_names_out().tolist()
    # Liste des variables catégorielles actualisée
    num_feat = col_quanti_keep
    # Nouveaux df
    X_train = X_train[num_feat + categ_feat]
    X_test = X_test[num_feat + categ_feat]

    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------
# ------------------------ MODELISATIONS -----------------------------
# --------------------------------------------------------------------

def DummyClassifier_model():
    """Fonction qui utilise un pipeline de transformation des données et applique
    un DummyClassifier pour effectuer la prédiction. Le pipeline est ensuite optimisé
    et entrainé en fonction du scoring f betascore (avec beta = 9) et d'une grille de paramètres.
    Le modèle, ses paramètres et son évaluation sont trackés et stockés sur MLFlow.

    Arguments:
    --------------------------------

    return:
    --------------------------------
    dummy: le modèle DummyClassifier entrainé"""

    # Récupération des données preprocessées
    df = pd.read_csv('../Dashboard_API/df.csv', nrows=None)

    # Train test split
    X_train, X_test, y_train, y_test = data_train_test(df)

    mlflow.set_experiment("mlflow-default-risk")
    experiment = mlflow.get_experiment_by_name("mlflow-default-risk")

    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name='Baseline - Dummy classifier'):
        (dummy, dummy_params,
         dummy_name, dummy_duration) = best_model(model_name='Baseline - Dummy classifier',
                                                  model=DummyClassifier(random_state=42),
                                                  cv=KFold(n_splits=5,
                                                           shuffle=True,
                                                           random_state=42),
                                                  xtrain=X_train,
                                                  ytrain=y_train,
                                                  preprocessor=StandardScaler(),
                                                  params={'classifier__strategy': ['most_frequent', 'prior',
                                                                                   'stratified',
                                                                                   'uniform'], },
                                                  scoring=make_scorer(fbeta_score, beta=9),
                                                  xtest=X_test,
                                                  ytest=y_test,
                                                  oversampling_strategy=0.1,
                                                  undersampling_strategy=0.5,
                                                  balanced=False,
                                                  Randomized=False)

        # Evaluation sur les données de test
        (biz_dummy, beta_dummy, rec_dummy, prec_dummy,
         acc_dummy, auc_dummy, y_pred_dummy) = eval_metrics(best_model=dummy,
                                                            xtest=X_test,
                                                            ytest=y_test,
                                                            beta_value=9)

        # log des paramètres et scores à chaque fois que le modèle est lancé
        mlflow.log_param("best parameters", dummy_params)

        mlflow.log_metric("score metier", biz_dummy)
        mlflow.log_metric("f betascore", beta_dummy)
        mlflow.log_metric("recall", rec_dummy)
        mlflow.log_metric("precision", prec_dummy)
        mlflow.log_metric("accuracy", acc_dummy)
        mlflow.log_metric("auc", auc_dummy)
        mlflow.log_metric("Tps_entrainement", dummy_duration)

        mlflow.sklearn.log_model(dummy, "dummyclassifier")

        return dummy


def RegLog_model():
    """Fonction qui utilise un pipeline de transformation des données et applique
    une régression logistique pour effectuer la prédiction. Le pipeline est optimisé
    et entrainé en fonction du scoring f betascore (avec beta = 9), cross validation et d'une grille de paramètres.
    Plusieurs versions du modèle seront testées :
    - sur données non équilibrées
    - sur données rééquilibrées avec class weight
    - sur données rééquilibrées avec sous échantillonnage puis SMOTE

    Le modèle, ses paramètres et les metrics servant à son évaluation sont trackés et stockés sur MLFlow.

    Arguments:
    --------------------------------

    return:
    --------------------------------
    reglog, reglog_cw, reglog_smote1050, reglog_smote2030, reglog_smote1060 :
    les différents modèles entrainés et optimisés"""

    # Régression logistique sur données non équilibrées

    # Récupération des données preprocessées
    df = pd.read_csv('../Dashboard_API/df.csv', nrows=None)

    # Train test split
    X_train, X_test, y_train, y_test = data_train_test(df)

    # Pipeline, optimisation et entrainement
    mlflow.set_experiment("mlflow-default-risk")
    experiment = mlflow.get_experiment_by_name("mlflow-default-risk")

    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name='Régression Logistique'):
        (reglog, reglog_params,
         reglog_name, reglog_duration) = best_model(model_name='Régression Logistique',
                                                    model=LogisticRegression(max_iter=200,
                                                                             random_state=42),
                                                    cv=KFold(n_splits=5,
                                                             shuffle=True,
                                                             random_state=42),
                                                    xtrain=X_train,
                                                    ytrain=y_train,
                                                    preprocessor=StandardScaler(),
                                                    params={
                                                        "preprocessor": [StandardScaler(), MinMaxScaler()],
                                                        "classifier__penalty": ['l2', 'l1'],
                                                        "classifier__solver": ['lbfgs', 'saga'],
                                                        "classifier__C": [100, 10, 1.0, 0.1, 0.01], },
                                                    scoring=make_scorer(fbeta_score, beta=9),
                                                    xtest=X_test,
                                                    ytest=y_test,
                                                    oversampling_strategy=0.1,
                                                    undersampling_strategy=0.5,
                                                    balanced=False,
                                                    Randomized=False)

        # Evaluation sur les données de test
        (biz_reglog, beta_reglog, rec_reglog, prec_reglog,
         acc_reglog, auc_reglog, y_pred_reglog) = eval_metrics(best_model=reglog,
                                                               xtest=X_test,
                                                               ytest=y_test,
                                                               beta_value=9)

        # log des paramètres et scores à chaque fois que le modèle est lancé
        mlflow.log_param("best parameters", reglog_params)

        mlflow.log_metric("score metier", biz_reglog)
        mlflow.log_metric("f betascore", beta_reglog)
        mlflow.log_metric("recall", rec_reglog)
        mlflow.log_metric("precision", prec_reglog)
        mlflow.log_metric("accuracy", acc_reglog)
        mlflow.log_metric("auc", auc_reglog)
        mlflow.log_metric("Tps_entrainement", reglog_duration)

        mlflow.sklearn.log_model(reglog, "logistic_regression")

    # Régression logistique sur données rééquilibrées avec Class Weight

    # Pipeline, optimisation et entrainement
    mlflow.set_experiment("mlflow-default-risk")
    experiment = mlflow.get_experiment_by_name("mlflow-default-risk")

    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name='Régression Logistique - Class Weight'):
        (reglog_cw, reglog_params_cw,
         reglog_name_cw, reglog_duration_cw) = best_model(model_name='Régression Logistique - Class Weight',
                                                          model=LogisticRegression(max_iter=200,
                                                                                   random_state=42,
                                                                                   class_weight='balanced'),
                                                          cv=KFold(n_splits=5,
                                                                   shuffle=True,
                                                                   random_state=42),
                                                          xtrain=X_train,
                                                          ytrain=y_train,
                                                          preprocessor=StandardScaler(),
                                                          params={
                                                              "preprocessor": [StandardScaler(), MinMaxScaler()],
                                                              "classifier__penalty": ['l2', 'l1'],
                                                              "classifier__solver": ['lbfgs', 'saga'],
                                                              "classifier__C": [100, 10, 1.0, 0.1, 0.01], },
                                                          scoring=make_scorer(fbeta_score, beta=9),
                                                          xtest=X_test,
                                                          ytest=y_test,
                                                          oversampling_strategy=0.1,
                                                          undersampling_strategy=0.5,
                                                          balanced=False,
                                                          Randomized=False)

        # Evaluation sur les données de test
        (biz_reglog_cw, beta_reglog_cw, rec_reglog_cw, prec_reglog_cw,
         acc_reglog_cw, auc_reglog_cw, y_pred_reglog_cw) = eval_metrics(best_model=reglog_cw,
                                                                        xtest=X_test,
                                                                        ytest=y_test,
                                                                        beta_value=9)

        # log des paramètres et scores à chaque fois que le modèle est lancé
        mlflow.log_param("best parameters", reglog_params_cw)

        mlflow.log_metric("score metier", biz_reglog_cw)
        mlflow.log_metric("f betascore", beta_reglog_cw)
        mlflow.log_metric("recall", rec_reglog_cw)
        mlflow.log_metric("precision", prec_reglog_cw)
        mlflow.log_metric("accuracy", acc_reglog_cw)
        mlflow.log_metric("auc", auc_reglog_cw)
        mlflow.log_metric("Tps_entrainement", reglog_duration_cw)

        mlflow.sklearn.log_model(reglog, "logistic_regression_CW")

    # Régression logistique sur données rééquilibrées avec SMOTE
    # (10% classe majoritaire et 50% de plus que classe minoritaire)

    # Pipeline, optimisation et entrainement
    mlflow.set_experiment("mlflow-default-risk")
    experiment = mlflow.get_experiment_by_name("mlflow-default-risk")

    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name='Régression Logistique - SMOTE 10/50'):
        (reglog_smote1050, reglog_smote_params1050,
         reglog_smote_name1050, reglog_smote_duration1050) = best_model(
            model_name='Régression Logistique - SMOTE 10/50',
            model=LogisticRegression(max_iter=200,
                                     random_state=42, ),
            cv=KFold(n_splits=5,
                     shuffle=True,
                     random_state=42),
            xtrain=X_train,
            ytrain=y_train,
            preprocessor=StandardScaler(),
            params={"preprocessor": [StandardScaler(), MinMaxScaler()],
                    "classifier__penalty": ['l2', 'l1'],
                    "classifier__solver": ['lbfgs', 'saga'],
                    "classifier__C": [100, 10, 1.0, 0.1, 0.01], },
            scoring=make_scorer(fbeta_score, beta=9),
            xtest=X_test,
            ytest=y_test,
            oversampling_strategy=0.1,
            undersampling_strategy=0.5,
            balanced=True,
            Randomized=True)

        # Evaluation sur les données de test
        (biz_reglog_SMOTE1050, beta_reglog_SMOTE1050, rec_reglog_SMOTE1050, prec_reglog_SMOTE1050,
         acc_reglog_SMOTE1050, auc_reglog_SMOTE1050, y_pred_reglog_SMOTE1050) = eval_metrics(
            best_model=reglog_smote1050,
            xtest=X_test,
            ytest=y_test,
            beta_value=9)

        # log des paramètres et scores à chaque fois que le modèle est lancé
        mlflow.log_param("best parameters", reglog_smote_params1050)

        mlflow.log_metric("score metier", biz_reglog_SMOTE1050)
        mlflow.log_metric("f betascore", beta_reglog_SMOTE1050)
        mlflow.log_metric("recall", rec_reglog_SMOTE1050)
        mlflow.log_metric("precision", prec_reglog_SMOTE1050)
        mlflow.log_metric("accuracy", acc_reglog_SMOTE1050)
        mlflow.log_metric("auc", auc_reglog_SMOTE1050)
        mlflow.log_metric("Tps_entrainement", reglog_smote_duration1050)

        mlflow.sklearn.log_model(reglog, "logistic_regression_smote1050")

    # Régression logistique sur données rééquilibrées avec SMOTE
    # (20% classe majoritaire et 30% de plus que classe minoritaire)

    # Pipeline, optimisation et entrainement
    mlflow.set_experiment("mlflow-default-risk")
    experiment = mlflow.get_experiment_by_name("mlflow-default-risk")

    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name='Régression Logistique - SMOTE 20/30'):
        (reglog_smote2030, reglog_smote_params2030,
         reglog_smote_name2030, reglog_smote_duration2030) = best_model(
            model_name='Régression Logistique - SMOTE 20/30',
            model=LogisticRegression(max_iter=200,
                                     random_state=42, ),
            cv=KFold(n_splits=5,
                     shuffle=True,
                     random_state=42),
            xtrain=X_train,
            ytrain=y_train,
            preprocessor=StandardScaler(),
            params={"preprocessor": [StandardScaler(), MinMaxScaler()],
                    "classifier__penalty": ['l2', 'l1'],
                    "classifier__solver": ['lbfgs', 'saga'],
                    "classifier__C": [100, 10, 1.0, 0.1, 0.01], },
            scoring=make_scorer(fbeta_score, beta=9),
            xtest=X_test,
            ytest=y_test,
            oversampling_strategy=0.2,
            undersampling_strategy=0.3,
            balanced=True,
            Randomized=True)

        # Evaluation sur les données de test
        (biz_reglog_SMOTE2030, beta_reglog_SMOTE2030, rec_reglog_SMOTE2030, prec_reglog_SMOTE2030,
         acc_reglog_SMOTE2030, auc_reglog_SMOTE2030, y_pred_reglog_SMOTE2030) = eval_metrics(
            best_model=reglog_smote2030,
            xtest=X_test,
            ytest=y_test,
            beta_value=9)

        # log des paramètres et scores à chaque fois que le modèle est lancé
        mlflow.log_param("best parameters", reglog_smote_params2030)
        mlflow.log_param("preprocessor", reglog_smote_params2030['preprocessor'])
        mlflow.log_param("penalty", reglog_smote_params2030['classifier__penalty'])
        mlflow.log_param("solver", reglog_smote_params2030['classifier__solver'])
        mlflow.log_param("C", reglog_smote_params2030['classifier__C'])

        mlflow.log_metric("score metier", biz_reglog_SMOTE2030)
        mlflow.log_metric("f betascore", beta_reglog_SMOTE2030)
        mlflow.log_metric("recall", rec_reglog_SMOTE2030)
        mlflow.log_metric("precision", prec_reglog_SMOTE2030)
        mlflow.log_metric("accuracy", acc_reglog_SMOTE2030)
        mlflow.log_metric("auc", auc_reglog_SMOTE2030)
        mlflow.log_metric("Tps_entrainement", reglog_smote_duration2030)

        mlflow.sklearn.log_model(reglog, "logistic_regression_smote2030")

    # Régression logistique sur données rééquilibrées avec SMOTE
    # (10% classe majoritaire et 60% de plus que classe minoritaire)

    mlflow.set_experiment("mlflow-default-risk")
    experiment = mlflow.get_experiment_by_name("mlflow-default-risk")

    # Pipeline, optimisation et entrainement
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name='Régression Logistique - SMOTE 10/60'):
        (reglog_smote1060, reglog_smote_params1060,
         reglog_smote_name1060, reglog_smote_duration1060) = best_model(
            model_name='Régression Logistique - SMOTE 20/30',
            model=LogisticRegression(max_iter=200,
                                     random_state=42, ),
            cv=KFold(n_splits=5,
                     shuffle=True,
                     random_state=42),
            xtrain=X_train,
            ytrain=y_train,
            preprocessor=StandardScaler(),
            params={"preprocessor": [StandardScaler(), MinMaxScaler()],
                    "classifier__penalty": ['l2', 'l1'],
                    "classifier__solver": ['lbfgs', 'saga'],
                    "classifier__C": [100, 10, 1.0, 0.1, 0.01], },
            scoring=make_scorer(fbeta_score, beta=9),
            xtest=X_test,
            ytest=y_test,
            oversampling_strategy=0.1,
            undersampling_strategy=0.6,
            balanced=True,
            Randomized=True)

        # Evaluation sur les données de test
        (biz_reglog_SMOTE1060, beta_reglog_SMOTE1060, rec_reglog_SMOTE1060, prec_reglog_SMOTE1060,
         acc_reglog_SMOTE1060, auc_reglog_SMOTE1060, y_pred_reglog_SMOTE1060) = eval_metrics(
            best_model=reglog_smote1060,
            xtest=X_test,
            ytest=y_test,
            beta_value=9)

        # log des paramètres et scores à chaque fois que le modèle est lancé
        mlflow.log_param("best parameters", reglog_smote_params1060)

        mlflow.log_metric("score metier", biz_reglog_SMOTE1060)
        mlflow.log_metric("f betascore", beta_reglog_SMOTE1060)
        mlflow.log_metric("recall", rec_reglog_SMOTE1060)
        mlflow.log_metric("precision", prec_reglog_SMOTE1060)
        mlflow.log_metric("accuracy", acc_reglog_SMOTE1060)
        mlflow.log_metric("auc", auc_reglog_SMOTE1060)
        mlflow.log_metric("Tps_entrainement", reglog_smote_duration1060)

        mlflow.sklearn.log_model(reglog_smote1060, "logistic_regression_smote1060")

        return reglog, reglog_cw, reglog_smote1050, reglog_smote2030, reglog_smote1060


def lgb_model():
    """Fonction qui utilise un pipeline de transformation des données et applique
    un LightGBM pour effectuer la prédiction. Le pipeline est optimisé et entrainé en fonction
    du scoring f betascore (avec beta = 9), cross validation et d'une grille de paramètres.
    Le modèle sera testé sur données rééquilibrées avec class weight + hyperparameters tuning

    Le modèle, ses paramètres et les metrics servant à son évaluation sont trackés et stockés sur MLFlow.

    Arguments:
    --------------------------------

    return:
    --------------------------------
    lgb: les différents modèles entrainés et optimisés"""

    # LightGBM sur données rééquilibrées avec class weight + hyperparameters tuning

    # Récupération des données preprocessées
    df = pd.read_csv('../Dashboard_API/df.csv', nrows=None)

    # Train test split
    X_train, X_test, y_train, y_test = data_train_test(df)

    # Pipeline, optimisation et entrainement
    mlflow.set_experiment("mlflow-default-risk")
    experiment = mlflow.get_experiment_by_name("mlflow-default-risk")
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name='LightGBM'):
        params = {"preprocessor": [StandardScaler(), MinMaxScaler()],
                  "classifier__ n_estimators": [10, 50, 100, 500, 1000, 5000],
                  "classifier__ max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  "classifier__ num_leaves": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
                  "classifier__ learning_rate": [0.0001, 0.001, 0.01, 0.1, 1.0],
                  "classifier__ boosting_type": ['gbdt', 'dart', 'rf’'], }

        (lgb, lgb_params, lgb_name, lgb_duration) = best_model(model_name='LightGBM',
                                                               model=LGBMClassifier(random_state=42,
                                                                                    class_weight='balanced'),
                                                               cv=KFold(n_splits=5,
                                                                        shuffle=True,
                                                                        random_state=42),
                                                               xtrain=X_train,
                                                               ytrain=y_train,
                                                               preprocessor=StandardScaler(),
                                                               params=params,
                                                               scoring=make_scorer(fbeta_score, beta=9),
                                                               xtest=X_test,
                                                               ytest=y_test,
                                                               oversampling_strategy=0.1,
                                                               undersampling_strategy=0.5,
                                                               balanced=False,
                                                               Randomized=True)

        # Evaluation sur les données de test
        (biz_lgb, beta_lgb, rec_lgb, prec_lgb,
         acc_lgb, auc_lgb, y_pred_lgb) = eval_metrics(best_model=lgb,
                                                      xtest=X_test,
                                                      ytest=y_test,
                                                      beta_value=9)

        # log des paramètres et scores à chaque fois que le modèle est lancé
        mlflow.log_param("best parameters", lgb_params)

        mlflow.log_metric("score metier", biz_lgb)
        mlflow.log_metric("f betascore", beta_lgb)
        mlflow.log_metric("recall", rec_lgb)
        mlflow.log_metric("precision", prec_lgb)
        mlflow.log_metric("accuracy", acc_lgb)
        mlflow.log_metric("auc", auc_lgb)
        mlflow.log_metric("Tps_entrainement", lgb_duration)

        mlflow.sklearn.log_model(lgb, "LightGBM")

        return lgb


def lgb_feat_imp_5():
    """Fonction qui utilise un pipeline de transformation des données et applique
    un LightGBM pour effectuer la prédiction. Le pipeline est optimisé et entrainé en fonction
    du scoring f betascore (avec beta = 9), cross validation et d'une grille de paramètres.
    Le modèle sera testé sur une sélection des 5 features les plus importantes.

    Le modèle, ses paramètres et les metrics servant à son évaluation sont trackés et stockés sur MLFlow.

    Arguments:
    --------------------------------

    return:
    --------------------------------
    lgb: le modèle entrainé et optimisé"""

    # LightGBM avec sélection de 5 Features via Features importance

    # Récupération des données preprocessées
    df = pd.read_csv('../Dashboard_API/df.csv', nrows=None)

    # Train test split
    X_train, X_test, y_train, y_test = data_train_test(df)

    # Pipeline, optimisation et entrainement
    feat_lgb5 = ['CREDIT_DURATION', 'EXT_SOURCE_2', 'INSTAL_DAYS_PAST_DUE_MEAN',
                 'PAYMENT_RATE', 'POS_CNT_INSTALMENT_FUTURE_MEAN']

    mlflow.set_experiment("mlflow-default-risk")
    experiment = mlflow.get_experiment_by_name("mlflow-default-risk")
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name='LightGBM - Feature Importance *5'):
        params = {"preprocessor": [StandardScaler(), MinMaxScaler()],
                  "classifier__ n_estimators": [10, 50, 100, 500, 1000, 5000],
                  "classifier__ max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  "classifier__ num_leaves": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
                  "classifier__ learning_rate": [0.0001, 0.001, 0.01, 0.1, 1.0],
                  "classifier__ boosting_type": ['gbdt', 'dart', 'rf’'], }

        lgb5, lgb_params5, lgb_name5, lgb_duration5 = best_model(model_name='LightGBM - Feature Importance *5',
                                                                 model=LGBMClassifier(random_state=42,
                                                                                      class_weight='balanced'),
                                                                 cv=KFold(n_splits=5,
                                                                          shuffle=True,
                                                                          random_state=42),
                                                                 xtrain=X_train[feat_lgb5],
                                                                 ytrain=y_train,
                                                                 preprocessor=StandardScaler(),
                                                                 params=params,
                                                                 scoring=make_scorer(fbeta_score, beta=9),
                                                                 xtest=X_test[feat_lgb5],
                                                                 ytest=y_test,
                                                                 oversampling_strategy=0.1,
                                                                 undersampling_strategy=0.5,
                                                                 balanced=False,
                                                                 Randomized=True)

        # Evaluation sur les données de test
        (biz_lgb5, beta_lgb5, rec_lgb5, prec_lgb5,
         acc_lgb5, auc_lgb5, y_pred_lgb5) = eval_metrics(best_model=lgb5,
                                                         xtest=X_test[feat_lgb5],
                                                         ytest=y_test,
                                                         beta_value=9)

        # log des paramètres et scores à chaque fois que le modèle est lancé
        mlflow.log_param("best parameters", lgb_params5)

        mlflow.log_metric("score metier", biz_lgb5)
        mlflow.log_metric("f betascore", beta_lgb5)
        mlflow.log_metric("recall", rec_lgb5)
        mlflow.log_metric("precision", prec_lgb5)
        mlflow.log_metric("accuracy", acc_lgb5)
        mlflow.log_metric("auc", auc_lgb5)
        mlflow.log_metric("Tps_entrainement", lgb_duration5)

        mlflow.sklearn.log_model(lgb5, "LightGBM_Feature_Importance_5")

        return lgb5


def lgb_feat_imp_10():
    """Fonction qui utilise un pipeline de transformation des données et applique
    un LightGBM pour effectuer la prédiction. Le pipeline est optimisé et entrainé en fonction
    du scoring f betascore (avec beta = 9), cross validation et d'une grille de paramètres.
    Le modèle sera testé sur une sélection des 10 features les plus importantes.

    Le modèle, ses paramètres et les metrics servant à son évaluation sont trackés et stockés sur MLFlow.

    Arguments:
    --------------------------------

    return:
    --------------------------------
    lgb: le modèle entrainé et optimisé"""

    # LightGBM avec sélection de 10 Features via Features importance

    # Récupération des données preprocessées
    df = pd.read_csv('../Dashboard_API/df.csv', nrows=None)

    # Train test split
    X_train, X_test, y_train, y_test = data_train_test(df)

    # Pipeline, optimisation et entrainement
    feat_lgb10 = ['CREDIT_DURATION', 'EXT_SOURCE_2', 'INSTAL_DAYS_PAST_DUE_MEAN', 'PAYMENT_RATE',
                  'POS_CNT_INSTALMENT_FUTURE_MEAN', 'CREDIT_GOODS_PERC', 'AGE', 'POS_NB_CREDIT',
                  'BURO_CREDIT_ACTIVE_Active_SUM', 'BURO_AMT_CREDIT_SUM_DEBT_MEAN']

    mlflow.set_experiment("mlflow-default-risk")
    experiment = mlflow.get_experiment_by_name("mlflow-default-risk")
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name='LightGBM - Feature Importance *10'):
        params = {"preprocessor": [StandardScaler(), MinMaxScaler()],
                  "classifier__ n_estimators": [10, 50, 100, 500, 1000, 5000],
                  "classifier__ max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  "classifier__ num_leaves": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
                  "classifier__ learning_rate": [0.0001, 0.001, 0.01, 0.1, 1.0],
                  "classifier__ boosting_type": ['gbdt', 'dart', 'rf’'], }

        lgb10, lgb_params10, lgb_name10, lgb_duration10 = best_model(
            model_name='LightGBM - Feature Importance *10',
            model=LGBMClassifier(random_state=42,
                                 class_weight='balanced'),
            cv=KFold(n_splits=5,
                     shuffle=True,
                     random_state=42),
            xtrain=X_train[feat_lgb10],
            ytrain=y_train,
            preprocessor=StandardScaler(),
            params=params,
            scoring=make_scorer(fbeta_score, beta=9),
            xtest=X_test[feat_lgb10],
            ytest=y_test,
            oversampling_strategy=0.1,
            undersampling_strategy=0.5,
            balanced=False,
            Randomized=True)

        # Evaluation sur les données de test
        (biz_lgb10, beta_lgb10, rec_lgb10, prec_lgb10,
         acc_lgb10, auc_lgb10, y_pred_lgb10) = eval_metrics(best_model=lgb10,
                                                            xtest=X_test[feat_lgb10],
                                                            ytest=y_test,
                                                            beta_value=9)

        # log des paramètres et scores à chaque fois que le modèle est lancé
        mlflow.log_param("best parameters", lgb_params10)

        mlflow.log_metric("score metier", biz_lgb10)
        mlflow.log_metric("f betascore", beta_lgb10)
        mlflow.log_metric("recall", rec_lgb10)
        mlflow.log_metric("precision", prec_lgb10)
        mlflow.log_metric("accuracy", acc_lgb10)
        mlflow.log_metric("auc", auc_lgb10)
        mlflow.log_metric("Tps_entrainement", lgb_duration10)

        mlflow.sklearn.log_model(lgb10, "LightGBM_Feature_Importance_10")

        return lgb10


def lgb_feat_imp_20():
    """Fonction qui utilise un pipeline de transformation des données et applique
    un LightGBM pour effectuer la prédiction. Le pipeline est optimisé et entrainé en fonction
    du scoring f betascore (avec beta = 9), cross validation et d'une grille de paramètres.
    Le modèle sera testé sur une sélection des 20 features les plus importantes.

    Le modèle, ses paramètres et les metrics servant à son évaluation sont trackés et stockés sur MLFlow.

    Arguments:
    --------------------------------

    return:
    --------------------------------
    lgb: le modèle entrainé et optimisé"""

    # LightGBM avec sélection de 20 Features via Features importance

    # Récupération des données preprocessées
    df = pd.read_csv('../Dashboard_API/df.csv', nrows=None)

    # Train test split
    X_train, X_test, y_train, y_test = data_train_test(df)

    # Pipeline, optimisation et entrainement
    feat_lgb20 = ['CREDIT_DURATION', 'EXT_SOURCE_2', 'INSTAL_DAYS_PAST_DUE_MEAN',
                  'PAYMENT_RATE', 'POS_CNT_INSTALMENT_FUTURE_MEAN', 'CREDIT_GOODS_PERC',
                  'AGE', 'POS_NB_CREDIT', 'BURO_CREDIT_ACTIVE_Active_SUM', 'BURO_AMT_CREDIT_SUM_DEBT_MEAN',
                  'YEARS_EMPLOYED', 'YEARS_ID_PUBLISH', 'INSTAL_PAYMENT_DIFF_MEAN', 'BURO_AMT_CREDIT_SUM_MEAN',
                  'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'BURO_YEARS_CREDIT_ENDDATE_MEAN', 'AMT_CREDIT',
                  'YEARS_LAST_PHONE_CHANGE', 'POS_MONTHS_BALANCE_MEAN']

    mlflow.set_experiment("mlflow-default-risk")
    experiment = mlflow.get_experiment_by_name("mlflow-default-risk")
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name='LightGBM - Feature Importance *20'):
        params = {"preprocessor": [StandardScaler(), MinMaxScaler()],
                  "classifier__ n_estimators": [10, 50, 100, 500, 1000, 5000],
                  "classifier__ max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  "classifier__ num_leaves": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
                  "classifier__ learning_rate": [0.0001, 0.001, 0.01, 0.1, 1.0],
                  "classifier__ boosting_type": ['gbdt', 'dart', 'rf’'], }

        lgb20, lgb_params20, lgb_name20, lgb_duration20 = best_model(
            model_name='LightGBM - Feature Importance *20',
            model=LGBMClassifier(random_state=42,
                                 class_weight='balanced'),
            cv=KFold(n_splits=5,
                     shuffle=True,
                     random_state=42),
            xtrain=X_train[feat_lgb20],
            ytrain=y_train,
            preprocessor=StandardScaler(),
            params=params,
            scoring=make_scorer(fbeta_score, beta=9),
            xtest=X_test[feat_lgb20],
            ytest=y_test,
            oversampling_strategy=0.1,
            undersampling_strategy=0.5,
            balanced=False,
            Randomized=True)

        # Evaluation sur les données de test
        (biz_lgb20, beta_lgb20, rec_lgb20, prec_lgb20,
         acc_lgb20, auc_lgb20, y_pred_lgb20) = eval_metrics(best_model=lgb20,
                                                            xtest=X_test[feat_lgb20],
                                                            ytest=y_test,
                                                            beta_value=9)

        # log des paramètres et scores à chaque fois que le modèle est lancé
        mlflow.log_param("best parameters", lgb_params20)

        mlflow.log_metric("score metier", biz_lgb20)
        mlflow.log_metric("f betascore", beta_lgb20)
        mlflow.log_metric("recall", rec_lgb20)
        mlflow.log_metric("precision", prec_lgb20)
        mlflow.log_metric("accuracy", acc_lgb20)
        mlflow.log_metric("auc", auc_lgb20)
        mlflow.log_metric("Tps_entrainement", lgb_duration20)

        mlflow.sklearn.log_model(lgb20, "LightGBM_Feature_Importance_20")

        return lgb20


def lgb_feat_imp_30():
    """Fonction qui utilise un pipeline de transformation des données et applique
    un LightGBM pour effectuer la prédiction. Le pipeline est optimisé et entrainé en fonction
    du scoring f betascore (avec beta = 9), cross validation et d'une grille de paramètres.
    Le modèle sera testé sur une sélection des 30 features les plus importantes.

    Le modèle, ses paramètres et les metrics servant à son évaluation sont trackés et stockés sur MLFlow.

    Arguments:
    --------------------------------

    return:
    --------------------------------
    lgb: le modèle entrainé et optimisé"""

    # LightGBM avec sélection de 30 Features via Features importance

    # Récupération des données preprocessées
    df = pd.read_csv('../Dashboard_API/df.csv', nrows=None)

    # Train test split
    X_train, X_test, y_train, y_test = data_train_test(df)

    # Pipeline, optimisation et entrainement
    feat_lgb30 = ['CREDIT_DURATION', 'EXT_SOURCE_2', 'INSTAL_DAYS_PAST_DUE_MEAN',
                  'PAYMENT_RATE', 'POS_CNT_INSTALMENT_FUTURE_MEAN', 'CREDIT_GOODS_PERC',
                  'AGE', 'POS_NB_CREDIT', 'BURO_CREDIT_ACTIVE_Active_SUM', 'BURO_AMT_CREDIT_SUM_DEBT_MEAN',
                  'YEARS_EMPLOYED', 'YEARS_ID_PUBLISH', 'INSTAL_PAYMENT_DIFF_MEAN', 'BURO_AMT_CREDIT_SUM_MEAN',
                  'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'BURO_YEARS_CREDIT_ENDDATE_MEAN', 'AMT_CREDIT',
                  'YEARS_LAST_PHONE_CHANGE', 'POS_MONTHS_BALANCE_MEAN', 'INSTAL_DAYS_BEFORE_DUE_MEAN',
                  'BURO_AMT_CREDIT_SUM_DEBT_SUM', 'CODE_GENDER', 'PREV_YEARS_DECISION_MEAN',
                  'REGION_POPULATION_RELATIVE', 'DEBT_RATIO', 'BURO_AMT_CREDIT_SUM_SUM',
                  'BURO_YEARS_CREDIT_ENDDATE_MAX', 'NAME_EDUCATION_TYPE_Lower Secondary & Secondary',
                  'PREV_PAYMENT_RATE_MEAN']

    mlflow.set_experiment("mlflow-default-risk")
    experiment = mlflow.get_experiment_by_name("mlflow-default-risk")
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name='LightGBM - Feature Importance *30'):
        params = {"preprocessor": [StandardScaler(), MinMaxScaler()],
                  "classifier__ n_estimators": [10, 50, 100, 500, 1000, 5000],
                  "classifier__ max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  "classifier__ num_leaves": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
                  "classifier__ learning_rate": [0.0001, 0.001, 0.01, 0.1, 1.0],
                  "classifier__ boosting_type": ['gbdt', 'dart', 'rf’'], }

        lgb30, lgb_params30, lgb_name30, lgb_duration30 = best_model(
            model_name='LightGBM - Feature Importance *30',
            model=LGBMClassifier(random_state=42,
                                 class_weight='balanced'),
            cv=KFold(n_splits=5,
                     shuffle=True,
                     random_state=42),
            xtrain=X_train[feat_lgb30],
            ytrain=y_train,
            preprocessor=StandardScaler(),
            params=params,
            scoring=make_scorer(fbeta_score, beta=9),
            xtest=X_test[feat_lgb30],
            ytest=y_test,
            oversampling_strategy=0.1,
            undersampling_strategy=0.5,
            balanced=False,
            Randomized=True)

        # Evaluation sur les données de test
        (biz_lgb30, beta_lgb30, rec_lgb30, prec_lgb30,
         acc_lgb30, auc_lgb30, y_pred_lgb30) = eval_metrics(best_model=lgb30,
                                                            xtest=X_test[feat_lgb30],
                                                            ytest=y_test,
                                                            beta_value=9)

        # log des paramètres et scores à chaque fois que le modèle est lancé
        mlflow.log_param("best parameters", lgb_params30)

        mlflow.log_metric("score metier", biz_lgb30)
        mlflow.log_metric("f betascore", beta_lgb30)
        mlflow.log_metric("recall", rec_lgb30)
        mlflow.log_metric("precision", prec_lgb30)
        mlflow.log_metric("accuracy", acc_lgb30)
        mlflow.log_metric("auc", auc_lgb30)
        mlflow.log_metric("Tps_entrainement", lgb_duration30)

        mlflow.sklearn.log_model(lgb30, "LightGBM_Feature_Importance_30")

        return lgb30


def lgb_gscv():
    """Fonction qui utilise un pipeline de transformation des données et applique
    un LightGBM pour effectuer la prédiction. Le pipeline est optimisé et entrainé en fonction
    du scoring f betascore (avec beta = 9), cross validation et d'une grille de paramètres.
    Le modèle sera optimisé via GridSearchCV avec des hyperparamètres proches de ceux sur
    le modèle avec les 30 features les plus importantes.

    Le modèle, ses paramètres et les metrics servant à son évaluation sont trackés et stockés sur MLFlow.

    Arguments:
    --------------------------------

    return:
    --------------------------------
    lgb: le modèle entrainé et optimisé"""

    # LightGBM avec Tentative de fine tuning via GridSearchCV

    # Récupération des données preprocessées
    df = pd.read_csv('../Dashboard_API/df.csv', nrows=None)

    # Train test split
    X_train, X_test, y_train, y_test = data_train_test(df)

    # Pipeline, optimisation et entrainement
    feat_lgb30 = ['CREDIT_DURATION', 'EXT_SOURCE_2', 'INSTAL_DAYS_PAST_DUE_MEAN',
                  'PAYMENT_RATE', 'POS_CNT_INSTALMENT_FUTURE_MEAN', 'CREDIT_GOODS_PERC',
                  'AGE', 'POS_NB_CREDIT', 'BURO_CREDIT_ACTIVE_Active_SUM', 'BURO_AMT_CREDIT_SUM_DEBT_MEAN',
                  'YEARS_EMPLOYED', 'YEARS_ID_PUBLISH', 'INSTAL_PAYMENT_DIFF_MEAN', 'BURO_AMT_CREDIT_SUM_MEAN',
                  'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'BURO_YEARS_CREDIT_ENDDATE_MEAN', 'AMT_CREDIT',
                  'YEARS_LAST_PHONE_CHANGE', 'POS_MONTHS_BALANCE_MEAN', 'INSTAL_DAYS_BEFORE_DUE_MEAN',
                  'BURO_AMT_CREDIT_SUM_DEBT_SUM', 'CODE_GENDER', 'PREV_YEARS_DECISION_MEAN',
                  'REGION_POPULATION_RELATIVE', 'DEBT_RATIO', 'BURO_AMT_CREDIT_SUM_SUM',
                  'BURO_YEARS_CREDIT_ENDDATE_MAX', 'NAME_EDUCATION_TYPE_Lower Secondary & Secondary',
                  'PREV_PAYMENT_RATE_MEAN']

    mlflow.set_experiment("mlflow-default-risk")
    experiment = mlflow.get_experiment_by_name("mlflow-default-risk")
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name='LightGBM 30 - GridSearchCV'):
        params = {"classifier__ n_estimators": [100, 200, 300, 500, 600, 800],
                  "classifier__ max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  "classifier__ num_leaves": [2, 4, 6, 8, 10, 15, 20, 32, 60, 64, 70, 128, 256, 512, 1024],
                  "classifier__ learning_rate": [0.001, 0.002, 0.005, 0.007, 0.01, 0.1, 0.2, 0.5, 1.0],
                  "classifier__ boosting_type": ['gbdt', 'rf’'], }

        (lgb30_gscv, lgb30_params_gscv,
         lgb30_name_gscv, lgb30_duration_gscv) = best_model(model_name='LightGBM 30 - GridSearchCV',
                                                            model=LGBMClassifier(random_state=42,
                                                                                 class_weight='balanced'),
                                                            cv=KFold(n_splits=5,
                                                                     shuffle=True,
                                                                     random_state=42),
                                                            xtrain=X_train[feat_lgb30],
                                                            ytrain=y_train,
                                                            preprocessor=StandardScaler(),
                                                            params=params,
                                                            scoring=make_scorer(fbeta_score, beta=9),
                                                            xtest=X_test[feat_lgb30],
                                                            ytest=y_test,
                                                            oversampling_strategy=0.1,
                                                            undersampling_strategy=0.5,
                                                            balanced=False,
                                                            Randomized=False)

        # Evaluation sur les données de test
        (biz_lgb30_gscv, beta_lgb30_gscv, rec_lgb30_gscv, prec_lgb30_gscv,
         acc_lgb30_gscv, auc_lgb30_gscv, y_pred_lgb30_gscv) = eval_metrics(best_model=lgb30_gscv,
                                                                           xtest=X_test[feat_lgb30],
                                                                           ytest=y_test,
                                                                           beta_value=9)

        # log des paramètres et scores à chaque fois que le modèle est lancé
        mlflow.log_param("best parameters", lgb30_params_gscv)

        mlflow.log_metric("score metier", biz_lgb30_gscv)
        mlflow.log_metric("f betascore", beta_lgb30_gscv)
        mlflow.log_metric("recall", rec_lgb30_gscv)
        mlflow.log_metric("precision", prec_lgb30_gscv)
        mlflow.log_metric("accuracy", acc_lgb30_gscv)
        mlflow.log_metric("auc", auc_lgb30_gscv)
        mlflow.log_metric("Tps_entrainement", lgb30_duration_gscv)

        mlflow.sklearn.log_model(lgb30_gscv, "LightGBM_30_GridSearchCV")

        return lgb30_gscv


def lgb_beta2():
    """Fonction qui utilise un pipeline de transformation des données et applique
    un LightGBM pour effectuer la prédiction. Le pipeline est optimisé et entrainé en fonction
    du scoring f betascore (avec beta = 2), cross validation et d'une grille de paramètres.

    Le modèle, ses paramètres et les metrics servant à son évaluation sont trackés et stockés sur MLFlow.

    Arguments:
    --------------------------------

    return:
    --------------------------------
    lgb: le modèle entrainé et optimisé"""

    # LightGBM avec tentative d'amélioration de la précision avec scoring sur valeur beta plus faible

    # Récupération des données preprocessées
    df = pd.read_csv('../Dashboard_API/df.csv', nrows=None)

    # Train test split
    X_train, X_test, y_train, y_test = data_train_test(df)

    # Pipeline, optimisation et entrainement
    feat_lgb30 = ['CREDIT_DURATION', 'EXT_SOURCE_2', 'INSTAL_DAYS_PAST_DUE_MEAN',
                  'PAYMENT_RATE', 'POS_CNT_INSTALMENT_FUTURE_MEAN', 'CREDIT_GOODS_PERC',
                  'AGE', 'POS_NB_CREDIT', 'BURO_CREDIT_ACTIVE_Active_SUM', 'BURO_AMT_CREDIT_SUM_DEBT_MEAN',
                  'YEARS_EMPLOYED', 'YEARS_ID_PUBLISH', 'INSTAL_PAYMENT_DIFF_MEAN', 'BURO_AMT_CREDIT_SUM_MEAN',
                  'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'BURO_YEARS_CREDIT_ENDDATE_MEAN', 'AMT_CREDIT',
                  'YEARS_LAST_PHONE_CHANGE', 'POS_MONTHS_BALANCE_MEAN', 'INSTAL_DAYS_BEFORE_DUE_MEAN',
                  'BURO_AMT_CREDIT_SUM_DEBT_SUM', 'CODE_GENDER', 'PREV_YEARS_DECISION_MEAN',
                  'REGION_POPULATION_RELATIVE', 'DEBT_RATIO', 'BURO_AMT_CREDIT_SUM_SUM',
                  'BURO_YEARS_CREDIT_ENDDATE_MAX', 'NAME_EDUCATION_TYPE_Lower Secondary & Secondary',
                  'PREV_PAYMENT_RATE_MEAN']

    mlflow.set_experiment("mlflow-default-risk")
    experiment = mlflow.get_experiment_by_name("mlflow-default-risk")
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name='LightGBM 30 - Beta = 2'):
        params = {"classifier__ n_estimators": [100, 200, 300, 500, 600, 800],
                  "classifier__ max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  "classifier__ num_leaves": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25],
                  "classifier__ learning_rate": [0.001, 0.002, 0.005, 0.007, 0.01, 0.1, 1.0],
                  "classifier__ boosting_type": ['gbdt', 'rf’'], }

        (lgb30_gscv_b2, lgb30_params_gscv_b2,
         lgb30_name_gscv_b2, lgb30_duration_gscv_b2) = best_model(model_name='LightGBM 30 - Beta = 2',
                                                                  model=LGBMClassifier(random_state=42,
                                                                                       class_weight='balanced'),
                                                                  cv=KFold(n_splits=5,
                                                                           shuffle=True,
                                                                           random_state=42),
                                                                  xtrain=X_train[feat_lgb30],
                                                                  ytrain=y_train,
                                                                  preprocessor=StandardScaler(),
                                                                  params=params,
                                                                  scoring=make_scorer(fbeta_score, beta=2),
                                                                  xtest=X_test[feat_lgb30],
                                                                  ytest=y_test,
                                                                  oversampling_strategy=0.1,
                                                                  undersampling_strategy=0.5,
                                                                  balanced=False,
                                                                  Randomized=False)

        # Evaluation sur les données de test
        (biz_lgb30_gscv_b2, beta_lgb30_gscv_b2, rec_lgb30_gscv_b2, prec_lgb30_gscv_b2,
         acc_lgb30_gscv_b2, auc_lgb30_gscv_b2, y_pred_lgb30_gscv_b2) = eval_metrics(best_model=lgb30_gscv_b2,
                                                                                    xtest=X_test[feat_lgb30],
                                                                                    ytest=y_test,
                                                                                    beta_value=9)

        # log des paramètres et scores à chaque fois que le modèle est lancé
        mlflow.log_param("best parameters", lgb30_params_gscv_b2)

        mlflow.log_metric("score metier", biz_lgb30_gscv_b2)
        mlflow.log_metric("f betascore", beta_lgb30_gscv_b2)
        mlflow.log_metric("recall", rec_lgb30_gscv_b2)
        mlflow.log_metric("precision", prec_lgb30_gscv_b2)
        mlflow.log_metric("accuracy", acc_lgb30_gscv_b2)
        mlflow.log_metric("auc", auc_lgb30_gscv_b2)
        mlflow.log_metric("Tps_entrainement", lgb30_duration_gscv_b2)

        mlflow.sklearn.log_model(lgb30_gscv_b2, "LightGBM_30_GridSearchCV")

        return lgb30_gscv_b2


def lgb_flaml():
    """Fonction qui utilise un pipeline de transformation des données et applique
    un LightGBM pour effectuer la prédiction. Le pipeline est optimisé et entrainé en fonction
    du scoring f betascore (avec beta = 9), cross validation et d'une grille de paramètres.

    Le modèle, ses paramètres et les metrics servant à son évaluation sont trackés et stockés sur MLFlow.

    Arguments:
    --------------------------------

    return:
    --------------------------------
    lgb: le modèle entrainé et optimisé"""

    # LightGBM avec optimisation des hyperparamètres via FLAML

    # Récupération des données preprocessées
    df = pd.read_csv('../Dashboard_API/df.csv', nrows=None)

    # Train test split
    X_train, X_test, y_train, y_test = data_train_test(df)

    # Pipeline, optimisation et entrainement
    feat_lgb30 = ['CREDIT_DURATION', 'EXT_SOURCE_2', 'INSTAL_DAYS_PAST_DUE_MEAN',
                  'PAYMENT_RATE', 'POS_CNT_INSTALMENT_FUTURE_MEAN', 'CREDIT_GOODS_PERC',
                  'AGE', 'POS_NB_CREDIT', 'BURO_CREDIT_ACTIVE_Active_SUM', 'BURO_AMT_CREDIT_SUM_DEBT_MEAN',
                  'YEARS_EMPLOYED', 'YEARS_ID_PUBLISH', 'INSTAL_PAYMENT_DIFF_MEAN', 'BURO_AMT_CREDIT_SUM_MEAN',
                  'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'BURO_YEARS_CREDIT_ENDDATE_MEAN', 'AMT_CREDIT',
                  'YEARS_LAST_PHONE_CHANGE', 'POS_MONTHS_BALANCE_MEAN', 'INSTAL_DAYS_BEFORE_DUE_MEAN',
                  'BURO_AMT_CREDIT_SUM_DEBT_SUM', 'CODE_GENDER', 'PREV_YEARS_DECISION_MEAN',
                  'REGION_POPULATION_RELATIVE', 'DEBT_RATIO', 'BURO_AMT_CREDIT_SUM_SUM',
                  'BURO_YEARS_CREDIT_ENDDATE_MAX', 'NAME_EDUCATION_TYPE_Lower Secondary & Secondary',
                  'PREV_PAYMENT_RATE_MEAN']

    mlflow.set_experiment("mlflow-default-risk")
    experiment = mlflow.get_experiment_by_name("mlflow-default-risk")
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name='LightGBM 30 - FLAML'):
        start_lgb30_FLAM = time.time()
        model = LGBMClassifier(n_estimators=31204, num_leaves=4,
                               min_child_samples=3, learning_rate=0.009033979476164342,
                               log_max_bin=10, colsample_bytree=0.5393339924944204,
                               reg_alpha=15.800090067239827, reg_lambda=34.82471227276953,
                               random_state=42, class_weight='balanced')

        lgb30_FLAM = pipeline_model(model=model,
                                    preprocessor=StandardScaler())

        lgb30_FLAM.fit(X_train[feat_lgb30], y_train)
        duration_lgb30_FLAM = time.time() - start_lgb30_FLAM

        # Evaluation sur les données de test
        (biz_lgb30_FLAM, beta_lgb30_FLAM, rec_lgb30_FLAM, prec_lgb30_FLAM,
         acc_lgb30_FLAM, auc_lgb30_FLAM, y_pred_lgb30_FLAM) = eval_metrics(best_model=lgb30_FLAM,
                                                                           xtest=X_test[feat_lgb30],
                                                                           ytest=y_test,
                                                                           beta_value=9)

        # log des paramètres et scores à chaque fois que le modèle est lancé
        mlflow.log_param("best parameters", {'n_estimators': 31204, 'num_leaves': 4, 'min_child_samples': 3,
                                             'learning_rate': 0.009033979476164342, 'log_max_bin': 10,
                                             'colsample_bytree': 0.5393339924944204, 'reg_alpha': 15.800090067239827,
                                             'reg_lambda': 34.82471227276953})

        mlflow.log_metric("score metier", biz_lgb30_FLAM)
        mlflow.log_metric("f betascore", beta_lgb30_FLAM)
        mlflow.log_metric("recall", rec_lgb30_FLAM)
        mlflow.log_metric("precision", prec_lgb30_FLAM)
        mlflow.log_metric("accuracy", acc_lgb30_FLAM)
        mlflow.log_metric("auc", auc_lgb30_FLAM)
        mlflow.log_metric("Tps_entrainement", duration_lgb30_FLAM)

        mlflow.sklearn.log_model(lgb30_FLAM, "LightGBM_30_FLAML")

        return lgb30_FLAM


def lgb_flaml_no_pipeline_explainer():
    """Fonction charge les données preprocessées, effectue un train test split et qui n'utilise pas
    de pipeline (incompatible avec l'utilisation de LIME). Les 30 features finales sont sélectionnées
    puis standardisées et le modèle LightGBM retenu (lgb_flaml) est entrainé sur les données d'entrainement.
    L'objet LimeTabularExplainer sera également entrainé sur les données d'entrainement standardisées et
    servira par la suite à comprendre la prédiction en se basant sur la contribution des caractéristiques.

    Le modèle, ses paramètres et les metrics servant à son évaluation sont trackés et stockés sur MLFlow.

    Arguments:
    --------------------------------

    return:
    --------------------------------
    lgb: le modèle entrainé sur les données d'entrainement standardisées sans utilisation de pipeline
    explainer : objet entrainé sur les données d'entrainement standardisées qui servira à comprendre la
    prédiction."""

    # Récupération des données preprocessées
    df = pd.read_csv('../Dashboard_API/df.csv', nrows=None)

    # Train test split
    X_train, X_test, y_train, y_test = data_train_test(df)

    # Sélection des features
    feat_lgb30 = ['CREDIT_DURATION', 'EXT_SOURCE_2', 'INSTAL_DAYS_PAST_DUE_MEAN',
                  'PAYMENT_RATE', 'POS_CNT_INSTALMENT_FUTURE_MEAN', 'CREDIT_GOODS_PERC',
                  'AGE', 'POS_NB_CREDIT', 'BURO_CREDIT_ACTIVE_Active_SUM', 'BURO_AMT_CREDIT_SUM_DEBT_MEAN',
                  'YEARS_EMPLOYED', 'YEARS_ID_PUBLISH', 'INSTAL_PAYMENT_DIFF_MEAN', 'BURO_AMT_CREDIT_SUM_MEAN',
                  'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'BURO_YEARS_CREDIT_ENDDATE_MEAN', 'AMT_CREDIT',
                  'YEARS_LAST_PHONE_CHANGE', 'POS_MONTHS_BALANCE_MEAN', 'INSTAL_DAYS_BEFORE_DUE_MEAN',
                  'BURO_AMT_CREDIT_SUM_DEBT_SUM', 'CODE_GENDER', 'PREV_YEARS_DECISION_MEAN',
                  'REGION_POPULATION_RELATIVE', 'DEBT_RATIO', 'BURO_AMT_CREDIT_SUM_SUM',
                  'BURO_YEARS_CREDIT_ENDDATE_MAX', 'NAME_EDUCATION_TYPE_Lower Secondary & Secondary',
                  'PREV_PAYMENT_RATE_MEAN']

    # Preprocessing des features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[feat_lgb30])
    X_test_scaled = scaler.transform(X_test[feat_lgb30])

    # Sauvegarde du scaler sur le disque
    dump(scaler, open('../Dashboard_API/credit_score_model_scaler.sav', 'wb'))

    # Modèle LightGBM retenu non intégré dans un objet pipeline
    mlflow.set_experiment("mlflow-default-risk")
    experiment = mlflow.get_experiment_by_name("mlflow-default-risk")
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name='Lgb - no pipeline'):
        start_lgb_no_pipe = time.time()
        credit_score_model_SHAP = LGBMClassifier(n_estimators=31204,
                                                 num_leaves=4,
                                                 min_child_samples=3,
                                                 learning_rate=0.009033979476164342,
                                                 colsample_bytree=0.5393339924944204,
                                                 reg_alpha=15.800090067239827,
                                                 reg_lambda=34.82471227276953,
                                                 random_state=42,
                                                 class_weight='balanced')
        credit_score_model_SHAP.fit(X_train_scaled, y_train)
        duration_lgb_no_pipe = time.time() - start_lgb_no_pipe

        # Sauvegarde du modèle sur le disque
        dump(credit_score_model_SHAP, open('../Dashboard_API/credit_score_model_SHAP.sav', 'wb'))

        # Evaluation sur les données de test
        (biz_lgb_no_pipe, beta_lgb_no_pipe, rec_lgb_no_pipe, prec_lgb_no_pipe,
         acc_lgb_no_pipe, auc_lgb_no_pipe, y_pred_lgb_no_pipe) = eval_metrics(best_model=credit_score_model_SHAP,
                                                                              xtest=X_test_scaled,
                                                                              ytest=y_test,
                                                                              beta_value=9)

        # log des paramètres et scores à chaque fois que le modèle est lancé
        mlflow.log_param("best parameters", {'n_estimators': 31204, 'num_leaves': 4, 'min_child_samples': 3,
                                             'learning_rate': 0.009033979476164342,
                                             'colsample_bytree': 0.5393339924944204, 'reg_alpha': 15.800090067239827,
                                             'reg_lambda': 34.82471227276953})

        mlflow.log_metric("score metier", biz_lgb_no_pipe)
        mlflow.log_metric("f betascore", beta_lgb_no_pipe)
        mlflow.log_metric("recall", rec_lgb_no_pipe)
        mlflow.log_metric("precision", prec_lgb_no_pipe)
        mlflow.log_metric("accuracy", acc_lgb_no_pipe)
        mlflow.log_metric("auc", auc_lgb_no_pipe)
        mlflow.log_metric("Tps_entrainement", duration_lgb_no_pipe)

        mlflow.sklearn.log_model(credit_score_model_SHAP, "credit_score_model_SHAP")

        # Création de l'objet shap.TreeExplainer basé sur les données d'entrainement
        # explainer = shap.TreeExplainer(credit_score_model_SHAP, X_train_scaled)
        explainer = shap.TreeExplainer(credit_score_model_SHAP, model_output="probability")

        # Sauvegarde du modèle sur le disque
        with open('../credit_score_model_SHAP_explainer.sav', 'wb') as f:
            dill.dump(explainer, f)

        # Création de l'objet LimeTabularExplainer basé sur les données d'entrainement
        # explainer = lime_tabular.LimeTabularExplainer(X_train_scaled,
        #                                              mode="classification",
        #                                              feature_names=feat_lgb30,
        #                                              random_state=42)

        # Sauvegarde du modèle sur le disque
        # with open('credit_score_model_explainer.sav', 'wb') as f:
        #    dill.dump(explainer, f)

        return credit_score_model_SHAP, explainer, scaler


# --------------------------------------------------------------------
# --------------------- FEATURES IMPORTANCE --------------------------
# --------------------------------------------------------------------

def LIME_explainer(xtrain, loan_sample, model):
    """Fonction qui n'utilise pas de pipeline de transformation des données. Les 30 features finales
    sont sélectionnées et standardisées puis le modèle LightGBM retenu (lgb_flaml) est entrainé
    sur ces données.

    Le modèle, ses paramètres et les metrics servant à son évaluation sont trackés et stockés sur MLFlow.

    Arguments:
    --------------------------------
    xtrain: données pour entrainer le modèle de scoring
    xtest : données pour tester le modèle de scoring
    ytrain : variable à prédire (défaillant, non défaillant) des données d'entrainement
    ytest : variable à prédire (défaillant, non défaillant) des données de test

    return:
    --------------------------------
    lgb: le modèle entrainé sur données standardisées sans utilisation de pipeline"""

    # Sélection des features
    feat_lgb30 = ['CREDIT_DURATION', 'EXT_SOURCE_2', 'INSTAL_DAYS_PAST_DUE_MEAN',
                  'PAYMENT_RATE', 'POS_CNT_INSTALMENT_FUTURE_MEAN', 'CREDIT_GOODS_PERC',
                  'AGE', 'POS_NB_CREDIT', 'BURO_CREDIT_ACTIVE_Active_SUM', 'BURO_AMT_CREDIT_SUM_DEBT_MEAN',
                  'YEARS_EMPLOYED', 'YEARS_ID_PUBLISH', 'INSTAL_PAYMENT_DIFF_MEAN', 'BURO_AMT_CREDIT_SUM_MEAN',
                  'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'BURO_YEARS_CREDIT_ENDDATE_MEAN', 'AMT_CREDIT',
                  'YEARS_LAST_PHONE_CHANGE', 'POS_MONTHS_BALANCE_MEAN', 'INSTAL_DAYS_BEFORE_DUE_MEAN',
                  'BURO_AMT_CREDIT_SUM_DEBT_SUM', 'CODE_GENDER', 'PREV_YEARS_DECISION_MEAN',
                  'REGION_POPULATION_RELATIVE', 'DEBT_RATIO', 'BURO_AMT_CREDIT_SUM_SUM',
                  'BURO_YEARS_CREDIT_ENDDATE_MAX', 'NAME_EDUCATION_TYPE_Lower Secondary & Secondary',
                  'PREV_PAYMENT_RATE_MEAN']

    # Preprocessing des features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(xtrain[feat_lgb30])
    loan_sample = scaler.transform(loan_sample)

    # Création de l'objet LimeTabularExplainer basé sur les données d'entrainement
    explainer = lime_tabular.LimeTabularExplainer(X_train_scaled,
                                                  mode="classification",
                                                  feature_names=feat_lgb30,
                                                  random_state=42)

    explanation = explainer.explain_instance(loan_sample,
                                             model.predict_proba)

    explanation.show_in_notebook()

    return explanation


if __name__ == '__main__':
    lgb_flaml_no_pipeline_explainer()