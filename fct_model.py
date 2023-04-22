''' Librairie personnelle de fonctions de ML
'''

import time

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, cross_validate, KFold, RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, fbeta_score
from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as pipe

# --------------------------------------------------------------------
# ------------------------- ENTRAINEMENT -----------------------------
# --------------------------------------------------------------------  


def pipeline_model(model, numeric_features, numeric_transformer):

    # Transformations à effectuer sur nos variables
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),])

    # Définition de la pipeline du modèle: étapes de preprocessing + classifier
    pipeline_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)])
    
    return pipeline_model



def pipeline_model_balanced(model, numeric_features, numeric_transformer,
                            oversampling_strategy, undersampling_strategy):

    # Sur échantillonnage de la classe minoritaire (10% de la classe majoritaire ~= 23000)
    oversampler = SMOTE(sampling_strategy = oversampling_strategy, random_state = 42)

    # Sous échantillonnage pour réduire la classe majoritaire (50% de plus que la classe minoritaire ~= 46000
    undersampler = RandomUnderSampler(sampling_strategy = undersampling_strategy, random_state = 42)
    
    # Transformations à effectuer sur nos variables
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),])
    
    # Définition de la pipeline du modèle: étapes de preprocessing + classifier
    pipeline_model_balanced = pipe(steps=[
        ('over', oversampler),
        ('under', undersampler),
        ('preprocessor', preprocessor),
        ('classifier', model)])
    
    return pipeline_model_balanced



def optimize_and_train_model(pipeline_model, xtrain, ytrain, params, scoring, cv):
    
    _ = pipeline_model.fit(xtrain, ytrain)


    # Réglage automatique des meilleurs hyperparamètres avec GridSearchCV
    #inner_cv = KFold(n_splits = 5, shuffle = True, random_state = 42)
    model_grid_cv = GridSearchCV(pipeline_model, 
                                 param_grid = params, 
                                 cv = cv, 
                                 scoring = scoring,
                                 refit=True)

    model_grid_cv.fit(xtrain, ytrain)
    
    best_model = model_grid_cv.best_estimator_
    best_params = model_grid_cv.best_params_

    return best_model, best_params



def optimize_and_train_model_RSCV(pipeline_model, xtrain, ytrain, params, scoring, cv):
    
    _ = pipeline_model.fit(xtrain, ytrain)


    # Réglage automatique des meilleurs hyperparamètres avec GridSearchCV
    #inner_cv = KFold(n_splits = 5, shuffle = True, random_state = 42)
    model_grid_cv = RandomizedSearchCV(pipeline_model, 
                                       param_distributions = params, 
                                       cv = cv, 
                                       scoring = scoring,
                                       refit=True)

    model_grid_cv.fit(xtrain, ytrain)
    
    best_model = model_grid_cv.best_estimator_
    best_params = model_grid_cv.best_params_

    return best_model, best_params



def best_model(model_name, model, cv,
               xtrain, numeric_features, numeric_transformer, 
               ytrain, params, scoring, xtest, ytest, 
               oversampling_strategy = 0.1, undersampling_strategy = 0.5, balanced = False, Randomized = False):
    
    if balanced == False:
        start = time.time()
        model = pipeline_model(model = model,
                               numeric_features = numeric_features,
                               numeric_transformer = numeric_transformer)

        if RandomizedSearchCV == False:
            # Optimisation via cross validation & GridSearch
            best_model, best_params = optimize_and_train_model(pipeline_model = model,
                                                               xtrain = xtrain[numeric_features],
                                                               ytrain = ytrain,
                                                               params = params,
                                                               scoring = scoring,
                                                               cv = cv)
            
        else:
            # Optimisation via cross validation & RandomizedSearch
            best_model, best_params = optimize_and_train_model_RSCV(pipeline_model = model,
                                                                    xtrain = xtrain[numeric_features],
                                                                    ytrain = ytrain,
                                                                    params = params,
                                                                    scoring = scoring,
                                                                    cv = cv)


        duration = time.time() - start
        
    else:
        
        start = time.time()
        model = pipeline_model_balanced(model, numeric_features, numeric_transformer,
                                        oversampling_strategy = oversampling_strategy, 
                                        undersampling_strategy = undersampling_strategy)

        if RandomizedSearchCV == False:
        # Optimisation via cross validation & GridSearch
            best_model, best_params = optimize_and_train_model(pipeline_model = model,
                                                               xtrain = xtrain[numeric_features],
                                                               ytrain = ytrain,
                                                               params = params,
                                                               scoring = scoring,
                                                               cv = cv)
        
        else:
            best_model, best_params = optimize_and_train_model_RSCV(pipeline_model = model,
                                                                    xtrain = xtrain[numeric_features],
                                                                    ytrain = ytrain,
                                                                    params = params,
                                                                    scoring = scoring,
                                                                    cv = cv)    


        duration = time.time() - start
    
    return best_model, best_params, model_name, duration



# --------------------------------------------------------------------
# ------------------------- EVALUATIONS ------------------------------
# --------------------------------------------------------------------            

def score_metier(ytest, y_pred):
    # Matrice de confusion transformée en array avec affectation aux bonnes catégories
    (vn, fp, fn, vp) = confusion_matrix(ytest, y_pred).ravel()
    
    # Rappel avec action fp => à minimiser
    score_metier = 10*fn + 2*fp
    
    return score_metier




def eval_metrics(best_model, xtest, ytest, beta_value):
    
    y_pred = best_model.predict(xtest)
    
    score_biz = score_metier(ytest, y_pred)
    betascore = fbeta_score(ytest, y_pred, beta = beta_value)
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
                cmap = sns.cubehelix_palette(as_cmap=True), cbar=False)
    plt.title(f'Matrice de confusion: {model_name}')
    plt.ylabel('Réalité')
    plt.xlabel('Prédiction')
    plt.show()

    

def eval_model(model_name, best_model, best_params, xtest, ytest, categ_features):
    # Evaluation du modèle sur les données de test
    (score_biz, recall, precision, accuracy, auc, y_pred) = eval_metrics(best_model = best_model,
                                                                         xtest = xtest,
                                                                         ytest = ytest)
    
    matrice_confusion(ytest, y_pred, model_name)
    
    # Récap
    dic_df_recap = {'Modèle':[model_name],
                    'Variables':[categ_features],
                    'Best_Params':[best_params],
                    'Score_metier':[score_biz],
                    'Recall':[recall], 
                    'Precision':[precision], 
                    'Accuracy':[accuracy], 
                    'AUC':[auc],}
                    #"Train_Time": [duration],
    df_recap = pd.DataFrame(data = dic_df_recap)
    display(df_recap)
    
    return best_model, df_recap