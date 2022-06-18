from random import random
import operator

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

from statsmodels.stats.outliers_influence import variance_inflation_factor


def get_log(value):
    if value is None or value <= -1:
        return np.nan

    return np.log(1+value)


def transform_long_tail_variables(df, long_tail_vars, transformation_type='log'):
    transform_options = {'log': get_log}

    for column in long_tail_vars:
        df[column] = df[column].apply(transform_options[transformation_type])
        df.rename(columns={column: 'log_' + column}, inplace=True)

    return df


def get_total_pixels(pixels):
    if (pixels is None) | (not isinstance(pixels, str)):
        return None
    if ('x' not in pixels) | (len(pixels.split('x')) <= 1):
        return None

    splitted_pixel_values = pixels.split('x')
    return float(splitted_pixel_values[0])*float(splitted_pixel_values[1])


def calculate_total_screen_pixels(df):
    df['pixels'] = df['pixels'].apply(get_total_pixels)
    df.rename(columns={'pixels': 'total_pixels'}, inplace=True)
    return df


def set_max_value(df, col_list, max_value):
    df[col_list] = df[col_list].clip(upper=max_value)
    df.rename(
        columns={col: 'ceil_'+col for col in col_list}, inplace=True)
    return df


def get_transformed_data(data, pipe):
    data_ = pipe.named_steps['imputer'].transform(data)
    data__ = pipe.named_steps['center'].transform(data_)
    return data__


def calculate_ks_score(model_prob):
    preds = pd.DataFrame(columns=['p0', 'p1'])
    preds['p0'] = model_prob
    preds['p1'] = 1-preds['p0']
    preds['predicted'] = np.where(preds['p0'] < 0.5, 0, 1)
    preds[target_name] = y_test.values

    preds_true = preds[preds[target_name] == 1]['p0']
    preds_false = preds[preds[target_name] == 0]['p0']

    ks_score = ks_2samp(preds_true, preds_false)[0]
    return ks_score


def calculate_vif_value(input_df):
    input_df['intercept'] = [1] * len(input_df)

    vif = pd.DataFrame()
    vif['feature'] = input_df.columns
    vif['VIF'] = [
        variance_inflation_factor(input_df.values, i) for i in range(len(input_df.columns))
    ]

    vif.drop(vif.index[vif['feature'] == 'intercept'], inplace=True)
    return vif


def get_feature_importance(model, columns, plot_importance=False):
    importance = model.coef_[0]

    features = {}

    for index, score in enumerate(importance):
        features[columns[index]] = abs(score)

    sorted_feature_imp = sorted(features.items(), key=operator.itemgetter(1))

    if plot_importance:
        plt.bar([x for x in range(len(importance))], importance)
        plt.show()

    return sorted_feature_imp


def eval_model(X_features, y_target, pipe):
    X_features = get_transformed_data(X_features, pipe)
    y_pred_prob = pipe.named_steps['sgd'].predict_proba(X_features)[:, 1]

    test_predictions = np.where(y_pred_prob < 0.5, 0, 1)

    precision = precision_score(y_target, test_predictions)
    recall = recall_score(y_target, test_predictions)
    ks_score = calculate_ks_score(y_pred_prob)

    # fig, ax = roc_pr_curve(y_target, y_pred_prob, show=True)
    # plt.show()

    return precision, recall, ks_score


def get_colinear_features(input_data, columns):
    imp = SimpleImputer(strategy='mean')
    input_data = imp.fit_transform(input_data)
    colinear_features = []
    dataframe = pd.DataFrame(input_data, columns=columns)

    vif = calculate_vif_value(dataframe)

    while len(vif[vif['VIF'] > 3]):
        feature_name = vif.sort_values(
            by=['VIF'], ascending=False)['feature'].values[0]
        colinear_features.append(feature_name)
        dataframe.drop(columns=[feature_name], inplace=True)
        vif = calculate_vif_value(dataframe)

    return colinear_features


def roc_pr_curve(y, y_pred_prob, show=True):

    fig, axes = plt.subplots(figsize=(16, 10), ncols=2, nrows=1)

    y_random = [random() for _ in range(len(y))]

    ms_fpr, ms_tpr, _ = roc_curve(y, y_pred_prob)
    rs_fpr, rs_tpr, _ = roc_curve(y, y_random)

    ms_precision, ms_recall, _ = precision_recall_curve(y, y_pred_prob)
    rs_precision, rs_recall, _ = precision_recall_curve(y, y_random)

    ax = axes[0]
    ax.plot(ms_fpr, ms_tpr, color='#F76800')
    ax.plot(rs_fpr, rs_tpr, linestyle='--', color='#333333')
    ax.grid(True)
    ax.set_title('ROC Curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.scatter(
        [], [], label=f'AUC: {round(roc_auc_score(y, y_pred_prob), 2)}', color='#F76800')
    ax.legend(loc='lower right')

    ax = axes[1]
    ax.plot(ms_recall, ms_precision, color='#F76800')
    ax.plot(rs_recall, rs_precision, linestyle='--', color='#333333')
    ax.grid(True)
    ax.set_title('Precision-Recall Curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.scatter(
        [], [], label=f'Random:  {round(auc(rs_recall, rs_precision), 2)}', color='#333333')
    ax.scatter(
        [], [], label=f'Model: {round(auc(ms_recall, ms_precision), 2)}', color='#F76800')
    ax.legend(loc='upper right')

    if not show:
        plt.close(fig)

    return fig, ax


df = pd.read_csv(
    '/home/alvaro/Downloads/Desafio_Base_Eng_ML/Desafio_Eng_ML/desafiocartola.csv')
df.set_index('GLOBO_ID', inplace=True)

target_name = 'pro_target'
target = df[target_name]

long_tail_variables = ['anos_desde_criacao', 'instagram_num', 'facebook_num',
                       'min_camp', 'interacoes_g1', 'tempo_desperd',
                       'iteracao_volei', 'iteracao_atletismo']

df = transform_long_tail_variables(df, long_tail_variables)
df = calculate_total_screen_pixels(df)
df = set_max_value(df, ['avg_3', 'avg_4'], 1)

X = df.drop(target_name, axis=1).copy()
columns = list(X.columns)

pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Only numeric columns
    ('center', StandardScaler()),
    ('sgd', LogisticRegression(random_state=0))
]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, target, train_size=0.7, random_state=0)

colinear_cols = get_colinear_features(X_train, columns)

X_train.drop(columns=colinear_cols, inplace=True)
X_test.drop(columns=colinear_cols, inplace=True)

pipe.fit(X_train, y_train)

precision, recall, ks_score = eval_model(X_test, y_test, pipe)
print(precision, recall, ks_score)

for _ in range(8):
    feature_imp = get_feature_importance(
        pipe.named_steps['sgd'], X_train.columns)
    feature = [feature_imp[0][0]]

    print(f'Removing feature {feature}')
    X_train.drop(columns=feature, inplace=True)
    X_test.drop(columns=feature, inplace=True)
    pipe.fit(X_train, y_train)
    precision, recall, ks_score = eval_model(X_test, y_test, pipe)
    print(precision, recall, ks_score)
