from random import random

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier


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
    data___ = pipe.named_steps['pca'].transform(data__)
    return data___


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
    'data/desafiocartola.csv')
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

pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Only numeric columns
    ('center', StandardScaler()),
    ('pca', PCA(n_components=15)),
    ('sgd', SGDClassifier(loss='log', verbose=5,
                          early_stopping=True, validation_fraction=0.3))
]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, target, train_size=0.7, random_state=0)

pipe.fit(X_train, y_train)

X_train = get_transformed_data(X_train, pipe)
y_train_pred_prob = pipe.named_steps['sgd'].predict_proba(X_train)[:, 1]

X_test = get_transformed_data(X_test, pipe)
y_test_pred_prob = pipe.named_steps['sgd'].predict_proba(X_test)[:, 1]

fig, ax = roc_pr_curve(y_test, y_test_pred_prob, show=True)
plt.show()
