import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from random import random
from sklearn.metrics import confusion_matrix
import seaborn as sns


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
import numpy as np

df = pd.read_csv('/home/alvaro/Downloads/Desafio_Base_Eng_ML/Desafio_Eng_ML/desafiocartola.csv')
df.set_index('GLOBO_ID',inplace=True)

lista_columns_drop = ['pro_target']
target = df['pro_target']

long_tail_variables = ['anos_desde_criacao', 'instagram_num', 'facebook_num', 'min_camp', 'interacoes_g1', 'tempo_desperd', 'iteracao_volei', 'iteracao_atletismo']

def get_long_tail(long):
    if long is None or long <= -1:
        return np.nan
    else:
        return np.log(1+long)

for column in long_tail_variables:
    df['log_' + column] = df[column].apply(get_long_tail)
    lista_columns_drop.append(column)

def get_total_pixels(pixels):
    if (pixels is None) | (type(pixels) != str):
        return None
    elif ('x' not in pixels) | (len(pixels.split('x'))<= 1):
        return None
    return float(pixels.split('x')[0])*float(pixels.split('x')[1])


df['total_pixels'] = df['pixels'].apply(get_total_pixels)
lista_columns_drop.append('pixels')

ceil = lambda x: 1 if x >= 1 else x
for column in ['avg_3', 'avg_4']:
    df['ceil_' + column] = df[column].apply(ceil)
    lista_columns_drop.append(column)

X = df.drop(lista_columns_drop,axis=1).copy()

pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  #Only numeric columns
        ('center', StandardScaler()),
        ('pca', PCA(n_components=15)),
        ('sgd', SGDClassifier(loss='log', verbose=5, early_stopping=True, validation_fraction=0.3))
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, target, train_size=0.7)

pipe.fit(X_train,y_train)

X_train_ = pipe.named_steps['imputer'].fit_transform(X_train)
X_train__ = pipe.named_steps['center'].fit_transform(X_train_)
X_train___ = pipe.named_steps['pca'].fit_transform(X_train__)
y_train_pred_prob = pipe.named_steps['sgd'].predict_proba(X_train___)[:,1]

X_test_ = pipe.named_steps['imputer'].fit_transform(X_test)
X_test__ = pipe.named_steps['center'].fit_transform(X_test_)
X_test___ = pipe.named_steps['pca'].fit_transform(X_test__)
y_test_pred_prob = pipe.named_steps['sgd'].predict_proba(X_test___)[:,1]


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
    ax.scatter([], [], label='AUC: {:.02f}'.format(roc_auc_score(y, y_pred_prob)), color='#F76800')    
    ax.legend(loc='lower right')

    ax = axes[1]
    ax.plot(ms_recall, ms_precision, color='#F76800')
    ax.plot(rs_recall, rs_precision, linestyle='--', color='#333333')
    ax.grid(True)
    ax.set_title('Precision-Recall Curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.scatter([], [], label='Random: {:.02f}'.format(auc(rs_recall, rs_precision)), color='#333333')
    ax.scatter([], [], label='Model: {:.02f}'.format(auc(ms_recall, ms_precision)), color='#F76800')
    ax.legend(loc='upper right')

    if not show:
        plt.close(fig)

    return fig, ax

fig, ax = roc_pr_curve(y_test, y_test_pred_prob, show=True)

joblib.dump(pipe,'propension_model.joblib')