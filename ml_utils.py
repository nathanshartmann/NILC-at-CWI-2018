from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from matplotlib import pyplot as plt

import numpy as np

def grid_search_classifiers(model, params):
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=kfold, scoring='roc_auc')
    grid_result = grid_search.fit(x_train, y_train)
    print(grid_result.best_params_)
    
def grid_search_regressors(model, params):
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=kfold, scoring='neg_mean_absolute_error')
    grid_result = grid_search.fit(x_train, y_train)
    print(grid_result.best_params_)
    
def print_roc(pred, y, message='Roc curve'):
    fpr, tpr, _ = roc_curve(y.ravel(), pred.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(message)
    plt.legend(loc="lower right")
    plt.show()
    
def best_threshold(pred, y, metric='accuracy'):
    best = 0
    th = 0
    if metric == 'precision':
        metric = precision_score
    elif metric == 'recall':
        metric = recall_score
    elif metric == 'f1':
        metric = f1_score
    else:
        metric = accuracy_score
    for i in np.arange(pred.min(), pred.max(), 0.01):
        aux = pred.copy()
        aux[aux >= i] = 1
        aux[aux < i] = 0
        new = metric(aux, y)
        if new > best:
            best = new
            th = i
    return th

def evaluate(pred_train, pred_test, y_train, y_test, optimize='accuracy', label='ROC Curve'):
    print('--Optimizing %s--' % optimize)
    th = best_threshold(pred_train, y_train, optimize)
    print('Threshold: %.2f' % th)
    aux = pred_test.copy()
    aux[aux >= th] = 1
    aux[aux < th] = 0
    print('--Scores--')
    print('Accuracy: %.2f' % accuracy_score(aux, y_test))
    print('Precision: %.2f' % precision_score(aux, y_test))
    print('Recall: %.2f' % recall_score(aux, y_test))
    print('F1: %.2f' % f1_score(aux, y_test))
    print('--Confusion matrix:--\n %s' % confusion_matrix(aux, y_test))
    print_roc(pred_test, y_test, label)