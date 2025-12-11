import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, log_loss

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

# Funzione Sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Funzione di costo (Binary Cross Entropy) con L2 Regularization opzionale
def compute_loss(w, X, y, lambda_reg=0):
    m = len(y)
    h = sigmoid(np.dot(X, w))
    loss = (-1/m) * np.sum(y * np.log(h + 1e-9) + (1-y) * np.log(1-h + 1e-9))

    l2_penalty = (lambda_reg / (2 * m)) * np.sum(w[1:] ** 2)
    return loss + l2_penalty

def predict(X, weights, threshold=0.5):
    prob = sigmoid(np.dot(X, weights))
    return (prob >= threshold).astype(int)

def train_centralized_sklearn_sgd(X_train, y_train, X_test, y_test, epochs: int = 300, lambda_reg= 0.01, lr= 0.5):
    print(f"Centralized Logistic Regression using SGDClassifier")
    sk_model = SGDClassifier(
        loss='log_loss', 
        penalty='l2', 
        alpha=lambda_reg, 
        learning_rate='constant', 
        eta0=lr, 
        max_iter=epochs,          
        warm_start=True,     
        shuffle=True,      
        random_state=42
    )
    sk_model.fit(X_train, y_train)
    y_pred = sk_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
    }

def train_centralized_sklearn_lr(X_train, y_train, X_test, y_test, epochs: int = 300, c=100):
    print(f"Centralized Logistic Regression using default lbfgs solver")
    sk_model = LogisticRegression(max_iter=epochs, C=c, solver='lbfgs')

    sk_model.fit(X_train, y_train)
    y_pred = sk_model.predict(X_test)

    #y_pred = centralized_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
    }

def grid_search_centralsgd(X_train, y_train, X_valid, y_valid):
    param_grid = {
        'alpha': [0.001, 0.0001, 0.01, 0.1],
        'eta0': [0.1, 0.5, 1.0, 2.0],
        'max_iter': [300, 500],
        'penalty': ['l2'],
        'learning_rate': ['constant'],
    }
    print("Starting Grid Search for Centralized SGDClassifier...")
    sgd_model = SGDClassifier(
        loss='log_loss', 
        penalty='l2', 
        learning_rate='constant', 
        warm_start=True,     
        shuffle=True,      
        random_state=42
    )
    grid = GridSearchCV(
        sgd_model, param_grid, 
        scoring='f1', 
        cv=[(np.arange(len(y_train)), np.arange(len(y_valid)))], # usa validation set
        refit=True
    )
    X_all = np.vstack([X_train, X_valid])
    y_all = np.concatenate([y_train, y_valid])
    grid.fit(X_all, y_all)
    print("Best parameters:", grid.best_params_)
    print("Best F1 score:", grid.best_score_)

    return grid.best_estimator_

def test_centralsgd(model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Test Set Performance Centralized sgd:")
    print(model)
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
    }