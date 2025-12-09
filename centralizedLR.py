import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, log_loss
from sklearn.model_selection import train_test_split

from logisticRegression import LogisticRegression

from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.linear_model import SGDClassifier
import gensim.downloader as api

from data_utils import load_dataset
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

def evaluate_model(x, y, w, prefix=''):
    if len(y) == 0: return {}
    
    pred_prob = sigmoid(np.dot(x, w))
    y_pred = (pred_prob >= 0.5).astype(int)
    
    return {
        f'{prefix}accuracy': accuracy_score(y, y_pred),
        f'{prefix}f1': f1_score(y, y_pred, zero_division=0),
        f'{prefix}precision': precision_score(y, y_pred, zero_division=0),
        f'{prefix}recall': recall_score(y, y_pred, zero_division=0),
        f'{prefix}loss': log_loss(y, pred_prob) 
    }

def train_centralized(X_train, y_train, X_test, y_test, epochs: int = 100, lambda_reg: float = 0.01, learning_rate = 0.5):
    m_train, n_features = X_train.shape
    # Aggiunta del bias
    X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    history = []
    print(f"Training Centralized Custom Logistic Regression using TF-IDF stats:")
    print("Num features (incluso bias):", X_train_b.shape[1])
    weights = np.zeros(X_train_b.shape[1])
    #weights = np.random.normal(0, 0.01, size=(n_features + 1))
    for epoch in range(epochs):
        predictions = sigmoid(np.dot(X_train_b, weights))
        errors = predictions - y_train
        gradient = np.dot(X_train_b.T, errors) / m_train

        gradient[1:] += (lambda_reg / m_train) * weights[1:]
        weights -= learning_rate * (gradient)
        #loss = compute_loss(weights, X_train_b, y_train, lambda_reg)
        metrics = evaluate_model(X_test_b, y_test, weights)
        #print(metrics)
        if epoch % 5 == 0:
            print(f"EPOCH {epoch}: Accuracy {metrics['accuracy']:.4f}, Precision {metrics['precision']:.4f}, Recall {metrics['recall']:.4f}, F1-Score {metrics['f1']:.4f}, Loss {metrics['loss']:.4f}")
        metrics['epoch'] = epoch
        history.append(metrics)

    #print(classification_report(y_test, y_pred))
    return weights, history

def train_centralized_sklearn(X_train, y_train, X_test, y_test, epochs: int = 300, lambda_reg= 0.01, lr= 0.5):
    print(f"Centralized Logistic Regression SKLEARN TF-IDF stats:")
    #centralized_model, centralized_history = train_centralized(X_train, y_train, X_test, y_test, epochs=300)
    #sk_model = SklearnLogisticRegression(max_iter=epochs, C=100, solver='sag')
    sk_model = SGDClassifier(
        loss='log_loss', 
        penalty='l2', 
        alpha=lambda_reg, 
        learning_rate='constant', 
        eta0=lr, 
        max_iter=epochs,          # Facciamo 1 epoca alla volta manualmente per tracciare
        warm_start=True,     # Mantiene i pesi tra una chiamata e l'altra
        shuffle=True,        # Importante per SGD
        random_state=42
    )
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
    print(classification_report(y_test, y_pred))

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
    }


def message_to_vector(message, model, dim):
    """
    Trasforma una frase in un singolo vettore facendo la media
    dei vettori delle parole che la compongono.
    """
    words = str(message).lower().split()
    vector_sum = np.zeros(dim)
    count = 0
    
    for word in words:
        if word in model:
            vector_sum += model[word]
            count += 1
            
    if count > 0:
        return vector_sum / count
    else:
        # Se nessuna parola è nel vocabolario, ritorna vettore di zeri
        return np.zeros(dim)
    
#word_vectors = api.load("glove-wiki-gigaword-100")
#EMBEDDING_DIM = 100
#X_vectors = np.array([message_to_vector(msg, word_vectors, EMBEDDING_DIM) for msg in df_full['message']])
#scaler = StandardScaler()
#X_vectors = scaler.fit_transform(X_vectors)
#print("most similar to cat", word_vectors.most_similar("cat"))
"""
df_full = load_dataset()
print("Dataset full")
print(df_full.head())
print(df_full.shape)
print(df_full['spam'].unique())

y = df_full['spam'].values
x_raw = df_full['message']
X_train, X_test, y_train, y_test = train_test_split(x_raw, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(max_features=2**12, stop_words='english')
vectorizer.fit(X_train)
X_train_vec = vectorizer.transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()

#print("features vectorized", vectorizer.get_feature_names_out())
print(f"Dimensione del vocabolario: {len(vectorizer.get_feature_names_out())}")

# Conversione esplicita a numpy array di int
y_train = np.array(y_train).astype(int)
y_test = np.array(y_test).astype(int)

train_centralized_sklearn(X_train_vec, y_train, X_test_vec, y_test, epochs=100)
train_centralized(X_train_vec, y_train, X_test_vec, y_test, epochs=1000, lambda_reg=0, learning_rate=3.0)

"""