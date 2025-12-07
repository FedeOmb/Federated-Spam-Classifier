import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from data_utils import load_dataset, split_preprocess_data_non_iid, split_preprocess_data_iid, visualize_distribution

df_full, x, y = load_dataset()

# Separazione Dati Pubblici e Privati

X_server_raw, X_clients_raw, y_server, y_clients = train_test_split(
    df_full['message'], df_full['spam'], test_size=0.90, random_state=42, stratify=df_full['spam']
)

print(f"Dati 'Pubblici' sul Server: {len(X_server_raw)}")
print(f"Dati 'Privati' sui Client: {len(X_clients_raw)}")

# Il server crea il vettorizzatore basandosi solo sui dati pubblici.
server_vectorizer = TfidfVectorizer(max_features=2**12, stop_words='english')
server_vectorizer.fit(X_server_raw) 

print("Vocabolario costruito dal server.")
#print("features vectorized", vectorizer.get_feature_names_out())
print(f"Dimensione del vocabolario: {len(server_vectorizer.get_feature_names_out())}")

hash_vectorizer = HashingVectorizer(
    n_features=2**12, 
    alternate_sign=False,  # Mantieni valori positivi come in BoW
    norm='l2',              # Normalizzazione locale (simula parzialmente TF-IDF)
    stop_words='english',
    dtype=np.float32              
)
# Distribuzione ai Client e Preprocessing Locale

NUM_CLIENTS = 10

X_clients_vals = X_clients_raw.values
print("X_clients_vals", type(X_clients_vals))
y_clients_vals = y_clients.values

# suddivisione dati privati in train federato e test set globale

X_train_fed, X_test_fed, y_train_fed, y_test_fed = train_test_split(
    X_clients_vals, y_clients_vals, test_size=0.2, random_state=42
)

print("X_train_fed", X_train_fed.shape)
print("X_test_fed", X_test_fed.shape)


client_datasets, stats = split_preprocess_data_iid(X_train_fed, y_train_fed, server_vectorizer, NUM_CLIENTS)
#client_datasets, stats = split_preprocess_data_non_iid(X_train_fed, y_train_fed, server_vectorizer, NUM_CLIENTS, alpha=1.0)
print(len(client_datasets))
print(len(client_datasets[0]['x']))
print(len(client_datasets[0]['y']))
#print(client_datasets[0])

visualize_distribution(stats)
# vettorizzazione test set globale
X_test_vectorized = server_vectorizer.transform(X_test_fed).toarray()
y_test = y_test_fed
x_test_bias = np.c_[np.ones((X_test_vectorized.shape[0], 1)), X_test_vectorized]


print(f"Dati distribuiti e vettorizzati localmente su {NUM_CLIENTS} client.")
print(f"Dimensione feature (fissa): {X_test_vectorized.shape[1]}")


# Funzioni comuni logistic regression

def sigmoid(z):
    z = np.clip(z, -250, 250)
    return 1 / (1 + np.exp(-z))

def compute_loss(w, X, y, lambda_reg=0):
    m = len(y)
    h = sigmoid(np.dot(X, w))
    loss = (-1/m) * np.sum(y * np.log(h + 1e-9) + (1-y) * np.log(1-h + 1e-9))

    l2_penalty = (lambda_reg / (2 * m)) * np.sum(w[1:] ** 2)
    return loss + l2_penalty

def predict(X, weights, threshold=0.5):
    prob = sigmoid(np.dot(X, weights))
    return (prob >= threshold).astype(int)

def train_fedavg(client_datasets, num_features, x_test_bias, y_test, rounds=50, local_epochs=10, lr=0.5, lambda_reg=0.01):
    # inizializzazione pesi globali + bias
    global_w = np.zeros(num_features + 1)
    acc_history = []
    
    print("\n--- Inizio Training fedavg ---")
    for r in range(rounds):
        local_weights = []
        total_samples = 0
        
        for i in range(NUM_CLIENTS):
            # Caricamento dati locali (già vettorizzati localmente) 
            x_local = client_datasets[i]['x']
            y_local = client_datasets[i]['y']

            n_local = len(y_local)

            if n_local == 0: continue # Salta client vuoti

            w_local = np.copy(global_w)
            x_local_bias = np.c_[np.ones((x_local.shape[0], 1)), x_local]

            # Training Locale
            for _ in range(local_epochs):
                predictions = sigmoid(np.dot(x_local_bias, w_local))
                err = predictions - y_local
                gradient = np.dot(x_local_bias.T, err) / n_local
                #regolarizzazione L2
                gradient[1:] += (lambda_reg / n_local) * w_local[1:]
                w_local -= lr * (gradient)

            local_weights.append((w_local, n_local))
            total_samples += n_local
        
        # Aggregazione pesata
        if total_samples > 0:
            weighted_weights = np.zeros_like(global_w)
            for w, n in local_weights:
                weighted_weights += w * n
            global_w = weighted_weights / total_samples
        
        # Valutazione sul Server, con test set riservato
        test_preds = predict(x_test_bias, global_w)
        acc = accuracy_score(y_test, test_preds)
        acc_history.append(acc)
        prec = precision_score(y_test, test_preds, zero_division=0)
        rec = recall_score(y_test, test_preds)
        f1 = f1_score(y_test, test_preds)
        if r % 5 == 0:
            print(f"Round {r}: Accuracy {acc:.4f}, Precision {prec:.4f}, Recall {rec:.4f}, F1-Score {f1:.4f}")
            
    return acc_history

def train_fed_admm(client_datasets, num_features, y_test, rounds=50, rho=1.0, local_epochs=5, lr=0.1):
    n_features = num_features + 1
    num_clients = len(client_datasets)

    # Calcolo dei pesi per ogni client in base alla quantità di dati
    client_samples = np.array([len(c['y']) for c in client_datasets])
    total_samples = np.sum(client_samples)
    client_weights = client_samples / total_samples
    
    # Variabili ADMM
    z = np.zeros(n_features)  # Variabile Globale (Consenso)
    u = np.zeros((num_clients, n_features)) # Variabili Duali (una per client)
    w = np.zeros((num_clients, n_features)) # Variabili Primali Locali
    
    acc_history = []
    
    # Preparazione dati (aggiunta bias una volta sola per efficienza)
    #client_datasets_b = [(add_bias_term(c[0]), c[1]) for c in client_datasets]
    #X_val_all = np.vstack([c[0] for c in client_datasets_b])
    #y_val_all = np.concatenate([c[1] for c in client_datasets_b])

    print(f"Inizio ADMM ({rounds} round)...")
    for r in range(rounds):
        
        # 1. Primal Update (w_i): Minimizza Loss locale + Penalità quadratica
        # Non c'è closed form per Logistic Regression, usiamo GD approssimato
        for i in range(num_clients):
            x_local = client_datasets[i]['x']
            y_local = client_datasets[i]['y']
            w_local = np.copy(w[i])
            x_local_bias = np.c_[np.ones((x_local.shape[0], 1)), x_local]
            
            for _ in range(local_epochs):
                predictions = sigmoid(np.dot(x_local_bias, w_local))
                err = predictions - y_local
                grad_loss = np.dot(x_local_bias.T, err) / len(y_local)
                # Gradiente del termine Augmented Lagrangian: rho * (w - z + u)
                grad_penalty = rho * (w_local - z + u[i])
                total_grad = grad_loss + grad_penalty
                total_grad = np.clip(total_grad, -1.0, 1.0)
                # Update
                w_local -= lr * total_grad
            
            w[i] = w_local
            
        # 2. Global Update (z): Media delle variabili locali (consenso)
        # media non pesata
        #z_new = np.mean(w + u, axis=0)
        #media pesata
        local_update = w + u
        z_new = np.average(local_update, axis=0, weights=client_weights)

        # Opzionale: Se volessimo feature selection (Lasso), qui applicheremmo Soft Thresholding
        z = z_new 
        
        # 3. Dual Update (u_i): Aggiorna i moltiplicatori di Lagrange
        for i in range(num_clients):
            u[i] = u[i] + (w[i] - z)
            
        # Monitoraggio
        # Valutazione sul Server, con test set riservato
        test_preds = predict(x_test_bias, z)
        acc = accuracy_score(y_test, test_preds)
        acc_history.append(acc)
        prec = precision_score(y_test, test_preds, zero_division=0)
        rec = recall_score(y_test, test_preds)
        f1 = f1_score(y_test, test_preds)
        if r % 5 == 0:
            print(f"Round {r}: Accuracy {acc:.4f}, Precision {prec:.4f}, Recall {rec:.4f}, F1-Score {f1:.4f}")
        
    return acc_history
# Esecuzione
fedavg_acc_history = train_fedavg(client_datasets, num_features=X_test_vectorized.shape[1], x_test_bias=x_test_bias, y_test=y_test, rounds=100, local_epochs=10, lr=3.0, lambda_reg=0.001)

admm_acc_history = train_fed_admm(client_datasets, num_features=X_test_vectorized.shape[1], y_test=y_test, rounds=100, rho=0.05, local_epochs=20, lr=3.0)
"""
import matplotlib.pyplot as plt
plt.plot(history)
plt.title("FedAvg con TF-IDF 'Pre-Addestrato' (Scenario Realistico)")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.grid()
plt.show()
  """