import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss

# Funzioni comuni logistic regression
def sigmoid(z):
    z = np.clip(z, -250, 250)
    return 1 / (1 + np.exp(-z))

def predict(X, weights, threshold=0.5):
    prob = sigmoid(np.dot(X, weights))
    return (prob >= threshold).astype(int)

def soft_thresholding(x, k):
    return np.sign(x) * np.maximum(0, np.abs(x) - k)

def add_bias(x):
    return np.c_[np.ones((x.shape[0], 1)), x]

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

def train_fedavg(client_datasets, x_test, y_test, rounds=50, local_epochs=10, lr=0.5, lambda_reg=0.01, batch_size=128):

    num_features = x_test.shape[1]
    num_features_b = num_features + 1 
    num_clients = len(client_datasets)
    x_test_bias = add_bias(x_test)
    history = []
    # aggiunta bias ai dati client
    client_datasets_bias = []
    for c in client_datasets:
        x_b = add_bias(c['x'])
        y = c['y']
        client_datasets_bias.append({'x': x_b, 'y': y})

    # inizializzazione pesi globale
    global_w = np.zeros(num_features_b)

    print("--- Start Federated Learning using FedAvg algorithm ---")
    for r in range(rounds):
        local_weights = []
        total_samples = 0
        
        for i in range(num_clients):
            # caricamento dati locali
            x_local = client_datasets_bias[i]['x']
            y_local = client_datasets_bias[i]['y']
            n_local = len(y_local)

            if n_local == 0: continue 

            w_local = np.copy(global_w)
            indices = np.arange(n_local)

            # training locale con SGD
            for j in range(local_epochs):
                if batch_size and batch_size < n_local:
                    np.random.shuffle(indices)

                    for start_idx in range(0, n_local, batch_size):
                        batch_idx = indices[start_idx : start_idx + batch_size]
                        x_batch = x_local[batch_idx]
                        y_batch = y_local[batch_idx]
                        curr_batch_size = len(y_batch)

                        predictions = sigmoid(np.dot(x_batch, w_local))
                        err = predictions - y_batch
                        gradient = np.dot(x_batch.T, err) / curr_batch_size
                        # aggiunta regolarizzazione L2
                        gradient[1:] += (lambda_reg / n_local) * w_local[1:]
                        w_local -= lr * (gradient)
                else:
                    predictions = sigmoid(np.dot(x_local, w_local))
                    err = predictions - y_local
                    gradient = np.dot(x_local.T, err) / n_local
                    
                    gradient[1:] += (lambda_reg / n_local) * w_local[1:]
                    w_local -= lr * gradient

            local_weights.append((w_local, n_local))
            total_samples += n_local
        
        # Aggregazione pesata
        if total_samples > 0:
            weighted_weights = np.zeros_like(global_w)
            for w, n in local_weights:
                weighted_weights += w * n
            global_w = weighted_weights / total_samples
        
        # valutazione sul server con validation set riservato
        metrics = evaluate_model(x_test_bias, y_test, global_w)
        #print(metrics)
        if r % 5 == 0:
            print(f"ROUND {r}: Accuracy {metrics['accuracy']:.4f}, Precision {metrics['precision']:.4f}, Recall {metrics['recall']:.4f}, F1-Score {metrics['f1']:.4f}, Loss {metrics['loss']:.4f}")
        metrics['round'] = r
        history.append(metrics)

    return global_w, history


def train_fed_prox(client_datasets, x_test, y_test, rounds=50, mu=0.1, local_epochs=5, lr=0.01, lambda_reg=0.01, batch_size=128):
    # mu: Parametro prossimale

    num_features = x_test.shape[1]
    num_features_b = num_features + 1 
    num_clients = len(client_datasets)
    x_test_bias = add_bias(x_test)
    history = []
    # aggiunta bias 
    client_datasets_bias = []
    for c in client_datasets:
        x_b = add_bias(c['x'])
        y = c['y']
        client_datasets_bias.append({'x': x_b, 'y': y})
        
    # inizializzazione pesi globali
    global_w = np.zeros(num_features_b)

    print(f"--- Start Federated Learning using FedProx algorithm ---")
    for r in range(rounds):
        local_weights = []
        total_samples = 0

        # Local Training
        for i in range(num_clients):
            x_local = client_datasets_bias[i]['x']
            y_local = client_datasets_bias[i]['y']
            n_local = len(y_local)

            if n_local == 0: continue # Salta client vuoti

            # riceve dal server il modello corrente
            w_local = np.copy(global_w)
            indices = np.arange(n_local)

            # training locale con SGD
            for j in range(local_epochs):

                if batch_size and batch_size < n_local:
                    np.random.shuffle(indices)                
                    for start_idx in range(0, n_local, batch_size):
                        batch_idx = indices[start_idx : start_idx + batch_size]
                        x_batch = x_local[batch_idx]
                        y_batch = y_local[batch_idx]
                        curr_batch_size = len(y_batch)
                        predictions = sigmoid(np.dot(x_batch, w_local))
                        err = predictions - y_batch
                        # gradiente Loss
                        grad_loss = np.dot(x_batch.T, err) / curr_batch_size
                        # gradiente L2
                        grad_l2 = (lambda_reg / n_local) * w_local
                        grad_l2[0] = 0 
                        # gradiente Proximal Term
                        grad_prox = mu * (w_local - global_w)
                        total_grad = grad_loss + grad_l2 + grad_prox
                        # update
                        w_local -= lr * total_grad                        
                else:
                    predictions = sigmoid(np.dot(x_local, w_local))
                    err = predictions - y_local
                    grad_loss = np.dot(x_local.T, err) / n_local
                    grad_l2 = (lambda_reg / n_local) * w_local
                    grad_l2[0] = 0
                    grad_prox = mu * (w_local - global_w)
        
                    total_grad = grad_loss + grad_l2 + grad_prox
                
                    w_local -= lr * total_grad
            
            local_weights.append((w_local, n_local))
            total_samples += n_local

        # Aggregazione pesata
        if total_samples > 0:
            weighted_weights = np.zeros_like(global_w)
            for w, n in local_weights:
                weighted_weights += w * n
            global_w = weighted_weights / total_samples

        # valutazione sul server con validation set riservato
        metrics = evaluate_model(x_test_bias, y_test, global_w)
        #print(metrics)
        if r % 5 == 0:
            print(f"ROUND {r}: Accuracy {metrics['accuracy']:.4f}, Precision {metrics['precision']:.4f}, Recall {metrics['recall']:.4f}, F1-Score {metrics['f1']:.4f}, Loss {metrics['loss']:.4f}")
        metrics['round'] = r
        history.append(metrics)

    return global_w, history

def train_fed_admm(client_datasets, x_test, y_test, rounds=50, rho=1.0, local_epochs=5, lr=0.1, lambda_l2=0.01, lambda_l1=0.01, batch_size=128):
    num_features = x_test.shape[1]
    num_features_b = num_features + 1 
    num_clients = len(client_datasets)
    x_test_bias = add_bias(x_test)
    history = []

    # aggiunta bias ai dati client
    client_datasets_bias = []
    n_client_samples = []
    for c in client_datasets:
        x_b = add_bias(c['x'])
        y = c['y']
        client_datasets_bias.append({'x': x_b, 'y': y})
        n_client_samples.append(len(y))
    
    # Calcolo dei pesi per ogni client in base alla quantità di dati
    total_samples = np.sum(n_client_samples)
    client_weights = n_client_samples / total_samples
    
    # Variabili ADMM
    z = np.zeros(num_features_b)  # Variabile Globale
    u = np.zeros((num_clients, num_features_b)) # Variabili Duali (una per client)
    w = np.zeros((num_clients, num_features_b)) # Variabili Primali Locali
    

    print(f"--- Start Federated Learning using FedADMM algorithm ---")
    for r in range(rounds):
        total_samples = 0
        # STEP 1 - Primal Update locale di w
        for i in range(num_clients):
            x_local = client_datasets_bias[i]['x']
            y_local = client_datasets_bias[i]['y']
            n_local = len(y_local)
            
            if n_local == 0: continue

            indices = np.arange(n_local)
            w_local = np.copy(w[i])
            
            # training locale con SGD
            for j in range(local_epochs):

                if batch_size and batch_size < n_local:
                    np.random.shuffle(indices)
                    
                    for start_idx in range(0, n_local, batch_size):
                        batch_idx = indices[start_idx : start_idx + batch_size]
                        x_batch = x_local[batch_idx]
                        y_batch = y_local[batch_idx]
                        curr_batch_size = len(y_batch)

                        predictions = sigmoid(np.dot(x_batch, w_local))
                        err = predictions - y_batch
                        # Gradiente Loss
                        grad_loss = np.dot(x_batch.T, err) / curr_batch_size
                        # Gradiente L2 scalato
                        grad_l2 = (lambda_l2 / n_local) * w_local
                        grad_l2[0] = 0 
                        # Gradiente del termine della lagrangiana aumentata
                        grad_penalty = rho * (w_local - z + u[i])
                        total_grad = grad_loss + grad_l2 + grad_penalty
                        total_grad = np.clip(total_grad, -5.0, 5.0)
                        # Update                        
                        w_local -= lr * total_grad
                else:
                    predictions = sigmoid(np.dot(x_local, w_local))
                    err = predictions - y_local
                    grad_loss = np.dot(x_local.T, err) / n_local
                    grad_l2 = (lambda_l2 / n_local) * w_local
                    grad_l2[0] = 0
                    grad_penalty = rho * (w_local - z + u[i])
                    total_grad = grad_loss + grad_l2 + grad_penalty
                    total_grad = np.clip(total_grad, -5.0, 5.0)
                    # Update
                    w_local -= lr * total_grad
            
            w[i] = w_local
            
        # STEP 2 - Global Update di z
        # media non pesata
        z_avg = np.mean(w + u, axis=0)
        #media pesata
        #local_update = w + u
        #z_avg = np.average(local_update, axis=0, weights=client_weights)

        # soft-thresholding per regolarizzazione L1 globale
        k = lambda_l1 / (rho * num_clients)
        z = soft_thresholding(z_avg, k)

        # STEP 3 - Dual Update locale di u
        for i in range(num_clients):
            u[i] = u[i] + (w[i] - z)
            
        # Valutazione sul server, con validation set riservato
        metrics = evaluate_model(x_test_bias, y_test, z)
        #print(metrics)
        if r % 5 == 0:
            print(f"ROUND {r}: Accuracy {metrics['accuracy']:.4f}, Precision {metrics['precision']:.4f}, Recall {metrics['recall']:.4f}, F1-Score {metrics['f1']:.4f}, Loss {metrics['loss']:.4f}")
        metrics['round'] = r
        history.append(metrics)

    return z, history