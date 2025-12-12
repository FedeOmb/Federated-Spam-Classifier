import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from federatedLR import train_fedavg, train_fed_prox, train_fed_admm, add_bias

def load_dataset():
    df = pd.read_csv("./dataset/SMS-spam-dataset-uci.csv", encoding="latin_1")
    print(df)
    print(df.shape)

    df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True)
    df.rename(columns={"v1": "label", "v2": "message"}, inplace=True)
    print(df.head())

    df['spam'] = df.label.map({'ham': 0, 'spam': 1})
    print(df.head())
    
    return df, df['message'], df['spam']

def split_preprocess_data_iid(x_clients, y_clients, vectorizer, num_clients):
    client_datasets = []
    stats = []

    indices = np.random.permutation(len(x_clients))
    chunk_size = len(x_clients) // num_clients

    for i in range(num_clients):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        idx = indices[start:end]
        # Dati grezzi del client
        client_text = x_clients[idx]
        client_label = y_clients[idx]
        
        if len(client_text) == 0:
            client_text_vectorized = np.empty((0, len(vectorizer.get_feature_names_out())))
        else:
            client_text_vectorized = vectorizer.transform(client_text).toarray()
        
        client_datasets.append({
            'x': client_text_vectorized,
            'y': client_label
        })

        # statistiche dati
        index_clean = np.where(client_label == 0)
        index_spam = np.where(client_label == 1)
        stats.append({
            "client": f"C{i}",
            "clean": len(index_clean),
            "spam": len(index_spam),
            "total": len(client_text)
        })
    return client_datasets, stats

def split_preprocess_data_non_iid(x_clients, y_clients, vectorizer, num_clients, alpha=0.5):
    client_datasets = []
    
    rng = np.random.default_rng(100)

    index_clean = np.where(y_clients == 0)[0]
    index_spam = np.where(y_clients == 1)[0]
    
    # Determina la distribuzione per le classi tra i client
    dist_clean = rng.dirichlet(np.repeat(alpha, num_clients))
    dist_spam = rng.dirichlet(np.repeat(alpha, num_clients))
    
    # Calcola quanti campioni dare a ciascun client per classe
    num_clean_samples = (dist_clean * len(index_clean)).astype(int)
    num_spam_samples = (dist_spam * len(index_spam)).astype(int)
    
    num_clean_samples[-1] += len(index_clean) - np.sum(num_clean_samples)
    num_spam_samples[-1] += len(index_spam) - np.sum(num_spam_samples)
    
    ptr_clean = 0
    ptr_spam = 0
    
    stats = [] # Per visualizzazione

    for i in range(num_clients):
        # Estrai gli indici per questo client
        curr_index_clean = index_clean[ptr_clean : ptr_clean + num_clean_samples[i]]
        curr_index_spam = index_spam[ptr_spam : ptr_spam + num_spam_samples[i]]
        
        # Aggiorna puntatori
        ptr_clean += num_clean_samples[i]
        ptr_spam += num_spam_samples[i]
        
        # Unisci e mescola localmente
        client_indices = np.concatenate([curr_index_clean, curr_index_spam])
        rng.shuffle(client_indices)
        
        # Dati grezzi del client
        client_text = x_clients[client_indices]
        client_label = y_clients[client_indices]
        
        if len(client_text) == 0:
            client_text_vectorized = np.empty((0, len(vectorizer.get_feature_names_out())))
        else:
            client_text_vectorized = vectorizer.transform(client_text).toarray()

        client_datasets.append({
            'x': client_text_vectorized,
            'y': client_label
        })
        
        # statistiche dati
        stats.append({
            "client": f"C{i}",
            "clean": len(curr_index_clean),
            "spam": len(curr_index_spam),
            "total": len(client_text)
        })

    return client_datasets, stats

def extract_best_configs(results_df, metric='f1', average_last_n=5):
    summary_data = []

    grouped = results_df.groupby(['algorithm', 'params_str'])
    
    for (algo, params_str), group in grouped:
        group = group.sort_values('round')
        last_n_rounds = group.tail(average_last_n)
        
        avg_metrics = {
            'algorithm': algo,
            'params_raw': params_str,
            'final_f1': last_n_rounds['f1'].mean(),
            'final_acc': last_n_rounds['accuracy'].mean(),
            'final_prec': last_n_rounds['precision'].mean(),
            'final_rec': last_n_rounds['recall'].mean(),
            'final_loss': last_n_rounds['loss'].mean(),
            'std_f1': last_n_rounds['f1'].std()
        }
        
        params_dict = ast.literal_eval(params_str)
        avg_metrics.update(params_dict)

        summary_data.append(avg_metrics)
        
    summary_df = pd.DataFrame(summary_data)

    ascending = True if metric == 'loss' else False
    target_col = f"final_{metric}" if f"final_{metric}" in summary_df.columns else metric
    
    summary_df = summary_df.sort_values(by=target_col, ascending=ascending).reset_index(drop=True)
    
    cols = ['algorithm', 'final_f1', 'final_acc', 'final_loss', 'final_prec', 'final_rec']
    param_cols = [c for c in summary_df.columns if c not in cols]
    final_cols = cols + param_cols
    
    return summary_df[final_cols]

def run_grid_search(algo, configs, data, x_valid, y_valid):
    all_histories = []
    print(f"Iniziando Grid Search su {len(configs)} configurazioni...")

    for idx, conf in enumerate(configs):
        params = {k:v for k,v in conf.items() }
        
        print(f"[{idx+1}/{len(configs)}] {algo} -> {params}")

        if algo == 'fedavg':
            weights, history = train_fedavg(data, x_valid, y_valid, rounds=100, batch_size=64, **params)
        elif algo == 'fedprox':
            weights, history = train_fed_prox(data, x_valid, y_valid, rounds=200, batch_size=64, lambda_reg=0.001, **params)
        elif algo == 'fedadmm':
            weights, history = train_fed_admm(data, x_valid, y_valid, rounds=100, batch_size=64, lambda_l2=0.001, **params)
        else:
            print("Algoritmo non riconosciuto")
            continue

        for entry in history:
            entry['algorithm'] = algo
            entry['params_str'] = str(params)
        
        all_histories.extend(history)

    results_df = pd.DataFrame(all_histories)
    return results_df

def run_alpha_comparison(X_train_fed, y_train_fed, X_test_vec, y_test, vectorizer, best_configs):
    
    alphas = [100.0, 1.0, 0.5, 0.1]
    results_history = []
    
    print("Test robustezza algoritmi sui diverse distribuzioni dei dati (IID vs Non-IID)")
    
    for alpha in alphas:
        # Rigenera i dati distribuiti con il nuovo alpha
        client_datasets, stats = split_preprocess_data_non_iid(
            X_train_fed, y_train_fed, vectorizer, num_clients=10, alpha=alpha
        )
        #visualize_data_distribution(stats)
        
        for algo, params in best_configs.items():
            print(f"Training {algo} with alpha={alpha} and params={params}")
            p = params.copy()
            
            if algo == 'fedavg':
                w, hist = train_fedavg(client_datasets, X_test_vec, y_test, **p)
            elif algo == 'fedprox':
                w, hist = train_fed_prox(client_datasets, X_test_vec, y_test, **p)
            elif algo == 'fedadmm':
                w, hist = train_fed_admm(client_datasets, X_test_vec, y_test, **p)
            
            # Arricchisci la history per il plotting
            for entry in hist:
                entry['algorithm'] = algo
                entry['alpha'] = alpha
            
            results_history.extend(hist)
            
    return pd.DataFrame(results_history)