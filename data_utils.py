import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

    # Indici delle classi
    index_clean = np.where(y_clients == 0)[0]
    index_spam = np.where(y_clients == 1)[0]
    
    # Determina la distribuzione per le classi tra i client
    dist_clean = rng.dirichlet(np.repeat(alpha, num_clients))
    dist_spam = rng.dirichlet(np.repeat(alpha, num_clients))
    
    # Calcola quanti campioni dare a ciascun client per classe
    num_clean_samples = (dist_clean * len(index_clean)).astype(int)
    num_spam_samples = (dist_spam * len(index_spam)).astype(int)
    
    # Correzione arrotondamenti
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

def visualize_data_distribution(stats):
    if not stats: return
    clients = [s["client"] for s in stats]
    cleans = [s["clean"] for s in stats]
    spams = [s["spam"] for s in stats]
    totals = [s["total"] for s in stats]
    
    x = np.arange(len(clients))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 5))
    rects1 = ax.bar(x - width/2, cleans, width, label='Clean', color='skyblue')
    rects2 = ax.bar(x + width/2, spams, width, label='Spam', color='orange')
    
    for i, v in enumerate(totals):
        ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
    
    ax.set_ylabel('Quantità Dati')
    ax.set_title('Distribuzione per classe tra i client')
    ax.set_xticks(x)
    ax.set_xticklabels(clients)
    ax.legend()
    
    plt.tight_layout()
    plt.show()