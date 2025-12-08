import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from data_utils import load_dataset, split_preprocess_data_non_iid, split_preprocess_data_iid, visualize_data_distribution
from federatedLR_v2 import train_fedavg, train_fed_prox, train_fed_admm, add_bias
from centralizedLR import train_centralized
NUM_CLIENTS = 10

df_full, x, y = load_dataset()

# Separazione Dati server e client
x_server_raw, x_clients_raw, y_server, y_clients = train_test_split(
    df_full['message'], df_full['spam'], test_size=0.90, random_state=42, stratify=df_full['spam']
)

print(f"Dati Pubblici sul Server: {len(x_server_raw)}")
print(f"Dati Privati sui Client: {len(x_clients_raw)}")


X_clients_vals = x_clients_raw.values
print("X_clients_vals", type(X_clients_vals))
y_clients_vals = y_clients.values

# suddivisione dati privati in training set federato, validation e test set globali
x_train_fed, x_val_test, y_train_fed, y_val_test = train_test_split(
    X_clients_vals, y_clients_vals, test_size=0.2, random_state=42
)

x_valid_fed, x_test_fed, y_valid_fed, y_test_fed = train_test_split(
    x_val_test, y_val_test, test_size=0.5, random_state=42
)

print("X training federato", x_train_fed.shape)
print("X Validation globale", x_valid_fed.shape)
print("X Test globale", x_test_fed.shape)

# Vettorizzazione solo sui dati pubblici del server
server_vectorizer = TfidfVectorizer(max_features=2**12, stop_words='english')
server_vectorizer.fit(x_server_raw) 
print("Vocabolario costruito dal server")
#print("features vectorized", vectorizer.get_feature_names_out())
print(f"Dimensione del vocabolario: {len(server_vectorizer.get_feature_names_out())}")

hash_vectorizer = HashingVectorizer(
    n_features=2**12, 
    alternate_sign=False,
    norm='l2',              
    stop_words='english',
    dtype=np.float32              
)

#Vettorizzazione globale sul training set
full_train_vectorizer = TfidfVectorizer(max_features=2**12, stop_words='english')
full_train_vectorizer.fit(x_train_fed)
x_train_central = full_train_vectorizer.transform(x_train_fed).toarray()
x_valid_central = full_train_vectorizer.transform(x_valid_fed).toarray()
x_test_central = full_train_vectorizer.transform(x_test_fed)
y_train_central = y_train_fed
y_valid_central = y_valid_fed
y_test_central = y_test_fed
print(f"Dimensione del vocabolario: {len(full_train_vectorizer.get_feature_names_out())}")
print(f"Dimensione features: {x_test_central.shape[1]}")

train_centralized(x_train_central, y_train_fed, x_test_central, y_test_fed, epochs=1000, lambda_reg=0.001, learning_rate=3.0)

# Distribuzione ai Client e Preprocessing Locale
#client_datasets, stats = split_preprocess_data_iid(x_train_fed, y_train_fed, server_vectorizer, NUM_CLIENTS)
client_datasets, stats = split_preprocess_data_non_iid(x_train_fed, y_train_fed, server_vectorizer, NUM_CLIENTS, alpha=1.0)
print(len(client_datasets))
print(len(client_datasets[0]['x']))
print(len(client_datasets[0]['y']))
#print(client_datasets[0])

visualize_data_distribution(stats)
# vettorizzazione validation e test set globale
x_valid_vectorized = server_vectorizer.transform(x_valid_fed).toarray()
x_test_vectorized = server_vectorizer.transform(x_test_fed).toarray()

print(f"Dati distribuiti e vettorizzati su {NUM_CLIENTS} client.")
print(f"Dimensione features: {x_test_vectorized.shape[1]}")

configs_fedavg = [
    {'rounds': 200, 'local_epochs': 10, 'lr': 0.5, 'lambda_reg': 0.01},
    {'rounds': 200, 'local_epochs': 10, 'lr': 1.0, 'lambda_reg': 0.01},
    {'rounds': 200, 'local_epochs': 10, 'lr': 2.0, 'lambda_reg': 0.01},
    {'rounds': 200, 'local_epochs': 10, 'lr': 3.0, 'lambda_reg': 0.01},
    {'rounds': 200, 'local_epochs': 10, 'lr': 4.0, 'lambda_reg': 0.01},
]

configs_fedprox = [
    {'rounds': 200, 'local_epochs': 10, 'lr': 0.1, 'mu': 0.01},
    {'rounds': 200, 'local_epochs': 10, 'lr': 0.5, 'mu': 0.01},
    {'rounds': 200, 'local_epochs': 10, 'lr': 1.0, 'mu': 0.01},
    {'rounds': 200, 'local_epochs': 10, 'lr': 2.0, 'mu': 0.01},
    {'rounds': 200, 'local_epochs': 10, 'lr': 3.0, 'mu': 0.01},
    ]
configs_fedprox2 = [
    {'rounds': 200, 'local_epochs': 10, 'lr': 3.0, 'mu': 0.001},
    {'rounds': 200, 'local_epochs': 10, 'lr': 3.0, 'mu': 0.01},
    {'rounds': 200, 'local_epochs': 10, 'lr': 3.0, 'mu': 0.1},
    {'rounds': 200, 'local_epochs': 10, 'lr': 3.0, 'mu': 1.0},
    ]
# Esecuzione
#fedavg_w, fedavg_history = train_fedavg(client_datasets, x_test=x_valid_vectorized, y_test=y_valid, rounds=100, local_epochs=10, lr=3.0, lambda_reg=0.001)
#fedprox_w, fedprox_acc_history = train_fed_prox(client_datasets, x_test=x_test, y_test=y_test, rounds=100, mu=0.1, local_epochs=10, lr=0.1)
#admm_z, admm_acc_history = train_fed_admm(client_datasets, x_test=x_test, y_test=y_test, rounds=100, rho=0.05, local_epochs=20, lr=3.0, lambda_l2=0.0001, lambda_l1=0.001)


def run_grid_search(configs, data, x_valid, y_valid):
    all_histories = []
    print(f"Iniziando Grid Search su {len(configs)} configurazioni...")

    for idx, conf in enumerate(configs):
        algo = "fedavg"
        params = {k:v for k,v in conf.items() }
        
        print(f"[{idx+1}/{len(configs)}] {algo} -> {params}")
        
        weights, history = train_fed_prox(data, x_valid, y_valid, **params)

        for entry in history:
            entry['algorithm'] = algo
            entry['params_str'] = str(params)
        
        all_histories.extend(history)

    results_df = pd.DataFrame(all_histories)
    return results_df


results_df = run_grid_search(configs_fedprox2, client_datasets, x_valid_vectorized, y_valid)
print(results_df.head())

plt.figure(figsize=(14, 6))

# Plot Validation (linea solida)
sns.lineplot(
    data=results_df, x='round', y='f1', 
    hue='params_str', linewidth=2, alpha=0.8
)
plt.xlabel('Round')
plt.ylabel('f1')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))

sns.lineplot(
    data=results_df, x='round', y='recall', 
    hue='params_str', 
    linewidth=2, alpha=0.8
)
plt.xlabel('Round')
plt.ylabel('recall')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))

sns.lineplot(
    data=results_df, x='round', y='precision', 
    hue='params_str', 
    linewidth=2, alpha=0.8
)
plt.xlabel('Round')
plt.ylabel('precision')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()