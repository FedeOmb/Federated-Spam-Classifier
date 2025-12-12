import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from data_utils import load_dataset, split_preprocess_data_non_iid, split_preprocess_data_iid, extract_best_configs, run_grid_search, run_alpha_comparison
from plot_utils import visualize_data_distribution, plot_metrics_single_run, plot_metrics_grid_search, plot_alpha_comparison
from federatedLR import train_fedavg, train_fed_prox, train_fed_admm, add_bias
from centralizedLR import train_centralized_sklearn_sgd, grid_search_centralsgd, test_centralsgd
import configs_grid_search as conf

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

# Distribuzione ai Client e Preprocessing Locale
#client_datasets, stats = split_preprocess_data_iid(x_train_fed, y_train_fed, server_vectorizer, NUM_CLIENTS)
client_datasets, stats = split_preprocess_data_non_iid(x_train_fed, y_train_fed, server_vectorizer, NUM_CLIENTS, alpha=0.5)
print(len(client_datasets))
print(len(client_datasets[0]['x']))
print(len(client_datasets[0]['y']))
visualize_data_distribution(stats)

# vettorizzazione validation e test set globale
x_valid_vectorized = server_vectorizer.transform(x_valid_fed).toarray()
x_test_vectorized = server_vectorizer.transform(x_test_fed).toarray()
x_train_full_vec = server_vectorizer.transform(x_train_fed).toarray()

print(f"Dati distribuiti e vettorizzati su {NUM_CLIENTS} client.")
print(f"Dimensione features: {x_test_vectorized.shape[1]}")

# --TRAINING CENTRALIZZATO--
train_centralized_sklearn_sgd(x_train_full_vec, y_train_fed, x_valid_vectorized, y_valid_fed, epochs=500, lambda_reg=0.0001, lr=1.0)
best_model = grid_search_centralsgd(x_train_full_vec, y_train_fed, x_valid_vectorized, y_valid_fed)
test_results = test_centralsgd(best_model, x_test_vectorized, y_test_fed)

pd.set_option('display.max_columns', None)

# -- TRAINING FEDERATO --
results_df = run_grid_search("fedavg",conf.configs_fedavg5, client_datasets, x_valid_vectorized, y_valid_fed)
#results_df = run_grid_search("fedprox",conf.configs_fedprox3, client_datasets, x_valid_vectorized, y_valid_fed)
#results_df = run_grid_search("fedadmm",conf.configs_fedadmm2, client_datasets, x_valid_vectorized, y_valid_fed)

print(results_df.head())

plot_metrics_grid_search(results_df)

best_configs_df = extract_best_configs(results_df, metric='f1', average_last_n=5)
print("Best Configurations:")
print(best_configs_df)

# -- TEST ROBUSTEZZZA ALGORITMI --
test_alpha_results = run_alpha_comparison(x_train_fed, y_train_fed, x_test_vectorized, y_test_fed, server_vectorizer, conf.best_params_valid)
plot_alpha_comparison(test_alpha_results)

# -- TUNING FEDPROX SU DATI NON-IID -- 

#client_datasets_noniid, stats = split_preprocess_data_non_iid(x_train_fed, y_train_fed, server_vectorizer, NUM_CLIENTS, alpha=0.1)
#visualize_data_distribution(stats)
#results_df = run_grid_search("fedprox",conf.configs_fedprox4, client_datasets_noniid, x_valid_vectorized, y_valid_fed)
#plot_metrics_grid_search(results_df)
#fedprox_w, fedprox_history = train_fed_prox(client_datasets_noniid, x_test=x_test_vectorized, y_test=y_test_fed, rounds=200, mu=0.01, local_epochs=20, lr=2.0, lambda_reg=0.001)
#plot_metrics_single_run(pd.DataFrame(fedprox_history))
