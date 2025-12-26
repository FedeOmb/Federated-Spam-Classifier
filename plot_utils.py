import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

def plot_metrics_single_run(history):
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=history, x='round', y='f1', linewidth=2, alpha=0.8)
    plt.title('F1')
    plt.xlabel('Round')
    plt.ylabel('F1')
    plt.grid(True, alpha=0.3)
    plt.show(block=False)
    
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=history, x='round', y='recall', linewidth=2, alpha=0.8)
    plt.title('Recall')
    plt.xlabel('Round')
    plt.ylabel('Recall')
    plt.grid(True, alpha=0.3)
    plt.show(block=False)

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=history, x='round', y='precision', linewidth=2, alpha=0.8)
    plt.title('Precision')
    plt.xlabel('Round')
    plt.ylabel('Precision')
    plt.grid(True, alpha=0.3)
    plt.show(block=False)

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=history, x='round', y='loss', linewidth=2, alpha=0.8)
    plt.title('Loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.show(block=False)

    plt.show()

def plot_metrics_grid_search(results_df):
    palette = sns.color_palette("tab20", n_colors=results_df['params_str'].nunique())

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=results_df, x='round', y='f1', hue='params_str', linewidth=2, alpha=0.8, palette=palette)
    plt.title('F1')
    plt.xlabel('Round')
    plt.ylabel('F1')
    plt.grid(True, alpha=0.3)
    plt.legend(title="parametri", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.show(block=False)

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=results_df, x='round', y='recall', hue='params_str', linewidth=2, alpha=0.8, palette=palette)
    plt.title('Recall')
    plt.xlabel('Round')
    plt.ylabel('Recall')
    plt.grid(True, alpha=0.3)
    plt.legend(title="parametri", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.82, 1])

    plt.show(block=False)

    plt.figure(figsize=(108, 5))
    sns.lineplot(data=results_df, x='round', y='precision', hue='params_str', linewidth=2, alpha=0.8, palette=palette)
    plt.title('Precision')
    plt.xlabel('Round')
    plt.ylabel('Precision')
    plt.grid(True, alpha=0.3)
    plt.legend(title="parametri", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.show(block=False)

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=results_df, x='round', y='loss', hue='params_str', linewidth=2, alpha=0.8, palette=palette)
    plt.title('Validation Loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend(title="parametri", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.show(block=False)

    plt.show()

def plot_alpha_comparison(df_results):
    sns.set_style("whitegrid")
    
    alphas = sorted(df_results['alpha'].unique(), reverse=True)
    
    for a in alphas:
        plt.figure(figsize=(10, 6))
        data = df_results[df_results['alpha'] == a]

        sns.lineplot(data=data, x='round', y='f1', hue='algorithm', linewidth=2.5)
        
        plt.title(f"Robustness Analysis alpha = {a}", fontsize=14, fontweight='bold')
        plt.xlabel("Rounds")
        plt.ylabel("Test F1-Score")
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower right')
        plt.tight_layout()

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=data, x='round', y='loss', hue='algorithm', linewidth=2.5)

        plt.title(f"Robustness Analysis alpha = {a}", fontsize=14, fontweight='bold')
        plt.xlabel("Rounds")
        plt.ylabel("Test Loss")
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        plt.tight_layout()

        plt.show()