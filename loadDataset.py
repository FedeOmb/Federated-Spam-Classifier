import pandas as pd

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