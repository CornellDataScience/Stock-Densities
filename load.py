import pandas as pd
def load_data():
    df = pd.read_csv("E-mini S&P 500 minute data (approx 10 days).csv", header=1)
    cols = ["Date Time", "Close"]
    data = df[cols]
    data = data.set_axis(["timestamp", "close"], axis=1)
    return data

load_data()

