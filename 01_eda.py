import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 100


df = pd.read_csv("titanic.csv")

missing = pd.DataFrame({
    "Total Missing": df.isnull().sum(),
    "Percentage Missing": (df.isnull().sum() / len(df)) * 100,
})
missing = missing[missing["Total Missing"] > 0].sort_values("Total Missing", ascending=False)

fig, ax = plt.subplots(figsize=(10, 4))
colors = ["#e74c3c" if df[col].isnull().any() else "#2ecc71" for col in df.columns]
ax.barh(df.columns, df.isnull().sum(), color=colors)
ax.set_xlabel("Number of Missing Values")
ax.set_title("Missing Values in Titanic Dataset")
for i, v in enumerate(df.isnull().sum()):
    if v > 0:
        ax.text(v + 0.5, i, str(v), fontweight="bold", va="center")
plt.tight_layout()
plt.show()