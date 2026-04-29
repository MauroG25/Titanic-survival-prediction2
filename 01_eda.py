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

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#count plot

survived_counts = df["Survived"].value_counts()
axes[0].bar(["Did Not Survive", "Survived"], survived_counts.values, color=["#3498db", "#e74c3c"])
axes[0].set_title("Survival Count")
axes[0].set_ylabel("Count")
for i, v in enumerate(survived_counts.values):
    axes[0].text(i, v + 1, str(v), fontweight="bold", ha="center")

axes[1].pie(survived_counts.values, labels=["Did Not Survive", "Survived"], autopct="%1.1f%%", colors=["#3498db", "#e74c3c"], startangle=90, explode=(0.01, 0.01))
axes[1].set_title("Survival Distribution")
plt.suptitle("Titanic Dataset", fontsize=14, fontweight="bold")
plt.tight_layout()  
plt.show()

print(f"\nSurvival Rate: {df['Survived'].mean() * 100:.2f}%")
print(f"Death Rate: {(1 - df['Survived'].mean()) * 100:.2f}%")