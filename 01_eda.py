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

Numeric_cols = ["Age", "Fare", "SibSp", "Parch"]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for i, col in enumerate(Numeric_cols):
    axes[i].hist(df[col].dropna(), bins=30, color="#3498db", edgecolor="black")
    axes[i].axvline(df[col].mean(), color="red", linestyle="dashed", linewidth=1)
    axes[i].axvline(df[col].median(), color="green", linestyle="dashed", linewidth=1)
    axes[i].set_title(f"{col} Distribution")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Frequency")
    axes[i].legend(["Mean", "Median"])

plt.suptitle("Numeric Feature Distributions", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 4, figsize=(16, 5 ))

for i, col in enumerate(Numeric_cols):
    sns.boxplot(y=df[col], ax=axes[i], color="#e74c3c")
    axes[i].set_title(f"{col} Box Plot")

plt.suptitle("Numeric Feature Box Plots", fontsize=14, fontweight="bold")
plt.tight_layout() 
plt.show()

catergorical_cols = ["Pclass", "Sex", "Embarked"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, col in enumerate(catergorical_cols):
    counts = df[col].value_counts()
    axes[i].bar(counts.index.astype(str), counts.values, color="#3498db")
    axes[i].set_title(f"{col} Distribution")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")
    for j, v in enumerate(counts.values):
        axes[i].text(j, v + 1, str(v), fontweight="bold", ha="center")
plt.suptitle("Categorical Feature Distributions", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

survival_by_sex = df.groupby("Sex")["Survived"].mean() * 100
axes[0].bar(survival_by_sex.index, survival_by_sex.values, color=["#e74c3c", "#3498db"])
axes[0].set_title("Survival Rate by Sex")
axes[0].set_ylabel("Survival Rate (%)")
for i, v in enumerate(survival_by_sex.values):
    axes[0].text(i, v + 1, f"{v:.1f}%", fontweight="bold", ha="center")
survival_by_pclass = df.groupby("Pclass")["Survived"].mean() * 100
axes[1].bar(survival_by_pclass.index.astype(str), survival_by_pclass.values, color="#3498db")
axes[1].set_title("Survival Rate by Pclass")
axes[1].set_ylabel("Survival Rate (%)")
for i, v in enumerate(survival_by_pclass.values):
    axes[1].text(i, v + 1, f"{v:.1f}%", fontweight="bold", ha="center")
survival_by_embarked = df.groupby("Embarked")["Survived"].mean() * 100
axes[2].bar(survival_by_embarked.index.astype(str), survival_by_embarked.values, color="#e74c3c")
axes[2].set_title("Survival Rate by Embarked")  
axes[2].set_ylabel("Survival Rate (%)")
for i, v in enumerate(survival_by_embarked.values):
    axes[2].text(i, v + 1, f"{v:.1f}%", fontweight="bold", ha="center")
plt.suptitle("Survival Rates by Categorical Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
survival_grouped = df.groupby(["Pclass", "Sex"])["Survived"].mean().unstack()
survival_grouped.plot(kind="bar", ax=ax, color=["#e74c3c", "#3498db"])
ax.set_title("Survival Rate by Pclass and Sex")
ax.set_xlabel("Pclass")
ax.set_xticklabels(["1st Class", "2nd Class", "3rd Class"], rotation=0)
ax.set_ylabel("Survival Rate")
ax.legend(title="Sex")
for i in range(survival_grouped.shape[0]):
    for j in range(survival_grouped.shape[1]):
        v = survival_grouped.iloc[i, j] * 100
        ax.text(i + j * 0.25 - 0.125, v + 1, f"{v:.1f}%", fontweight="bold", ha="center")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].hist(df[df["Survived"] == 1]["Age"].dropna(), bins=30, color="#e74c3c", edgecolor="black", alpha=0.7, label="Survived")
axes[0].hist(df[df["Survived"] == 0]["Age"].dropna(), bins=30, color="#3498db", edgecolor="black", alpha=0.7, label="Not Survived")
axes[0].set_title("Age Distribution by Survival")
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Frequency")
axes[0].legend()

df[df["Survived"] == 1]["Age"].dropna().plot(kind="kde", ax=axes[1], color="#e74c3c", label="Survived")
df[df["Survived"] == 0]["Age"].dropna().plot(kind="kde", ax=axes[1], color="#3498db", label="Not Survived")
axes[1].set_title("Age Density by Survival")
axes[1].set_xlabel("Age")
axes[1].legend()
plt.suptitle("Age Distribution and Density by Survival", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x="Survived", y="Fare", data=df, ax=ax, palette=["#3498db", "#e74c3c"])
ax.set_title("Fare Distribution by Survival")
ax.set_xlabel("Survived")
ax.set_xticklabels(["Not Survived", "Survived"])
ax.set_ylabel("Fare")
plt.tight_layout()
plt.show()

print(f"\nAverage Fare for Survivors: ${df[df['Survived'] == 1]['Fare'].mean():.2f}")
print(f"Average Fare for Non-Survivors: ${df[df['Survived'] == 0]['Fare'].mean():.2f}")

#Family size analysis
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

family_counts = df["FamilySize"].value_counts().sort_index()
axes[0].bar(family_counts.index, family_counts.values, color="#3498db")
axes[0].set_title("Family Size Distribution")
axes[0].set_xlabel("Family Size")
axes[0].set_ylabel("Count")

survival_by_family = df.groupby("FamilySize")["Survived"].mean() * 100
axes[1].bar(survival_by_family.index, survival_by_family.values, color="#e74c3c")
axes[1].set_title("Survival Rate by Family Size")
axes[1].set_xlabel("Family Size")
axes[1].set_ylabel("Survival Rate (%)")
axes[1].axhline(y=df["Survived"].mean() * 100, color="green", linestyle="dashed", linewidth=1, label="Overall Survival Rate")
axes[1].legend()
plt.suptitle("Family Size Analysis", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

numeric_df = df[["Age", "Fare", "SibSp", "Parch", "FamilySize", "Pclass", "Survived"]].copy()
numeric_df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
correlation_matrix = numeric_df.corr()
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Correlation Matrix of Numeric Features")
plt.tight_layout()
plt.show()

target_corr = correlation_matrix["Survived"].drop("Survived").sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#e74c3c" if val > 0 else "#3498db" for val in target_corr]
ax.barh(target_corr.index, target_corr.values, color=colors)
ax.set_title("Correlation of Features with Survival")
ax.set_xlabel("Correlation Coefficient")
ax.axvline(0, color="black", linestyle="dashed", linewidth=1)
plt.tight_layout()
plt.show()

df["Title"] = df["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
title_counts = df["Title"].value_counts()
axes[0].bar(title_counts.index, title_counts.values, color="#3498db")
axes[0].set_title("Title Distribution")
axes[0].set_xlabel("Title")

survival_by_title = df.groupby("Title")["Survived"].mean().sort_values()
Color = ["#e74c3c" if val > 0 else "#3498db" for val in survival_by_title]
axes[1].barh(survival_by_title.index, survival_by_title.values, color=Color)
axes[1].set_title("Survival Rate by Title")
axes[1].set_xlabel("Survival Rate")
axes[1].axvline(x=0.5, color="black", linestyle="dashed", linewidth=1)
plt.suptitle("Title Analysis", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()