import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

df = pd.read_csv("titanic.csv")
print("Initial Data Shape:", df.shape)
print("First 5 rows of the dataset:")
print(df.head())

print(df.isnull().sum())
print(df.isnull().sum().sum())

df["HasCabin"] = df["Cabin"].notnull().astype(int)

print(df.groupby("HasCabin")["Survived"].mean())

df = df.drop(columns = ["Cabin"])


print(df.groupby(["Pclass", "Sex"])["Age"].median())

print(df["Age"].isnull().sum())

df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(lambda x: x.fillna(x.median()))


df["Age"] = df["Age"].fillna(df["Age"].median())

print(df["Age"].isnull().sum())



fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df["Age"], bins=30, color="#3498db", edgecolor="black")
ax.set_title("Age Distribution After Imputation")
ax.set_xlabel("Age")
ax.set_ylabel("count")
plt.tight_layout()
plt.show()

print(df["Embarked"].isnull().sum())

print(df["Embarked"].mode()[0])

df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# let see the null values again

print(df.isnull().sum().sum())

print(df.isnull().sum())

df["Title"] = df["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())
print(df["Title"].value_counts())

title_mapping = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
}

df["Title"] = df["Title"].apply(lambda x: title_mapping.get(x, "Rare"))

print(df["Title"].value_counts())

print(df.groupby("Title")["Survived"].mean().sort_values(ascending=False))

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

print(df["IsAlone"].sum())
print(f"Proportion of passengers who were alone: {df['IsAlone'].mean():.1%}")
print(f"Survival rate for passengers who were alone: {df.groupby('IsAlone')['Survived'].mean().iloc[1]:.1%}")
print(f"Survival rate for passengers who were alone: {df.groupby('IsAlone')['Survived'].mean().iloc[0]:.1%}")


df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 12, 18, 35, 60, 80], labels=["Child", "Teen", "Adult", "Senior", "Elderly"])

print(df["AgeGroup"].value_counts().sort_index())

print(df.groupby("AgeGroup")["Survived"].mean().sort_index())

df["FareBin"] = pd.qcut(df["Fare"], q=4, labels=["Low", "Medium", "High", "Very High"])
print(df["FareBin"].value_counts().sort_index())
print(df.groupby("FareBin")["Survived"].mean().sort_index())

## Encoding categorical variables

print(df.dtypes)
print(df.shape)

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# One-hot encode Embarked, Title, AgeGroup, FareBin
df = pd.get_dummies(df, columns=["Embarked", "Title", "AgeGroup", "FareBin"], drop_first=True, dtype=int)

print(df.shape)
print(df.head())