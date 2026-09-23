import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
df =  pd.read_csv("comprehensive_mutual_funds_data.csv")
print(df.shape)
print(df.columns)
print(df.info())
print(df.isnull().sum ())
print(df.describe())
print(df.isnull().sum())
print(df.isnull().mean()*100)
print(df["returns_3yr"].median())
print(df["returns_5yr"].median())
df["returns_3yr"] = df["returns_3yr"].fillna(df["returns_3yr"].median())
df["returns_5yr"] = df["returns_5yr"].fillna(df["returns_5yr"].median())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df["category"].value_counts())
print(df.mode(numeric_only=True).iloc[0])
print(df["risk_level"].value_counts())
print(df["fund_age_yr"].describe())
print(df["returns_1yr"].mean())
print(df["returns_1yr"].median())
print(df["returns_1yr"].mode().iloc[0])
print(df["returns_1yr"].min())
print(df["returns_1yr"].max())
print(df["returns_1yr"].std())
df["returns_1yr"].hist()
plt.show()
df["risk_level"].value_counts().plot(kind="bar")
plt.show()
df.groupby("risk_level")["returns_1yr"].mean().plot(kind="bar")
plt.show()
df.groupby("category")["returns_1yr"].mean() 
print(df[["returns_1yr","returns_3yr","returns_5yr"]].mean())
print(df.groupby("category")[["returns_1yr","returns_3yr","returns_5yr"]].mean())
scaler = MinMaxScaler()

cols = ["returns_1yr", "returns_3yr", "returns_5yr", "expense_ratio", "fund_age_yr"]

df[["returns_1yr_scaled",
    "returns_3yr_scaled",
    "returns_5yr_scaled",
    "expense_ratio_scaled",
    "fund_age_scaled"]] = scaler.fit_transform(df[cols])

print(df[["returns_1yr_scaled",
          "returns_3yr_scaled",
          "returns_5yr_scaled",
          "expense_ratio_scaled",
          "fund_age_scaled"]].head())
# Stage 4: Fund Scoring

df["expense_score"] = 1 - df["expense_ratio_scaled"]

df["score"] = (
    df["returns_3yr_scaled"] * 0.40
    + df["expense_score"] * 0.25
    + df["fund_age_scaled"] * 0.15
    + df["returns_1yr_scaled"] * 0.10
    + df["returns_5yr_scaled"] * 0.10
)

df = df.sort_values("score", ascending=False)

print(df[["category", "returns_1yr", "returns_3yr",
          "returns_5yr", "expense_ratio", "fund_age_yr",
          "score"]].head(10))
# Top 30 funds
top30 = df.head(30)

# Export to Excel
top30.to_excel("top_30_mutual_funds.xlsx", index=False)

print(top30)
print("Top 30 funds Excel file save ")
