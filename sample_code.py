import pandas as pd
from sklearn.ensemble import RandomForestRegressor

data = {
    "Week": [1, 2, 3, 4, 5],
    "Sales": [200, 250, 300, 280, 350]
}

df = pd.DataFrame(data)

X = df[["Week"]]
y = df["Sales"]

model = RandomForestRegressor()
model.fit(X, y)

prediction = model.predict([[6]])

print("Predicted Sales:", prediction)
