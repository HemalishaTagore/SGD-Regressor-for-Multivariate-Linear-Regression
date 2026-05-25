import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor

data = {
    'Size': [1000, 1200, 1500, 1800, 2000],
    'Bedrooms': [2, 2, 3, 3, 4],
    'Price': [300000, 350000, 400000, 450000, 500000],
    'Occupants': [2, 3, 4, 5, 6]
}

df = pd.DataFrame(data)

X = df[['Size', 'Bedrooms']]

y = df[['Price', 'Occupants']]

model = MultiOutputRegressor(SGDRegressor())

model.fit(X, y)

prediction = model.predict([[1600, 3]])

print("Predicted Price:", prediction[0][0])
print("Predicted Occupants:", prediction[0][1])

plt.scatter(df['Size'], df['Price'])

plt.plot(df['Size'], model.predict(X)[:,0])

plt.xlabel("House Size")
plt.ylabel("House Price")
plt.title("House Price Prediction")

plt.show()

plt.scatter(df['Size'], df['Occupants'])

plt.plot(df['Size'], model.predict(X)[:,1])

plt.xlabel("House Size")
plt.ylabel("Occupants")
plt.title("Occupants Prediction")

plt.show()
