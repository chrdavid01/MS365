import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

from data import download_kaggle_dataset

KAGGLE_DATA_URL = "https://www.kaggle.com/datasets/jeremylarcher/american-house-prices-and-demographics-of-top-cities"
download_kaggle_dataset(KAGGLE_DATA_URL, "house_prices")

plt.rcParams['figure.figsize'] = [20, 5]

df = pandas.read_csv("/content/drive/MyDrive/MS365/Assignments/assignment4_data.csv")
df.drop(columns=["Latitude", "Longitude"], inplace=True)
df.head(15)

df.describe()

df["Median Household Income"].fillna(value=df["Median Household Income"].mean(), inplace=True)

df.drop(columns=["Zip Code", "Address", "City", "State", "County"], inplace=True)

df.head(15)

y_column = "Price"
x_columns = list(df.columns)
x_columns.remove(y_column)

x_data = df[x_columns]
y_data = df[y_column]

print(x_data.head())
print("-----")
print(y_data.head())

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.20, random_state=14)

lr = LinearRegression()

lr.fit(x_train, y_train)

df.plot.scatter(x="Beds", y="Price")
df.plot.scatter(x="Baths", y="Price")
df.plot.scatter(x="Living Space", y="Price")
df.plot.scatter(x="Zip Code Population", y="Price")
df.plot.scatter(x="Zip Code Density", y="Price")
df.plot.scatter(x="Median Household Income", y="Price")
plt.show()

x_data.corr()

y_estimates = lr.predict(x_test)
residuals = y_test - y_estimates

plt.scatter(y_estimates, residuals)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.grid(True)
plt.show()

me = mean_absolute_error(y_test, y_estimates)

mse = mean_squared_error(y_test, y_estimates, squared=True)

rmse = mean_squared_error(y_test, y_estimates, squared=False)

r_squared = lr.score(x_test, y_test)
print(f'ME = {me}')
print(f'MSE = {mse}')
print(f'RMSE = {rmse}')
print(f'R_squared = {r_squared}')

