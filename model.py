
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

#CLEANING THE DATA TO BE VOID OF NULL VALUES
df = pd.read_csv("/content/HousePriceIndia.csv")
df.fillna(method='ffill', inplace=True)

#CLEANING DATA TO REMOVE ALL REDUNDANT AND IRRELEVANT COLUMNS
df.drop(columns=["Date","waterfrontpresent", "RenovationYear", "Lattitude","Longitude"], inplace=True)
X = df.drop('Price', axis=1)
y = df['Price']
df.info()
scaler = MinMaxScaler()

numeric_columns= [ 'Price','number of bathrooms', 'living area', 'number of bedrooms', 'Built Year']
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
plt.figure(figsize = (16, 10))
sns.heatmap(df[numeric_columns].corr(), annot = True, cmap="YlGnBu")
plt.show()
categorical_columns=["Postal Code"]
for col in categorical_columns:
    df[col] = df[col].astype(str).str.lower().str.strip()
    df[col] = df[col].replace(to_replace=r'[^a-zA-Z0-9\s]', value='', regex=True)
city_data = df[['Postal Code']].values.reshape(-1, 1)
onehot_encoder = OneHotEncoder()
city_encoded = onehot_encoder.fit_transform(df[['Postal Code']])
s = (df.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)
print('No. of. categorical features: ', len(object_cols))
OH_encoder = OneHotEncoder(handle_unknown= 'ignore', sparse_output=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(df[object_cols]))
OH_cols.index = df.index
OH_cols.columns = OH_encoder.get_feature_names_out()
df_final = df.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)
df = pd.concat([df, df_final], axis=1)
df
label = df['Price']
features = df.drop('Price', axis=1)
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
ridge = Ridge(alpha=10)
# Train the model
ridge.fit(X_train, y_train)

# Make predictions on the test set
y_pred = ridge.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'R2 Score: {r2:.2f}')
print(f'Mean Absolute Error: {mae:.2f}')
print(f'Root Mean Squared Error: {rmse:.2f}')
print('Mean Squared Error:', mse)