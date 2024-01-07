import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib

df = pd.read_csv('Data/combined_filtered2.csv')
df['brand'] = df['brand'].astype('string')
df['model'] = df['model'].astype('string')

avg_val = np.mean((df[["price"]]),axis =0)
label_encoder_model = LabelEncoder()
df['model'] = label_encoder_model.fit_transform(df['model'])

label_encoder_year = LabelEncoder()
df['year'] = label_encoder_year.fit_transform(df['year'])

label_encoder_brand = LabelEncoder()
df['brand'] = label_encoder_brand.fit_transform(df['brand'])

joblib.dump(label_encoder_model, 'label_encoder_model.joblib')
joblib.dump(label_encoder_brand, 'label_encoder_brand.joblib')
joblib.dump(label_encoder_year, 'label_encoder_year.joblib')

scaler_price = MinMaxScaler()
scaler_mileage = MinMaxScaler()
df["price"] = scaler_price.fit_transform(df[["price"]])
df['mileage'] = scaler_mileage.fit_transform(df[["mileage"]])

joblib.dump(scaler_price, 'scaler_price.joblib')
joblib.dump(scaler_mileage, 'scaler_mileage.joblib')

y = df["price"]
x = df.drop(['price'], axis=1)

X_temp, X_test, y_temp, y_test = train_test_split(x, y, test_size=0.2, random_state = 42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state = 42)

rf_regressor = RandomForestRegressor(n_estimators=100,random_state=42)
rf_regressor.fit(X_train, y_train)

model_dump = 'car_predictor.joblib'
joblib.dump(rf_regressor, model_dump)



rf_regressor = RandomForestRegressor(n_estimators=100,random_state=42)
rf_regressor.fit(X_train, y_train)


model_dump = 'car_predictor.joblib'
joblib.dump(rf_regressor, model_dump)
