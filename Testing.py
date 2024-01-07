import pandas as pd
import joblib

# Load the model car_predictor.joblib
loaded_model = joblib.load('car_predictor.joblib')
label_encoder_model = joblib.load('label_encoder_model.joblib')
label_encoder_brand = joblib.load('label_encoder_brand.joblib')
label_encoder_year = joblib.load('label_encoder_year.joblib')
scaler_price = joblib.load('scaler_price.joblib')
scaler_mileage = joblib.load('scaler_mileage.joblib')

new_data = [{
    'brand': 'merc',  # Replace with actual values and column names
    'model': 'G Class',
    'year': 2020,
    'mileage': 1350,
    'fuel_economy': 21.4
}]
new_df = pd.DataFrame(new_data)

new_df['brand'] = label_encoder_brand.transform(new_df['brand'])
new_df['model'] = label_encoder_model.transform(new_df['model'])
new_df['year'] = label_encoder_year.transform(new_df['year'])   # Convert to string if necessary
scaled_new_mileage = scaler_mileage.transform(new_df[['mileage']])
new_df['mileage'] = scaler_mileage.inverse_transform(scaled_new_mileage)

prediction = loaded_model.predict(new_df)
prediction_scaled = scaler_price.inverse_transform(prediction.reshape(-1, 1))

print(int(prediction_scaled[0][0]))