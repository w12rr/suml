"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.6
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_car_sale(car_sale_dataset: pd.DataFrame):

    car_sale_dataset.drop(columns=['Index', 'Offer_publication_date', 'Offer_location', 'Features', 'Origin_country', 'First_owner',
                     'First_registration_date'], inplace=True)

    # Inicjalizacja LabelEncoder
    label_encoder = LabelEncoder()

    # Lista kolumn kategorycznych do zakodowania
    categorical_columns = [
        "Currency",
        "Condition",
        "Vehicle_brand",
        "Vehicle_model",
        "Vehicle_version",
        "Vehicle_generation",
        "Fuel_type",
        "Drive",
        "Transmission",
        "Type",
        "Colour"
    ]

    # Kodowanie etykiet dla kolumn kategorycznych
    for column in categorical_columns:
        car_sale_dataset[column] = label_encoder.fit_transform(car_sale_dataset[column])

    # Usunięcie wierszy z brakującymi wartościami
    car_sale_dataset.dropna(inplace=True)

    return car_sale_dataset
