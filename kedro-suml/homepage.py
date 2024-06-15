import streamlit as st
import pandas as pd
from autogluon.tabular import TabularPredictor
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from pathlib import Path

# Uzyskanie ścieżki do katalogu głównego projektu
project_path = Path.cwd()

# Inicjalizacja projektu Kedro
bootstrap_project(project_path)

# Tworzenie sesji Kedro
with KedroSession.create(project_path=project_path) as session:
    context = session.load_context()
    catalog = context.catalog

    # Załaduj wytrenowany model
    predictor = catalog.load('autoML_predictor')

    # Funkcja do wykonywania predykcji na podstawie danych wejściowych
    def predict_price(input_data: pd.DataFrame):
        prediction = predictor.predict(input_data)
        return prediction[0]

    # Strona główna Streamlit
    st.title("Car Price Prediction")

    currency = st.selectbox('Currency', ['USD', 'EUR', 'PLN'])
    condition = st.selectbox('Condition', ['New', 'Used'])
    vehicle_brand = st.text_input('Vehicle Brand')
    vehicle_model = st.text_input('Vehicle Model')
    vehicle_version = st.text_input('Vehicle Version', value="None")
    vehicle_generation = st.text_input('Vehicle Generation', value='NaN')
    production_year = st.number_input('Production Year', min_value=1900, max_value=2024, step=1)
    mileage_km = st.number_input('Mileage (km)', min_value=0)
    power_hp = st.number_input('Power (HP)', min_value=0)
    displacement_cm3 = st.number_input('Displacement (cm³)', min_value=0)
    fuel_type = st.text_input('Fuel Type')
    co2_emissions = st.number_input('CO2 Emissions', min_value=0)
    drive = st.text_input('Drive')
    transmission = st.text_input('Transmission')
    vehicle_type = st.text_input('Type')
    doors_number = st.number_input('Doors Number', min_value=2, max_value=5, step=1)
    colour = st.text_input('Colour')

    # Tworzymy DataFrame z wprowadzonymi danymi
    input_data = pd.DataFrame({
        'Currency': [currency],
        'Condition': [condition],
        'Vehicle_brand': [vehicle_brand],
        'Vehicle_model': [vehicle_model],
        'Vehicle_version': [vehicle_version if vehicle_version != "" else "None"],
        'Vehicle_generation': [vehicle_generation if vehicle_generation != "" else "NaN"],
        'Production_year': [production_year],
        'Mileage_km': [mileage_km],
        'Power_HP': [power_hp],
        'Displacement_cm3': [displacement_cm3],
        'Fuel_type': [fuel_type],
        'CO2_emissions': [co2_emissions],
        'Drive': [drive],
        'Transmission': [transmission],
        'Type': [vehicle_type],
        'Doors_number': [doors_number],
        'Colour': [colour]
    })

    if st.button('Predict Price'):
        # Wykonaj predykcję
        price = predict_price(input_data)
        st.write(f'The predicted price of the car is: {price}')
