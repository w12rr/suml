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
    predictor = catalog.load('regressor')

    # Funkcja do wykonywania predykcji na podstawie danych wejściowych
    def predict_price(input_data: pd.DataFrame):
        prediction = predictor.predict(input_data)
        return prediction[0]

    # Mappings for label encoding
    mappings = {
        'Currency': {'USD': 0, 'EUR': 1, 'PLN': 2},
        'Condition': {'New': 0, 'Used': 1},
        'Fuel_type': {'Diesel': 0, 'Electric': 1, 'Gasoline': 2, 'Gasoline\\LPG': 3},
        'Drive': {'4x4 (attached automatically)': 0, '4x4 (permanent)': 1, 'Front wheels': 2, 'Rear wheels': 3},
        'Transmission': {'Automatic': 0, 'Manual': 1},
        'Type': {'city_cars': 0, 'compact': 1, 'convertible': 2, 'coupe': 3, 'sedan': 4, 'small_cars': 5,
                 'station_wagon': 6, 'SUV': 7},
        'Colour': {'beige': 0, 'black': 1, 'blue': 2, 'brown': 3, 'burgundy': 4, 'golden': 5, 'gray': 6, 'green': 7,
                   'other': 8, 'red': 9, 'silver': 10, 'violet': 11, 'white': 12, 'yellow': 13},
        'Vehicle_brand': {brand: i for i, brand in enumerate(["Abarth", "Acura", "Aixam", "Alfa Romeo"])},
        'Vehicle_model': {model: i for i, model in enumerate(
            ["124", "145", "146", "147", "155", "156", "159", "164", "166", "4C", "500", "595", "A721", "A741", "A751",
             "Brera", "City", "Coupe", "Crossline", "Crossover", "Crosswagon", "Giulia", "Giulietta", "Grande Punto",
             "GT", "GTO", "GTV", "MDX", "Mito", "Other", "RDX", "RL", "Roadline", "Scouty", "Scouty R", "Spider",
             "Sportwagon", "Stelvio", "TL", "TSX"])},
        'Vehicle_version': {version: i for i, version in enumerate(
            ["0.9 TwinAir", "1.3 JTDM", "1.3 JTDM (ECO)", "1.3 JTDM Distinctive", "1.3 JTDM ECO",
             "1.3 JTDM Progression", "1.4 16V", "1.4 16V MultiAir", "1.4 8V", "1.4 Distinctive", "1.4 Impression",
             "1.4 Junior", "1.4 MultiAir", "1.4 MultiAir Distinctive", "1.4 MultiAir Progression",
             "1.4 MultiAir Progression S&S", "1.4 MultiAir Scorpione", "1.4 Progression", "1.4 TB 16V",
             "1.4 TB 16V Multiair", "1.4 TB 16V Multiair TCT", "1.4 TB Distinctive", "1.4 TB Impression",
             "1.4 TB MultiAir", "1.4 TB MultiAir Distinctive", "1.4 TB MultiAir Distinctive S&S",
             "1.4 TB MultiAir Distinctive S&S TCT EU6", "1.4 TB MultiAir Distinctive TCT", "1.4 TB MultiAir Exclusive",
             "1.4 TB MultiAir Progression", "1.4 TB MultiAir Progression TCT", "1.4 TB MultiAir QV",
             "1.4 TB MultiAir QV S&S TCT EU6", "1.4 TB MultiAir Sprint TCT", "1.4 TB MultiAir Super",
             "1.4 TB MultiAir Super TCT", "1.4 TB Progression", "1.4 TB Sport", "1.4 TB Sprint", "1.4 TB Super",
             "1.4 TB Veloce", "1.4 TSpark L", "1.6 JTDM", "1.6 JTDM 16V", "1.6 JTDM Distinctive",
             "1.6 JTDM Distinctive S&S", "1.6 JTDM Progression", "1.6 JTDM Sport TCT", "1.6 JTDM Sprint",
             "1.6 JTDM Sprint TCT", "1.6 JTDM Super", "1.6 JTDM TCT", "1.6 JTDM Veloce", "1.6 T.S 16V",
             "1.6 T.S Black Line", "1.6 T.S Distinctive", "1.6 T.S Exclusive", "1.6 T.S Impression",
             "1.6 T.S Progression", "1.6 TSpark L", "1.8 MPI 16V", "1.8 TBI 16V", "1.8 TS Impression",
             "1.8 TS Progression", "1.8MPI Impression", "1.8MPI Progression", "1.8TS Impression", "1.9 JTD",
             "1.9 JTD 16V Distinctive", "1.9 JTD 16V Progress", "1.9 JTD 16V Q2", "1.9 JTD 8V Impression",
             "1.9 JTD 8V Progression", "1.9 JTD Distinctive", "1.9 JTD Impression", "1.9 JTD Progression",
             "1.9 JTD16v Impression", "1.9 JTDM 16V DPF", "1.9 JTDM 16V DPF Q-Tronic", "1.9 JTDM 8V", "1.9 JTDM 8V DPF",
             "1.9 JTDM 8V DPF Eco", "1.9JTD 16V Black Line", "1.9JTD 16V Progression", "1.9JTD Impr Business",
             "1.9JTD16v Distinctive", "1.9JTD16v Progression", "1.9JTDM Distinctive", "1.9JTDM Impression",
             "1.9JTDM Progression", "1.9JTDM Q-Distinctive", "1.9JTDM ti", "1.9JTS Distinctive", "1.9JTS Impression",
             "1.9JTS Progression", "156 1.9JTD Distinctive", "156 1.9JTD Progression", "156 1.9JTD Stradale Progr",
             "159 1.9JTDM Distinctive", "159 2.4JTDM Q-Progression", "1750 TBi Quadrifoglio Verde", "1750 TBi QV TCT",
             "1750 TBi Veloce TCT", "1750TBi", "1750TBi Distinctive", "1750TBi Progression", "1750TBi Sport Plus",
             "2.0 16v TSpark", "2.0 JTDM", "2.0 JTDM 16V", "2.0 JTDM 16V DPF", "2.0 JTDM 16V TCT",
             "2.0 JTDM Distinctive", "2.0 JTDM Distinctive EU6", "2.0 JTDM Exclusive", "2.0 JTDM Exclusive TCT",
             "2.0 JTDM Exclusive TCT EU6", "2.0 JTDM Progression", "2.0 JTDM Progression TCT", "2.0 JTDM Super",
             "2.0 JTS", "2.0 JTS Selespeed", "2.0 Seles.Distinctive", "2.0 T.S Distinctive", "2.0 T.S Exclusive",
             "2.0 TS", "2.0 TS Distinctive", "2.0 TS Progression", "2.0 TSpark 16v", "2.0 TSpark 16V",
             "2.0 TSpark Super", "2.0 Turbo", "2.0 Turbo 16V AT8", "2.0 Turbo 16V AT8-Q4",
             "2.0 Turbo B-Tech Edition Q4", "2.0 Turbo Business", "2.0 Turbo Business Q4", "2.0 Turbo Executive Q4",
             "2.0 Turbo First Edition Q4", "2.0 Turbo Q4", "2.0 Turbo Sprint", "2.0 Turbo Sprint Q4", "2.0 Turbo Super",
             "2.0 Turbo Super Q4", "2.0 Turbo TI", "2.0 Turbo TI Q4", "2.0 Turbo Veloce", "2.0 Turbo Veloce Q4",
             "2.0 Turbo Veloce TI Q4", "2.0 V6 TB", "2.0-16 TSpark", "2.0JTDM", "2.0JTDM Distinctive",
             "2.0JTDM Progression", "2.0JTDM Sport", "2.0JTDM Sport Plus", "2.0JTS Distinctive", "2.0JTS Progression",
             "2.2 D Turbo", "2.2 D Turbo Sprint", "2.2 D Turbo Super", "2.2 D Turbo Veloce Q4", "2.2 Diesel",
             "2.2 Diesel 16V AT8", "2.2 Diesel 16V AT8-Q4", "2.2 Diesel AT8", "2.2 Diesel AT8-Q4", "2.2 JTDM Veloce Q4",
             "2.2 JTS 16V", "2.2JTS", "2.2JTS Distinctive", "2.2JTS Progression", "2.2JTS Sel Distinctive",
             "2.2JTS Selespeed ti", "2.2JTS Sky View", "2.4 JTD", "2.4 JTD Distinctive", "2.4 JTDM 20V DPF",
             "2.4 JTDM 20V DPF Q-Tronic", "2.4 JTDM 20V DPF Q4", "2.4JTD 20V Progress", "2.4JTD20v Progression",
             "2.4JTDM", "2.4JTDM 4x4 Sportiva", "2.4JTDM 4x4 ti", "2.4JTDM Distinctive", "2.4JTDM Progression",
             "2.4JTDM Q-Distinctive", "2.4JTDM Q-Progression", "2.4JTDM Q-ti", "2.4JTDM Q4 Progression",
             "2.4JTDM Sky View", "2.4JTDM Sportiva", "2.4JTDM ti", "2.5 V6 24v", "2.9 Bi Turbo V6 Quadrifoglio",
             "2.9 Bi Turbo V6 Quadrifoglio Nring", "2.9 V6 Bi-Turbo Quadrifoglio Q4", "3.0 Quadrifoglio Verd",
             "3.0 Super", "3.0 V6 Lusso", "3.2 24V Distinctive", "3.2 JTS V6 24V Q4", "3.2 JTS V6 24V Q4 Q-Tronic",
             "3.2JTS 4x2 Sportiva", "3.2JTS Q4", "3.2JTS Q4 Distinctive", "3.2JTS Q4 Q-ti", "3.2JTS Q4 ti",
             "3.7 V6 Base", "595 1.4 T-Jet 16v Elaborabile", "TB 1.4 16V", "TB 1.4 16V MultiAir",
             "TB 1.4 16V MultiAir TCT", "ver-1-4-multiair-distinctive-s-s", "ver-1-9-jtd-m--jet-dpf",
             "ver-2-4-jtdm-20v-dpf"])}
    }
    # Funkcja do mapowania wartości kategorycznych na liczby
    def map_values(df, mappings):
        for col, mapping in mappings.items():
            df[col] = df[col].map(mapping)
        return df

    # Strona główna Streamlit
    st.title("Car Price Prediction")
    currency = st.selectbox('Currency', ['USD', 'EUR', 'PLN'])
    condition = st.selectbox('Condition', ['New', 'Used'])
    vehicle_brand = st.selectbox('Vehicle Brand', ["Abarth", "Acura", "Aixam", "Alfa Romeo"])
    vehicle_model = st.selectbox('Vehicle Model', ["124", "145", "146", "147", "155", "156", "159", "164", "166", "4C", "500", "595", "A721", "A741", "A751", "Brera", "City", "Coupe", "Crossline", "Crossover", "Crosswagon", "Giulia", "Giulietta", "Grande Punto", "GT", "GTO", "GTV", "MDX", "Mito", "Other", "RDX", "RL", "Roadline", "Scouty", "Scouty R", "Spider", "Sportwagon", "Stelvio", "TL", "TSX"])
    vehicle_version = st.selectbox('Vehicle Version', ["0.9 TwinAir", "1.3 JTDM", "1.3 JTDM (ECO)", "1.3 JTDM Distinctive", "1.3 JTDM ECO", "1.3 JTDM Progression", "1.4 16V", "1.4 16V MultiAir", "1.4 8V", "1.4 Distinctive", "1.4 Impression", "1.4 Junior", "1.4 MultiAir", "1.4 MultiAir Distinctive", "1.4 MultiAir Progression", "1.4 MultiAir Progression S&S", "1.4 MultiAir Scorpione", "1.4 Progression", "1.4 TB 16V", "1.4 TB 16V Multiair", "1.4 TB 16V Multiair TCT", "1.4 TB Distinctive", "1.4 TB Impression", "1.4 TB MultiAir", "1.4 TB MultiAir Distinctive", "1.4 TB MultiAir Distinctive S&S", "1.4 TB MultiAir Distinctive S&S TCT EU6", "1.4 TB MultiAir Distinctive TCT", "1.4 TB MultiAir Exclusive", "1.4 TB MultiAir Progression", "1.4 TB MultiAir Progression TCT", "1.4 TB MultiAir QV", "1.4 TB MultiAir QV S&S TCT EU6", "1.4 TB MultiAir Sprint TCT", "1.4 TB MultiAir Super", "1.4 TB MultiAir Super TCT", "1.4 TB Progression", "1.4 TB Sport", "1.4 TB Sprint", "1.4 TB Super", "1.4 TB Veloce", "1.4 TSpark L", "1.6 JTDM", "1.6 JTDM 16V", "1.6 JTDM Distinctive", "1.6 JTDM Distinctive S&S", "1.6 JTDM Progression", "1.6 JTDM Sport TCT", "1.6 JTDM Sprint", "1.6 JTDM Sprint TCT", "1.6 JTDM Super", "1.6 JTDM TCT", "1.6 JTDM Veloce", "1.6 T.S 16V", "1.6 T.S Black Line", "1.6 T.S Distinctive", "1.6 T.S Exclusive", "1.6 T.S Impression", "1.6 T.S Progression", "1.6 TSpark L", "1.8 MPI 16V", "1.8 TBI 16V", "1.8 TS Impression", "1.8 TS Progression", "1.8MPI Impression", "1.8MPI Progression", "1.8TS Impression", "1.9 JTD", "1.9 JTD 16V Distinctive", "1.9 JTD 16V Progress", "1.9 JTD 16V Q2", "1.9 JTD 8V Impression", "1.9 JTD 8V Progression", "1.9 JTD Distinctive", "1.9 JTD Impression", "1.9 JTD Progression", "1.9 JTD16v Impression", "1.9 JTDM 16V DPF", "1.9 JTDM 16V DPF Q-Tronic", "1.9 JTDM 8V", "1.9 JTDM 8V DPF", "1.9 JTDM 8V DPF Eco", "1.9JTD 16V Black Line", "1.9JTD 16V Progression", "1.9JTD Impr Business", "1.9JTD16v Distinctive", "1.9JTD16v Progression", "1.9JTDM Distinctive", "1.9JTDM Impression", "1.9JTDM Progression", "1.9JTDM Q-Distinctive", "1.9JTDM ti", "1.9JTS Distinctive", "1.9JTS Impression", "1.9JTS Progression", "156 1.9JTD Distinctive", "156 1.9JTD Progression", "156 1.9JTD Stradale Progr", "159 1.9JTDM Distinctive", "159 2.4JTDM Q-Progression", "1750 TBi Quadrifoglio Verde", "1750 TBi QV TCT", "1750 TBi Veloce TCT", "1750TBi", "1750TBi Distinctive", "1750TBi Progression", "1750TBi Sport Plus", "2.0 16v TSpark", "2.0 JTDM", "2.0 JTDM 16V", "2.0 JTDM 16V DPF", "2.0 JTDM 16V TCT", "2.0 JTDM Distinctive", "2.0 JTDM Distinctive EU6", "2.0 JTDM Exclusive", "2.0 JTDM Exclusive TCT", "2.0 JTDM Exclusive TCT EU6", "2.0 JTDM Progression", "2.0 JTDM Progression TCT", "2.0 JTDM Super", "2.0 JTS", "2.0 JTS Selespeed", "2.0 Seles.Distinctive", "2.0 T.S Distinctive", "2.0 T.S Exclusive", "2.0 TS", "2.0 TS Distinctive", "2.0 TS Progression", "2.0 TSpark 16v", "2.0 TSpark 16V", "2.0 TSpark Super", "2.0 Turbo", "2.0 Turbo 16V AT8", "2.0 Turbo 16V AT8-Q4", "2.0 Turbo B-Tech Edition Q4", "2.0 Turbo Business", "2.0 Turbo Business Q4", "2.0 Turbo Executive Q4", "2.0 Turbo First Edition Q4", "2.0 Turbo Q4", "2.0 Turbo Sprint", "2.0 Turbo Sprint Q4", "2.0 Turbo Super", "2.0 Turbo Super Q4", "2.0 Turbo TI", "2.0 Turbo TI Q4", "2.0 Turbo Veloce", "2.0 Turbo Veloce Q4", "2.0 Turbo Veloce TI Q4", "2.0 V6 TB", "2.0-16 TSpark", "2.0JTDM", "2.0JTDM Distinctive", "2.0JTDM Progression", "2.0JTDM Sport", "2.0JTDM Sport Plus", "2.0JTS Distinctive", "2.0JTS Progression", "2.2 D Turbo", "2.2 D Turbo Sprint", "2.2 D Turbo Super", "2.2 D Turbo Veloce Q4", "2.2 Diesel", "2.2 Diesel 16V AT8", "2.2 Diesel 16V AT8-Q4", "2.2 Diesel AT8", "2.2 Diesel AT8-Q4", "2.2 JTDM Veloce Q4", "2.2 JTS 16V", "2.2JTS", "2.2JTS Distinctive", "2.2JTS Progression", "2.2JTS Sel Distinctive", "2.2JTS Selespeed ti", "2.2JTS Sky View", "2.4 JTD", "2.4 JTD Distinctive", "2.4 JTDM 20V DPF", "2.4 JTDM 20V DPF Q-Tronic", "2.4 JTDM 20V DPF Q4", "2.4JTD 20V Progress", "2.4JTD20v Progression", "2.4JTDM", "2.4JTDM 4x4 Sportiva", "2.4JTDM 4x4 ti", "2.4JTDM Distinctive", "2.4JTDM Progression", "2.4JTDM Q-Distinctive", "2.4JTDM Q-Progression", "2.4JTDM Q-ti", "2.4JTDM Q4 Progression", "2.4JTDM Sky View", "2.4JTDM Sportiva", "2.4JTDM ti", "2.5 V6 24v", "2.9 Bi Turbo V6 Quadrifoglio", "2.9 Bi Turbo V6 Quadrifoglio Nring", "2.9 V6 Bi-Turbo Quadrifoglio Q4", "3.0 Quadrifoglio Verd", "3.0 Super", "3.0 V6 Lusso", "3.2 24V Distinctive", "3.2 JTS V6 24V Q4", "3.2 JTS V6 24V Q4 Q-Tronic", "3.2JTS 4x2 Sportiva", "3.2JTS Q4", "3.2JTS Q4 Distinctive", "3.2JTS Q4 Q-ti", "3.2JTS Q4 ti", "3.7 V6 Base", "595 1.4 T-Jet 16v Elaborabile", "TB 1.4 16V", "TB 1.4 16V MultiAir", "TB 1.4 16V MultiAir TCT", "ver-1-4-multiair-distinctive-s-s", "ver-1-9-jtd-m--jet-dpf", "ver-2-4-jtdm-20v-dpf"])
    vehicle_generation = st.text_input('Vehicle Generation', value='NaN')
    production_year = st.number_input('Production Year', min_value=1900, max_value=2024, step=1)
    mileage_km = st.number_input('Mileage (km)', min_value=0)
    power_hp = st.number_input('Power (HP)', min_value=0)
    displacement_cm3 = st.number_input('Displacement (cm³)', min_value=0)
    fuel_type = st.selectbox('Fuel Type', ["Diesel", "Electric", "Gasoline", "Gasoline\\LPG"])
    co2_emissions = st.number_input('CO2 Emissions', min_value=0)
    drive = st.selectbox('Drive', ["4x4 (attached automatically)", "4x4 (permanent)", "Front wheels", "Rear wheels"])
    transmission = st.selectbox('Transmission', ["Automatic", "Manual"])
    vehicle_type = st.selectbox('Type', ["city_cars", "compact", "convertible", "coupe", "sedan", "small_cars", "station_wagon", "SUV"])
    doors_number = st.number_input('Doors Number', min_value=2, max_value=5, step=1)
    colour = st.selectbox('Colour', ["beige", "black", "blue", "brown", "burgundy", "golden", "gray", "green", "other", "red", "silver", "violet", "white", "yellow"])

    # Tworzymy DataFrame z wprowadzonymi danymi
    input_data = pd.DataFrame({
        'Currency': [currency],
        'Condition': [condition],
        'Vehicle_brand': [vehicle_brand],
        'Vehicle_model': [vehicle_model],
        'Vehicle_version': [vehicle_version if vehicle_version != "NaN" else 1],
        'Vehicle_generation': [vehicle_generation if vehicle_generation != "NaN" else 1],
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

    # Mapowanie wartości kategorycznych na liczby
    input_data = map_values(input_data, mappings)

    if st.button('Predict Price'):
        # Wykonaj predykcję
        price = predict_price(input_data) *-1
        st.write(f'The predicted price of the car is: {price}')
