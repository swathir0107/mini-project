import streamlit as st
import requests
import pandas as pd
import numpy as np
import folium
import plotly.express as px

from streamlit_folium import st_folium
from folium.plugins import HeatMap

from geopy.geocoders import Nominatim

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense



st.set_page_config(
    page_title="Ultimate AI Weather Dashboard",
    layout="wide"
)

st.title("🌍 ULTIMATE AI WEATHER DASHBOARD")

st.write(
    "Live Weather + AI Forecast + Global Weather Layers"
)



API_KEY = "3f773cfe00da97f1e0036f4b72d43fa9"



DEFAULT_LAT = 12.97
DEFAULT_LON = 77.59



st.subheader("🗺️ Click Anywhere On Map")

main_map = folium.Map(
    location=[DEFAULT_LAT, DEFAULT_LON],
    zoom_start=5
)


weather_layers = {
    "🌡️ Temperature": "temp_new",
    "🌧️ Rain": "precipitation_new",
    "☁️ Clouds": "clouds_new",
    "🌬️ Wind": "wind_new",
    "🌀 Pressure": "pressure_new",
    "❄️ Snow": "snow_new",
    "🌊 Sea Level": "sea_level_new",
    "🥵 Feels Like": "feels_like_new"
}

for layer_name, layer_type in weather_layers.items():

    folium.TileLayer(
        tiles=f"https://tile.openweathermap.org/map/{layer_type}/{{z}}/{{x}}/{{y}}.png?appid={API_KEY}",
        attr="OpenWeatherMap",
        name=layer_name,
        overlay=True,
        control=True
    ).add_to(main_map)



folium.TileLayer(
    tiles="OpenStreetMap",
    name="🗺️ Street Map"
).add_to(main_map)

folium.TileLayer(
    tiles="CartoDB dark_matter",
    name="🌙 Dark Mode"
).add_to(main_map)

folium.TileLayer(
    tiles="CartoDB positron",
    name="☀️ Light Mode"
).add_to(main_map)



heat_data = [
    [12.97, 77.59, 30],
    [13.08, 80.27, 35],
    [17.38, 78.48, 38],
    [19.07, 72.87, 33],
    [28.61, 77.20, 40]
]

HeatMap(heat_data).add_to(main_map)



main_map.add_child(
    folium.LatLngPopup()
)

folium.LayerControl().add_to(main_map)


map_data = st_folium(
    main_map,
    width=1400,
    height=700
)



clicked_lat = DEFAULT_LAT
clicked_lon = DEFAULT_LON

city = "Bangalore"

if map_data["last_clicked"] is not None:

    clicked_lat = map_data["last_clicked"]["lat"]

    clicked_lon = map_data["last_clicked"]["lng"]

    st.success(
        f"📍 Selected Coordinates: {clicked_lat}, {clicked_lon}"
    )

    geolocator = Nominatim(
        user_agent="weather_app"
    )

    location = geolocator.reverse(
        f"{clicked_lat}, {clicked_lon}"
    )

    if location:

        address = location.raw["address"]

        city = (
            address.get("city")
            or address.get("town")
            or address.get("village")
            or address.get("state")
            or "Unknown"
        )


city = st.text_input(
    "📍 Selected City",
    city
)

st.write(f"🌍 Current City: {city}")



if st.button("Get Weather Forecast"):

    try:

      

        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={clicked_lat}&lon={clicked_lon}&appid={API_KEY}&units=metric"

        weather_response = requests.get(
            weather_url
        )

        weather_json = weather_response.json()

       

        if "main" not in weather_json:

            st.error("❌ Invalid API Key or API Error")
            st.write(weather_json)
            st.stop()

       

        main = weather_json["main"]

        wind = weather_json["wind"]

        weather = weather_json["weather"][0]

        st.subheader("🌦️ Current Weather")

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric(
            "🌡️ Temp",
            f"{main['temp']} °C"
        )

        col2.metric(
            "💧 Humidity",
            f"{main['humidity']} %"
        )

        col3.metric(
            "🌬️ Wind",
            f"{wind['speed']} km/h"
        )

        col4.metric(
            "☁️ Pressure",
            f"{main['pressure']} mb"
        )

        col5.metric(
            "🌤️ Condition",
            weather["main"]
        )


        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={clicked_lat}&lon={clicked_lon}&appid={API_KEY}&units=metric"

        forecast_response = requests.get(
            forecast_url
        )

        forecast_json = forecast_response.json()

      

        weather_data = []

        if "list" in forecast_json:

            for item in forecast_json["list"]:

                weather_data.append([

                    item["dt_txt"],

                    item["main"]["temp"],

                    item["main"]["humidity"],

                    item["main"]["pressure"],

                    item.get(
                        "rain",
                        {}
                    ).get("3h", 0)

                ])

        else:

            st.error("❌ Forecast API Error")
            st.write(forecast_json)
            st.stop()

    

        df = pd.DataFrame(

            weather_data,

            columns=[
                "Date",
                "Temperature",
                "Humidity",
                "Pressure",
                "Rain"
            ]
        )


        st.subheader("📋 Weather Forecast Dataset")

        st.dataframe(df)

       

        st.subheader("🌡️ Temperature Forecast Graph")

        temp_fig = px.line(
            df,
            x="Date",
            y="Temperature",
            markers=True,
            title="Temperature Forecast"
        )

        st.plotly_chart(
            temp_fig,
            use_container_width=True
        )


        st.subheader("💧 Humidity Graph")

        humidity_fig = px.line(
            df,
            x="Date",
            y="Humidity",
            markers=True,
            title="Humidity Forecast"
        )

        st.plotly_chart(
            humidity_fig,
            use_container_width=True
        )

   

        st.subheader("🌀 Pressure Graph")

        pressure_fig = px.line(
            df,
            x="Date",
            y="Pressure",
            markers=True,
            title="Pressure Forecast"
        )

        st.plotly_chart(
            pressure_fig,
            use_container_width=True
        )


        st.subheader("🌧️ Rain Forecast")

        rain_fig = px.bar(
            df,
            x="Date",
            y="Rain",
            title="Rain Prediction"
        )

        st.plotly_chart(
            rain_fig,
            use_container_width=True
        )

       

        st.subheader("🤖 AI Temperature Prediction")

        temperature_data = df[
            "Temperature"
        ].values.reshape(-1, 1)

        scaler = MinMaxScaler()

        scaled_data = scaler.fit_transform(
            temperature_data
        )

        X = []
        y = []

        sequence_length = 5

        for i in range(
            sequence_length,
            len(scaled_data)
        ):

            X.append(
                scaled_data[
                    i-sequence_length:i,
                    0
                ]
            )

            y.append(
                scaled_data[i, 0]
            )

        X = np.array(X)
        y = np.array(y)

        X = X.reshape(
            X.shape[0],
            X.shape[1],
            1
        )

    

        model = Sequential()

        model.add(
            LSTM(
                64,
                return_sequences=True,
                input_shape=(X.shape[1], 1)
            )
        )

        model.add(
            LSTM(64)
        )

        model.add(
            Dense(25)
        )

        model.add(
            Dense(1)
        )

  

        model.compile(
            optimizer="adam",
            loss="mean_squared_error"
        )

    

        st.write("🧠 Training AI Model...")

        history = model.fit(
            X,
            y,
            epochs=10,
            batch_size=4,
            verbose=0
        )

        st.success(
            "✅ AI Model Training Completed"
        )


        loss_df = pd.DataFrame({

            "Epoch": range(
                1,
                len(history.history["loss"]) + 1
            ),

            "Loss": history.history["loss"]
        })

        loss_fig = px.line(
            loss_df,
            x="Epoch",
            y="Loss",
            title="AI Training Loss"
        )

        st.plotly_chart(
            loss_fig,
            use_container_width=True
        )

  

        future_values = []

        current_batch = scaled_data[
            -sequence_length:
        ].reshape(
            1,
            sequence_length,
            1
        )

        future_days = 90

        progress_bar = st.progress(0)

        for i in range(future_days):

            pred = model.predict(
                current_batch,
                verbose=0
            )

            predicted_temp = scaler.inverse_transform(
                pred
            )[0][0]

            future_values.append(
                round(predicted_temp, 2)
            )

            current_batch = np.concatenate(
                (
                    current_batch[:, 1:, :],
                    pred.reshape(1, 1, 1)
                ),
                axis=1
            )

            progress_bar.progress(
                (i + 1) / future_days
            )

    

        future_dates = pd.date_range(
            start=pd.Timestamp.today(),
            periods=future_days,
            freq="D"
        )



        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Temperature": future_values
        })

      

        st.subheader("📅 AI Weather Forecast")

        st.dataframe(forecast_df)

 

        st.subheader("📈 AI Forecast Graph")

        future_fig = px.line(
            forecast_df,
            x="Date",
            y="Predicted Temperature",
            markers=True,
            title="📈 AI 90-Day Temperature Forecast"
        )

        future_fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Temperature (°C)",
            template="plotly_dark"
        )

        st.plotly_chart(
            future_fig,
            use_container_width=True
        )


        st.subheader("📊 Forecast Statistics")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "🔥 Max Temp",
            f"{forecast_df['Predicted Temperature'].max():.2f} °C"
        )

        col2.metric(
            "❄️ Min Temp",
            f"{forecast_df['Predicted Temperature'].min():.2f} °C"
        )

        col3.metric(
            "📈 Avg Temp",
            f"{forecast_df['Predicted Temperature'].mean():.2f} °C"
        )

   

        csv = forecast_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download Forecast CSV",
            data=csv,
            file_name="ai_weather_forecast.csv",
            mime="text/csv"
        )

        st.success(
            "✅ AI Forecast Generated Successfully"
        )

    except Exception as e:

        st.error("❌ Error Occurred")
        st.write(e)
