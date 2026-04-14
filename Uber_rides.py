import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.cluster import KMeans

# Function to load and clean data
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)

    # Rename datetime column
    for col in df.columns:
        if isinstance(col, str) and ("at" in col.lower() or "date" in col.lower() or "time" in col.lower()):
            df.rename(columns={col: 'pickup_datetime'}, inplace=True)
            break

    # Convert to datetime
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
    df = df.dropna(subset=['pickup_datetime'])

    # Extract time features
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['month'] = df['pickup_datetime'].dt.month

    # Try to split location coordinates
    try:
        df[['start_lat', 'start_lon']] = df[df.columns[-3]].astype(str).str.split(',', expand=True).astype(float)
        df[['end_lat', 'end_lon']] = df[df.columns[-2]].astype(str).str.split(',', expand=True).astype(float)
    except:
        pass

    return df

# Main app
def main():
    st.title("Uber Rides Analysis Dashboard (Apr–Aug 2014)")

    st.sidebar.header("Select a Month's Dataset")
    dataset_paths = {
        "April 2014": r"D:\today\uber-raw-data-apr14.csv",
        "May 2014":   r"D:\today\uber-raw-data-may14.csv",
        "June 2014":  r"D:\today\uber-raw-data-jun14.csv",
        "July 2014":  r"D:\today\uber-raw-data-jul14.csv",
        "August 2014":r"D:\today\uber-raw-data-aug14.csv"
    }

    selected_month = st.sidebar.selectbox("Choose dataset", list(dataset_paths.keys()))
    file_path = dataset_paths[selected_month]

    # Load data
    df = load_data(file_path)

    st.subheader(f"Sample Data – {selected_month}")
    st.write(df.head())

    # Ride Frequency by Hour
    st.subheader("Ride Frequency by Hour")
    fig1, ax1 = plt.subplots()
    df['hour'].value_counts().sort_index().plot(kind='bar', color='skyblue', ax=ax1)
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Number of Rides")
    st.pyplot(fig1)

    # Ride Frequency by Day of Week
    st.subheader("Ride Frequency by Day of Week")
    fig2, ax2 = plt.subplots()
    df['day_of_week'].value_counts().sort_index().plot(kind='bar', color='lightgreen', ax=ax2)
    ax2.set_xlabel("Day of Week (0 = Mon)")
    ax2.set_ylabel("Number of Rides")
    st.pyplot(fig2)

    # KMeans Clustering of Start Locations
    if 'start_lat' in df.columns and 'start_lon' in df.columns:
        st.subheader("KMeans Clustering of Start Locations")
        coords = df[['start_lat', 'start_lon']].dropna()
        if len(coords) >= 5:
            kmeans = KMeans(n_clusters=5, random_state=0)
            coords['cluster'] = kmeans.fit_predict(coords)

            fig3, ax3 = plt.subplots()
            sns.scatterplot(x='start_lon', y='start_lat', hue='cluster', data=coords, ax=ax3, palette='Set1')
            ax3.set_title('KMeans Clustering of Start Locations')
            st.pyplot(fig3)
        else:
            st.warning("Not enough data for clustering.")

    # Forecasting Ride Demand
    st.subheader("Forecasting Ride Demand (Prophet)")
    daily_counts = df.groupby(df['pickup_datetime'].dt.date).size().reset_index(name='rides')
    daily_counts.columns = ['ds', 'y']

    if len(daily_counts) >= 2:
        model = Prophet()
        model.fit(daily_counts)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        fig4 = model.plot(forecast)
        st.pyplot(fig4)
    else:
        st.warning("Not enough data for time series forecasting.")

# Run the app
if __name__ == "__main__":
    main()
