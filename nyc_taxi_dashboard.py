import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# 🔐 Replace with your own Mapbox token or store it securely
px.set_mapbox_access_token("your_mapbox_token_here")



# Load your data
df = pd.read_csv("/Users/jeffreysedlack/Desktop/borough_taxi_data.csv")
st.write(df.columns)


st.title("🚕 NYC Yellow Taxi Dashboard (with Boroughs)")

# Raw data
if st.checkbox("Show raw data"):
    st.write(df.head())

# Trip Distance Distribution
st.subheader("📏 Trip Distance Distribution")
fig1, ax1 = plt.subplots()
ax1.hist(df["trip_distance"], bins=30)
ax1.set_xlabel("Trip Distance (miles)")
ax1.set_ylabel("Number of Rides")
st.pyplot(fig1)

# Dropoff Borough Counts
st.subheader("🏙️ Dropoff Borough Counts")
dropoff_counts = df["DropoffBorough"].value_counts().reset_index()
dropoff_counts.columns = ["DropoffBorough", "Ride Count"]

fig2, ax2 = plt.subplots()
ax2.bar(dropoff_counts["DropoffBorough"], dropoff_counts["Ride Count"], color="orange")
ax2.set_xlabel("Dropoff Borough")
ax2.set_ylabel("Ride Count")
ax2.set_title("Dropoffs by Borough")
st.pyplot(fig2)

# Average Trip Distance by Pickup Hour
st.subheader("Average Trip Distance by Pickup Hour")
avg_distance = df.groupby("pickup_hour")["trip_distance"].mean()
st.line_chart(avg_distance)

# dropoffs by borough
st.subheader("Dropoff Borough Counts")
dropoff_counts = df["DropoffBorough"].value_counts()
st.bar_chart(dropoff_counts)

# rides by pickup hour
st.subheader("Ride Count by Pickup Hour")
hour_counts = df["pickup_hour"].value_counts().sort_index()
st.line_chart(hour_counts)

# Pickup and dropoff map
import pydeck as pdk

st.subheader("Pickup Locations Map")
st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state=pdk.ViewState(
        latitude=40.75,
        longitude=-73.98,
        zoom=10,
        pitch=50,
    ),
    layers=[
        pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position='[pickup_longitude, pickup_latitude]',
            get_color='[200, 30, 0, 160]',
            get_radius=100,
        ),
    ],
))

# dropoff borough with bar chart
st.subheader("Dropoffs by Borough")
dropoff_counts = df["DropoffBorough"].value_counts()
st.bar_chart(dropoff_counts)

#Ride count by pickup hour
st.subheader("Ride Count by Pickup Hour")
ride_counts = df["pickup_hour"].value_counts().sort_index()
st.line_chart(ride_counts)


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.subheader("🧠 Clustering: Trip Distance vs Pickup Hour")

# Drop NA rows and cluster
cluster_data = df[["trip_distance", "pickup_hour"]].dropna()
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_data["Cluster"] = kmeans.fit_predict(cluster_data)

# Scatter plot
fig, ax = plt.subplots()
scatter = ax.scatter(
    cluster_data["trip_distance"],
    cluster_data["pickup_hour"],
    c=cluster_data["Cluster"],
    cmap="viridis",
    s=10
)
ax.set_xlabel("Trip Distance")
ax.set_ylabel("Pickup Hour")
ax.set_title("KMeans Clustering: Distance vs Hour")
fig.colorbar(scatter, label="Cluster")
st.pyplot(fig)

# Show table
st.markdown("### 📋 Clustered Data Sample")
st.dataframe(cluster_data.head(50))  # Show first 50 rows (adjust as needed)


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.subheader("📈 Clustering Quality vs Sample Size")

# 1. Your existing model_results dict
#    (e.g. {50: {...}, 100: {...}, ..., 20000: {...}})
#    Make sure you have `model_results` populated from your previous runs.

# Mock silhouette score results from previous clustering runs
model_results = {
    50:    {"silhouette": 0.7408},
    100:   {"silhouette": 0.7963},
    250:   {"silhouette": 0.7856},
    500:   {"silhouette": 0.7878},
    1000:  {"silhouette": 0.7862},
    2000:  {"silhouette": 0.8269},
    3000:  {"silhouette": 0.8270},
    4000:  {"silhouette": 0.8082},
    5000:  {"silhouette": 0.8361},
    10000: {"silhouette": 0.8216},
    15000: {"silhouette": 0.8277},
    20000: {"silhouette": 0.8266}
}


sample_list   = []
silhouette_list = []
for size, results in model_results.items():
    sample_list.append(size)
    silhouette_list.append(results["silhouette"])

# 2. Create a DataFrame
sil_df = pd.DataFrame({
    "Sample Size": sample_list,
    "Silhouette Score": silhouette_list
})
sil_df = sil_df.sort_values("Sample Size")

# 3. Plot the line chart
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sil_df["Sample Size"], sil_df["Silhouette Score"], marker="o")
ax.set_xlabel("Sample Size")
ax.set_ylabel("Silhouette Score")
ax.set_title("Clustering Quality vs Sample Size")
ax.grid(True)
st.pyplot(fig)

# 4. Show the data table
st.markdown("### 📋 Silhouette Scores by Sample Size")
st.dataframe(sil_df.set_index("Sample Size"))


#trips 
import pandas as pd

# Create ride_id column
df["ride_id"] = df.index  # or df.reset_index().index if needed

pickup_df = df[["ride_id", "LAT", "LONG", "pickup_hour"]].copy()
pickup_df.rename(columns={"LAT": "lat", "LONG": "lon", "pickup_hour": "hour"}, inplace=True)
pickup_df["point_type"] = "pickup"

dropoff_df = df[["ride_id", "LAT_DROPOFF", "LONG_DROPOFF", "pickup_hour"]].copy()
dropoff_df.rename(columns={"LAT_DROPOFF": "lat", "LONG_DROPOFF": "lon", "pickup_hour": "hour"}, inplace=True)
dropoff_df["point_type"] = "dropoff"

# Combine to long format
df_long = pd.concat([pickup_df, dropoff_df], axis=0)
df_long.sort_values(by=["ride_id", "point_type"], inplace=True)

import plotly.express as px

fig = px.line_map(
    df_long,
    lat="lat",
    lon="lon",
    color="ride_id",
    line_group="ride_id",
    animation_frame="hour",
    zoom=10,
    center={"lat": 40.75, "lon": -73.98},
    height=600,
    title="Animated Taxi Ride Paths on Map"
)
fig.update_layout(mapbox_style="carto-positron", showlegend=False)

# In Streamlit:
st.subheader("🗺️ Animated Taxi Ride Paths on Map")
st.plotly_chart(fig)




import matplotlib.pyplot as plt

# Group and count pickups by borough
pickup_counts = df["PickupBorough"].value_counts().sort_values(ascending=False)

# Plot
plt.figure(figsize=(8, 5))
pickup_counts.plot(kind="bar", color="cornflowerblue")
plt.title("Number of Pickups by NYC Borough", fontsize=14)
plt.xlabel("Borough", fontsize=12)
plt.ylabel("Pickup Count", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

st.markdown("## 🗺️ Heatmap of Ride Volume Between NYC Boroughs")

# Check if the needed columns exist
if {'PickupBorough', 'DropoffBorough', 'Ride Count'}.issubset(df_borough.columns):
    heatmap_data = df_borough.pivot(index='PickupBorough', columns='DropoffBorough', values='Ride Count')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu", ax=ax, cbar_kws={'label': 'Ride Count'})
    ax.set_title("Ride Volume Between Boroughs")
    ax.set_xlabel("Dropoff Borough")
    ax.set_ylabel("Pickup Borough")
    st.pyplot(fig)
else:
    st.warning("Required columns not found for heatmap.")
