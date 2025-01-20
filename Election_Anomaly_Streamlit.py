import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np

# Function to load and preprocess data
def load_data():
  
    df_scaled = pd.read_csv("C:\\Users\\DIME\\Desktop\\DIME\\SCHOOL\\SEMESTER 5\\ISRAEL ELECTIONS\\israeli_elections_results_1996_to_2015_scaled.csv")
    
    # Preprocess your data if needed
    
    return df_scaled

# Function to train Isolation Forest model
def train_model(df_selected):
    # Initialize the Isolation Forest model
    isolation_forest = IsolationForest(random_state=42)

    # Fit the model to the selected features
    isolation_forest.fit(df_selected)
    
    return isolation_forest

# Function to detect anomalies
def detect_anomalies(model, df_selected):
    # Predict outliers/anomalies
    outliers = model.predict(df_selected)

    # Convert predictions to binary labels (1 for normal, -1 for anomaly)
    anomaly_labels = np.where(outliers == 1, 0, 1)

    # Count the number of anomalies detected
    num_anomalies = np.sum(anomaly_labels == 1)
    
    return num_anomalies

# Main function
def main():
    st.title("Anomaly Detection App")
    
    # Load data
    st.header("Load Data")
    df_scaled = load_data()  # Load your scaled data here
    
    # Train model
    st.header("Train Model")
    model = train_model(df_scaled)
    st.success("Model trained successfully!")
    
    # Sidebar for user input
    st.sidebar.title("Input Features")
    # Add input fields for features
    registered_voters = st.sidebar.number_input("Registered Voters", min_value=0)
    votes = st.sidebar.number_input("Votes", min_value=0)
    invalid_votes = st.sidebar.number_input("Invalid Votes", min_value=0)
    valid_votes = st.sidebar.number_input("Valid Votes", min_value=0)
    
    # Create dataframe with selected features
    df_selected = pd.DataFrame({
        'Registered_voters': [registered_voters],
        'votes': [votes],
        'invalid_votes': [invalid_votes],
        'valid_votes': [valid_votes]
    })
    
    # Perform anomaly detection
    st.header("Anomaly Detection")
    num_anomalies = detect_anomalies(model, df_selected)
    st.write("Number of anomalies detected:", num_anomalies)

if __name__ == "__main__":
    main()
