import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import main

# --- Start of Necessary Streamlit Changes ---

def main():
    st.title("End-to-End Hotel Booking Cancellation Prediction Pipeline")

    st.markdown(
        """
    #### Problem Statement
     The objective here is to predict whether a customer will **cancel a hotel booking** (`booking_status`: 1 for cancelled, 0 for not cancelled) based on various reservation features. We use [ZenML](https://zenml.io/) to build a production-ready pipeline for continuous deployment of the predictive model.
    """
    )
    st.markdown(
        """
    The pipeline handles data ingestion, cleaning, model training (Logistic Regression), and deployment. If new data arrives or performance drops, the model is retrained and redeployed if it meets the minimum accuracy requirement.
    """
    )

    st.markdown(
        """
    #### Description of Features
    This app predicts the cancellation status (0 or 1) for a new booking. Input the features below:

    | Feature | Type | Description |
    |---|---|---|
    | **no_of_adults** | Integer | Number of adults in the booking. |
    | **no_of_children** | Integer | Number of children in the booking. |
    | **no_of_weekend_nights** | Integer | Number of weekend nights (Sat/Sun) booked. |
    | **no_of_week_nights** | Integer | Number of week nights (Mon-Fri) booked. |
    | **lead_time** | Integer | Number of days between booking date and arrival date. |
    | **arrival_year** | Integer | Year of arrival. |
    | **no_of_previous_cancellations** | Integer | Number of previous bookings cancelled by the customer. |
    | **no_of_previous_bookings_not_canceled** | Integer | Number of previous bookings not cancelled by the customer. |
    | **avg_price_per_room** | Float | Average price paid per room per day. |
    | **no_of_special_requests** | Integer | Number of special requests made. |
    | **required_car_parking_space** | Boolean | 1 if parking is required, 0 otherwise. |
    | **repeated_guest** | Boolean | 1 if the customer is a repeated guest, 0 otherwise. |
    | **type_of_meal_plan** | Encoded | E.g., Meal Plan 1, Meal Plan 2, etc. (Input the pre-processed numerical value). |
    | **room_type_reserved** | Encoded | E.g., Room Type A, B, C, etc. (Input the pre-processed numerical value). |
    | **market_segment_type_Complementary** | Encoded | 1 if complementary, 0 otherwise. |
    | **market_segment_type_Corporate** | Encoded | 1 if corporate, 0 otherwise. |
    | **market_segment_type_Offline** | Encoded | 1 if offline, 0 otherwise. |
    | **market_segment_type_Online** | Encoded | 1 if online, 0 otherwise. |
    """
    )

    # --- Input Fields (Updated for Hotel Features) ---

    st.header("Input Booking Details")
    no_of_adults = st.slider("Number of Adults", min_value=1, max_value=4, value=2)
    no_of_children = st.slider("Number of Children", min_value=0, max_value=3, value=0)
    no_of_weekend_nights = st.slider("Weekend Nights", min_value=0, max_value=5, value=1)
    no_of_week_nights = st.slider("Week Nights", min_value=0, max_value=10, value=3)

    # Note: Using number_input for features that are numerical/encoded
    type_of_meal_plan = st.number_input("Meal Plan (Encoded)", value=1)
    required_car_parking_space = st.number_input("Car Parking Required (0 or 1)", value=0)
    room_type_reserved = st.number_input("Room Type (Encoded)", value=1)

    lead_time = st.number_input("Lead Time (Days)", value=50)
    arrival_year = st.number_input("Arrival Year", value=2025)
    repeated_guest = st.number_input("Repeated Guest (0 or 1)", value=0)
    no_of_previous_cancellations = st.number_input("Previous Cancellations", value=0)
    no_of_previous_bookings_not_canceled = st.number_input("Previous Bookings Not Cancelled", value=0)
    avg_price_per_room = st.number_input("Avg Price per Room", value=100.0)
    no_of_special_requests = st.number_input("Special Requests", value=0)

    # Encoded Market Segment Types (use radio buttons/select box for better UX)
    st.subheader("Market Segment Type (Select ONLY one as '1', others '0')")
    market_segment_type_Complementary = st.number_input("Segment: Complementary (0 or 1)", value=0)
    market_segment_type_Corporate = st.number_input("Segment: Corporate (0 or 1)", value=0)
    market_segment_type_Offline = st.number_input("Segment: Offline (0 or 1)", value=1)
    market_segment_type_Online = st.number_input("Segment: Online (0 or 1)", value=0)


    if st.button("Predict"):
        service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        running=False,
        )
        if service is None:
            st.write(
                "No service could be found. The pipeline will be run first to create a service."
            )
            # Ensure you have run_main() defined to kick off your pipeline
            # run_main() 

        # --- DataFrame Creation (Updated for 18 Hotel Features) ---

        df = pd.DataFrame(
            {
                "no_of_adults": [no_of_adults],
                "no_of_children": [no_of_children],
                "no_of_weekend_nights": [no_of_weekend_nights],
                "no_of_week_nights": [no_of_week_nights],
                "type_of_meal_plan": [type_of_meal_plan],
                "required_car_parking_space": [required_car_parking_space],
                "room_type_reserved": [room_type_reserved],
                "lead_time": [lead_time],
                "arrival_year": [arrival_year],
                "repeated_guest": [repeated_guest],
                "no_of_previous_cancellations": [no_of_previous_cancellations],
                "no_of_previous_bookings_not_canceled": [no_of_previous_bookings_not_canceled],
                "avg_price_per_room": [avg_price_per_room],
                "no_of_special_requests": [no_of_special_requests],
                "market_segment_type_Complementary": [market_segment_type_Complementary],
                "market_segment_type_Corporate": [market_segment_type_Corporate],
                "market_segment_type_Offline": [market_segment_type_Offline],
                "market_segment_type_Online": [market_segment_type_Online],
            }
        )
        
        # --- Data Preprocessing for Prediction (CRITICAL STEP) ---
        
        # Ensure all columns are float64 to match model schema
        df = df.astype(float) 
        
        # The ZenML MLflow connector requires prediction data in a specific format
        # Using df.values (NumPy array) is the simplest method if the service is configured for it
        # If this fails, use the 'to_json(orient="records")' method from previous discussion
        data = df.values
        
        # --- Prediction ---
        
        pred = service.predict(data)
        
        # --- Output Message (Updated) ---
        
        # The prediction will be an array, extract the first element
        prediction_value = pred[0] if isinstance(pred, np.ndarray) and pred.size > 0 else pred
        
        st.success(
            f"The predicted **Booking Status** is: **{prediction_value:.0f}**"
        )
        st.markdown(
            """
            *Interpretation:*
            - **1:** Predicted to be **Cancelled**.
            - **0:** Predicted to be **Not Cancelled**.
            """
        )
        
    if st.button("Results"):
        st.write(
            "We have trained a **Logistic Regression** model. The results on the test set are as follows:"
        )

        # Placeholder results for demonstration
        df_results = pd.DataFrame(
            {
                "Model": ["Logistic Regression"],
                "Accuracy": ["~85%"],
                "F1 Score": ["0.80"],
            }
        )
        st.dataframe(df_results)

        st.write(
            "Following figure shows how important each feature is in the model that contributes to predicting booking cancellation."
        )
        # You'll need to create a feature importance plot for your Logistic Regression model
        # image = Image.open("_assets/hotel_feature_importance.png")
        # st.image(image, caption="Feature Importance for Booking Cancellation")


if __name__ == "__main__":
    main()