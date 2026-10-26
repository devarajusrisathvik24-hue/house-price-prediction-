
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered",
)


@st.cache_resource
def train_model():
    # Same sample data as your existing main.py
    X = np.array([
        [800], [1000], [1200], [1500],
        [1800], [2000], [2200], [2500],
    ])

    y = np.array([
        25, 30, 35, 42, 50, 55, 60, 68
    ])

    # Evaluate on held-out data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    evaluation_model = LinearRegression()
    evaluation_model.fit(X_train, y_train)

    predictions = evaluation_model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    # Refit the final model on all available sample data
    final_model = LinearRegression()
    final_model.fit(X, y)

    return final_model, mae


model, mae = train_model()

st.title("🏠 House Price Prediction")
st.write(
    "Enter a house size to estimate its price "
    "using a machine learning model."
)

area = st.slider(
    "House area (square feet)",
    min_value=800,
    max_value=2500,
    value=1600,
    step=50,
)

if st.button("Predict House Price", type="primary"):
    prediction = model.predict(
        np.array([[area]])
    )[0]

    st.success(f"Estimated price: ₹{prediction:.2f} lakhs")
    st.caption(f"House area: {area:,} sq. ft.")

with st.expander("Model information"):
    st.write("Algorithm: Linear Regression")
    st.write("Training samples: 8")
    st.write(f"Test-set MAE: {mae:.2f} lakhs")
    st.warning(
        "This is a demonstration using eight sample records. "
        "Its predictions are not validated market valuations."
    )
