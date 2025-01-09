import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('/Users/anuhyasamudrala/Documents/Anu_uncc/Deeplearning/ANNClassification/ANN/model.h5')

# Load the encoders and scaler
with open('/Users/anuhyasamudrala/Documents/Anu_uncc/Deeplearning/ANNClassification/ANN/label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('/Users/anuhyasamudrala/Documents/Anu_uncc/Deeplearning/ANNClassification/ANN/onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('/Users/anuhyasamudrala/Documents/Anu_uncc/Deeplearning/ANNClassification/ANN/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


## streamlit app
st.title('Customer Churn PRediction')

# User input
# Input fields
credit_score = st.number_input(
    'Credit Score', 
    min_value=0, 
    max_value=1000, 
    value=619, 
    help="A numerical score that represents the creditworthiness of the customer. "
         "Higher scores indicate better creditworthiness."
)

geography = st.selectbox(
    'Geography', 
    onehot_encoder_geo.categories_[0], 
    help="The region or country where the customer resides."
)

gender = st.selectbox(
    'Gender', 
    label_encoder_gender.classes_, 
    help="The gender of the customer. Please select Male or Female."
)

age = st.slider(
    'Age', 
    18, 
    100, 
    help="The age of the customer in years."
)

tenure = st.slider(
    'Tenure', 
    0, 
    50, 
    help="The number of years the customer has been with the bank."
)

num_of_products = st.number_input(
    'Number of Products Used',
    min_value=1,
    max_value=4,
    value=1,
    help="The number of products or services the customer is using with the bank. "
         "For example: a savings account, a credit card, a loan, or an investment account."
)

is_active_member = st.selectbox(
    'Is Active Member', 
    [0, 1], 
    format_func=lambda x: "Yes" if x == 1 else "No",
    help="Indicates whether the customer is currently an active member of the bank (Yes = 1, No = 0)."
)

balance = st.number_input(
    'Balance', 
    min_value=0.0, 
    value=0.0, 
    help="The total balance in the customer's bank account (in dollars)."
)

has_cr_card = st.selectbox(
    'Has Credit Card', 
    [0, 1], 
    format_func=lambda x: "Yes" if x == 1 else "No",
    help="Indicates whether the customer owns a credit card (Yes = 1, No = 0)."
)

estimated_salary = st.number_input(
    'Estimated Salary', 
    min_value=0.0, 
    help="The annual estimated salary of the customer (in dollars)."
)

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)


# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]


if prediction_proba > 0.5:
    st.error(
            f"The customer may leave the bank. "
        )
else:
    st.success(
            f"The customer is likely to stay with the bank. "
        )
