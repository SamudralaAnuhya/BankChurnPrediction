# Bank Churn Prediction

This project predicts whether a bank customer is likely to leave (churn) or stay, using machine learning techniques. The model is built using an **Artificial Neural Network (ANN) Classification** approach to ensure high prediction accuracy. 

The application provides an interactive interface for real-time predictions, making it a valuable tool for customer retention strategies.

# Live Demo
You can try out the application here: [Bank Churn Prediction App](https://bankchurnpredictions.streamlit.app/)

<img width="631" alt="image" src="https://github.com/user-attachments/assets/baaf6501-e5b1-4af4-8d7b-fdb3b8cdbd71" />



## Features

- **Artificial Neural Network (ANN)**: Utilizes a deep learning model designed for binary classification.
- **Interactive User Interface**: Built with Streamlit for easy user interaction.
- **Real-Time Predictions**: Takes customer data as input and predicts churn likelihood instantly.
- **Scalable Deployment**: Deployed on Streamlit Cloud for easy accessibility.


## Technologies Used

- **Python 3.10**
- **TensorFlow 2.14.1**
- **Keras**
- **Streamlit 1.41.1**
- **Pandas**
- **Scikit-Learn**

## Getting Started

### Prerequisites

- Python 3.10 or later
- Recommended: Virtual environment (e.g., `venv` or Conda)

### Clone the Repository

```bash
git clone https://github.com/SamudralaAnuhya/BankChurnPrediction.git
cd BankChurnPrediction
```

### Create a Virtual Environment

```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application Locally

```bash
streamlit run app.py
```

## Deployment

This project is deployed using **Streamlit Cloud**.

1. Ensure `requirements.txt` includes all dependencies.
2. Use `runtime.txt` to specify the Python version:
   ```
   python-3.10.12
   ```
3. Push changes to the GitHub repository. The app will be automatically deployed if connected to Streamlit Cloud.

---

## How to Use

1. Launch the app.
2. Input customer details, including:
   - Credit score
   - Geography
   - Gender
   - Age
   - Tenure
   - Balance
   - Number of products
   - Active member status
   - Estimated salary
3. "Predict" to see whether the customer is likely to churn or not to  churn.

## Model Details

- **Model Type**: TensorFlow Keras Sequential Model
- **Classification Method**: Artificial Neural Network (ANN)
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam
- **Input Features**:
  - Demographics: Geography, Gender, Age
  - Account Details: Tenure, Balance, Number of Products
  - Financial Indicators: Credit Score, Active Member Status, Estimated Salary
