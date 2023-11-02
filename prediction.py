import streamlit as st
import pandas as pd
import json
import joblib
import pickle

# Load models and transformers
with open('model_scaler.pkl', 'rb') as file_1:
    num_transformer = pickle.load(file_1)

with open('model_train', 'rb') as file_2:
    adaboost_model = pickle.load(file_2)

with open('model_encoder', 'rb') as file_3:
    cat_transformer = pickle.load(file_3)

with open('list_num.json', 'r') as file_4:
    num_col = json.load(file_4)

with open('list_cat.json', 'r') as file_5:
    cat_nom_col = json.load(file_5)

with open('model_prepocessor', 'rb') as file_6:
    preprocessor = pickle.load(file_6)

# Load AdaBoostClassifier model
with open('model_adaboost', 'rb') as file_7:
    ada_model = pickle.load(file_7)

# Define selected features
selected_features = ['age', 'workclass', 'educational-num', 'marital-status', 'occupation', 'relationship', 'gender',
                     'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

def run():
    # Membuat Form
    with st.form(key='form_credit_default_check'):
        age = st.number_input('Age', min_value=0, max_value=100, value=25, help='Age'), 
        workclass = st.selectbox('Workclass', ['Private', 'Local-gov']), 
        educational_num = st.number_input('Educational Num', min_value=1, max_value=16, value=7, help='Education level'), 
        marital_status = st.selectbox('Marital Status', ['Never-married', 'Married-civ-spouse']), 
        occupation = st.selectbox('Occupation', ['Machine-op-inspct', 'Farming-fishing']), 
        relationship = st.selectbox('Relationship', ['Own-child', 'Husband']), 
        gender = st.selectbox('Gender', ['Male', 'Female']), 
        capital_gain = st.number_input('Capital Gain', min_value=0, max_value=99999, value=0, help='Capital Gain'), 
        capital_loss = st.number_input('Capital Loss', min_value=0, max_value=99999, value=0, help='Capital Loss'), 
        hours_per_week = st.number_input('Hours per Week', min_value=1, max_value=100, value=40, help='Hours per week'), 
        native_country = st.selectbox('Native Country', ['United-States']), 
        income = st.radio('Income', [0, 1], index=0)

        submitted = st.form_submit_button('Predict')

    data_inf = {
        'age': age[0],
        'workclass': workclass,
        'educational-num': educational_num[0],
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'gender': gender,
        'capital-gain': capital_gain[0],
        'capital-loss': capital_loss[0],
        'hours-per-week': hours_per_week[0],
        'native-country': native_country
    }

    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)

    if submitted:
        # Preprocess the input data
        data_inf_transformed = preprocessor.transform(data_inf)
        
        # Predict using Adaboost model
        y_pred_adaboost = ada_model.predict(data_inf_transformed[selected_features])
        st.write('# Prediction (AdaBoost): ', str(int(y_pred_adaboost)))

if __name__ == '__main__':
    run()
