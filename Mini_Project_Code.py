# Importing Dependencies...
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st


# Load Data
smart_card_data = pd.read_csv(r"E:\Smart Card Fraud Detection System\Dataset\smartcard.csv")

# separate legitimate and fraudulent transactions...
legit = smart_card_data[smart_card_data.Class == 0]
fraud = smart_card_data[smart_card_data.Class == 1]

# undersamplinglegitimate transactions to balance the classes..
legit_sample = legit.sample(n = len(fraud), random_state=2)
smart_card_data = pd.concat([legit_sample, fraud], axis = 0)

List = ['Time','Class']
# Splitting data into training and test data...
X = smart_card_data.drop(columns=List,axis = 1)
Y = smart_card_data['Class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 2)

# The remaining number of columns..
Columns_Remain = len(smart_card_data.columns) - len(List)

# train with LOGISTIC REGRESSION MODEL...
model = LogisticRegression(max_iter=1000, solver='lbfgs')
model.fit(X_train, Y_train)

# Evaluate the accuracy scores of both TRAINING and TEST data...
train_accuracy = accuracy_score(model.predict(X_train), Y_train)
test_accuracy = accuracy_score(model.predict(X_test), Y_test)

# WEB APP...
st.set_page_config(page_title="Fraud Detection App", page_icon=":smard_card:", layout="wide")

custom_css = f"""
<div style="padding: 10px;">
    <h1 style="color: blue;">Smart Card Fraud Detection System</h1>
</div>
"""

st.markdown(custom_css, unsafe_allow_html=True)


st.info(
    "Welcome to the Smart Card Fraud Detection System. "
    "Please enter transaction details below and click 'Submit' for predictions."
)

st.header("User Input")

user_input = st.text_input("Enter transaction details separated by commas:", key="user_input")

submit = st.button("Submit",type="primary")

if submit:
    if user_input:

        user_input_df = user_input.split(",")
        input_length = len(user_input_df)
        res=any(' ' in ele for ele in user_input_df)
        

        if(res):
            st.error("Entered null values!!!")

        else:
            # ELSE BLOCK...
            if(Columns_Remain > input_length):
                st.write("Enter remaining ", Columns_Remain - input_length, " values.")

            elif(Columns_Remain < input_length):
                st.write("Entered ", input_length - Columns_Remain, " additional values.")

            else:
                try:
                    features = np.asarray(user_input_df,dtype=np.float64)
                    prediction = model.predict(features.reshape(1,-1))

                    if prediction[0] == 1:
                        st.error("🚨 Fraudulent Transaction Detected!")

                    else:
                        st.success("✅ Legitimate Transaction")

                except:
                    st.error("Please Enter Float type values only")