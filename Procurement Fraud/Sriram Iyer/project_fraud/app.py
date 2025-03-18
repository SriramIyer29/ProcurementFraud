import pickle
import streamlit as st
pickle_in = open('model_dep.pkl', 'rb')
Random_Forest= pickle.load(pickle_in)



def prediction(step,amount,type_CASH_IN,type_CASH_OUT,type_DEBIT,type_PAYMENT,type_TRANSFER,oldbalanceOrg,newbalanceOrig, oldbalanceDest, newbalanceDest):  
   
    prediction = Random_Forest.predict([[step,amount,type_CASH_IN,type_CASH_OUT,type_DEBIT,type_PAYMENT,type_TRANSFER,oldbalanceOrg,newbalanceOrig, oldbalanceDest, newbalanceDest]])
    if (prediction == 0):
      return 'The transaction is not fraud'
    else:
      return 'The transaction is fraud'

def main():
    st.title("Fraud detection")
    htm_temp = """
    <div>
    <h2>Fraud detection ML App </h2>
    </div>
    """
    st.markdown(htm_temp,unsafe_allow_html=True)
    step = st.text_input("step")
    amount = st.text_input("amount")
    type_CASH_IN =st.text_input("type_CASH_IN")
    type_CASH_OUT =st.text_input("type_CASH_OUT")
    type_DEBIT = st.text_input("type_DEBIT")
    type_PAYMENT =st.text_input("type_PAYMENT")
    type_TRANSFER = st.text_input("text_TRANSFER")
    oldbalanceOrg = st.text_input("oldbalanceOrg")
    newbalanceOrig = st.text_input("newbalanceOrig")
    oldbalanceDest = st.text_input("oldbalanceDest")
    newbalanceDest = st.text_input("newbalanceDest")
    result=""
    if st.button("Predict"):
        result=prediction(step,amount,type_CASH_IN,type_CASH_OUT,type_DEBIT,type_PAYMENT,type_TRANSFER,oldbalanceOrg,newbalanceOrig, oldbalanceDest, newbalanceDest)
    st.success('The output is {}'.format(result))


if __name__=='__main__':
    main()
