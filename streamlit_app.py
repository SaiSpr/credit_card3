#-----------------------#
# IMPORT DES LIBRAIRIES #
#-----------------------#

import streamlit as st
import joblib
import streamlit.components.v1 as components
# import plotly.graph_objects as go
import matplotlib.pyplot as plt
# import plotly.express as px
st.set_option('deprecation.showPyplotGlobalUse', False)
import shap
import requests as re
import numpy as np
import pickle
import json
import pandas as pd
from streamlit_echarts import st_echarts
import seaborn as sns 
plt.style.use('ggplot')


st.title("Bank Loan Detection Web App")

st.image("image.jpg")





###################################################

data = joblib.load('sample_test_set.pickle')
infos_client = joblib.load('infos_client.pickle')
pret_client = joblib.load('pret_client.pickle')
preprocessed_data = joblib.load('preprocessed_data.pickle')
model = joblib.load('model.pkl')

column_names = preprocessed_data.columns.tolist()
expected_value = -2.9159221699244515
threshold = 100-10.344827586206896


data = joblib.load('sample_test_set.pickle')

threshold = 100-10.344827586206896

#Profile Client
profile_ID = st.sidebar.selectbox('Client Selection:',
                                  list(data.index))

st.write("profile_ID")


########################################################




st.sidebar.header('Select the Client_id:')


  
st.title("****Calculating the probability that a customer will repay their credit or not****")  

# logo sidebar 
st.sidebar.image("home_credit.png", use_column_width=True)
  
 
# Read 
list_file = open('cols_shap_local.pickle','rb')
cols_shap_local = pickle.load(list_file)
print(cols_shap_local)



#df_test_prod = pd.read_csv('df_test_ok_prod_100.csv', index_col=[0])
df_test_prod = pd.read_csv('df_test_ok_prod_100_V7.csv', index_col=[0])
df_test_prod['LOAN_DURATION'] = 1/df_test_prod['PAYMENT_RATE']
df_test_prod.drop(columns=['TARGET'], inplace=True)
df_test_prod_request  = df_test_prod.set_index('SK_ID_CURR')



df_train = pd.read_csv('df_train_prod_1.csv', index_col=[0])
df_train['LOAN_DURATION'] = 1/df_train['PAYMENT_RATE']

# Liste clients id sidebar 
list_client_prod = df_test_prod['SK_ID_CURR'].tolist()
client_id = st.sidebar.selectbox("Client Id list",list_client_prod)
client_id = int(client_id)

st.header(f'Credit request result for client {client_id}')



##################################    
# step = client_id
step = profile_ID    
threshold = 100-10.344827586206896
#########################################
    
  
  
#################################################    
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)    
    
#################################################
def explain_plot(id, pred):
    
    pipe_prod = joblib.load('LGBM_pipe_version7.pkl')
    df_test_prod_1 = df_test_prod.reset_index(drop=True)
    df_test_prod_request_1 = df_test_prod_1.reset_index().set_index(['SK_ID_CURR', 'index'])
    df_shap_local = df_test_prod_request_1[df_test_prod_request_1.columns[df_test_prod_request_1.columns.isin(cols_shap_local)]]
    values_id_client = df_shap_local.loc[[id]]
    

    explainer = shap.TreeExplainer(pipe_prod.named_steps['LGBM'])
      
    observation = pipe_prod.named_steps["transform"].transform(df_shap_local)
    observation_scale = pipe_prod.named_steps["scaler"].transform(observation)

    shap_values = explainer.shap_values(observation_scale)

    if pred == 1:
        p = st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][values_id_client.index[0][1],:],values_id_client))
    else:
        p = st_shap(shap.force_plot(explainer.expected_value[0], shap_values[0][values_id_client.index[0][1],:],values_id_client))
    return p
###################################################  
  
  
# Filtrer les clients rembourser et non rembourser 
df_train_rembourse = df_train[df_train['TARGET']== 0.0]
df_train_not_rembourse = df_train[df_train['TARGET']== 1.0]

# Sélectionner les colonnes pour le dashboard
cols_dashbord = ['SK_ID_CURR','AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_INCOME_TOTAL', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'AGE', 'LOAN_DURATION']



df_train_not_rembourse = df_train_not_rembourse[cols_dashbord]
df_train_rembourse = df_train_rembourse[cols_dashbord]

###################################################  
  
  
  
if st.button("Detection Result"):
    values = {
    "client_id": profile_ID,
    }


    st.write(f"""### These are the details:\n

    Client Id is: {profile_ID}\n
    
                """)

    res = re.post( url ="https://creditcard3-production.up.railway.app/predict", data = json.dumps(values))
    
    st.write("res", res)
    json_str = json.dumps(res.json())
    st.write("json_str", json_str)
        
#     st.write(json_str)
#     st.write(type(json_str))
    resp = json.loads(json_str)
    
#     API_GET = API_PRED+(str(profile_ID))
#     score_client = 100-int(re.get(API_GET).json()*100)
    score_client = resp
    st.write(score_client)
    if score_client < threshold:
        st.sidebar.write("Prêt refusé")
    else:
        st.sidebar.write("Prêt accordé.")
    
    pred = resp["prediction"]

    probability_value_0 = round(resp["probability_0"] * 100,2)
    probability_value_1 = round(resp["probability_1"] * 100,2)


    st.header(f'*Result of the credit application for the customer {client_id} is:*')


