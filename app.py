from fastapi import FastAPI, File, Query, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, PlainTextResponse
import uvicorn
import joblib
import numpy as np
from pydantic import BaseModel
import pandas as pd 


app = FastAPI(
    title="Bank Loan Detection API",
    description="""An API that utilises a Machine Learning model that detects the customers eligibility of a loan""",
    version="1.0.0", debug=True)


# model = joblib.load('credit_fraud.pkl')

@app.get("/", response_class=PlainTextResponse)
async def running():
  note = """
Credit Card Fraud Detection API 🙌🏻

Note: add "/docs" to the URL to get the Swagger UI Docs or "/redoc"
  """
  return note

favicon_path = 'favicon.png'
@app.get('/favicon.png', include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path)
																	
class fraudDetection(BaseModel):
    client_id:float

	
#importing dataframe of test customer data
df_test_prod = pd.read_csv('df_test_ok_prod_100_V7.csv', index_col=[0])
# supprimer target
df_test_prod.drop(columns=['TARGET'], inplace=True)
# mettre SK_ID_CURR en index 
df_test_prod_request  = df_test_prod.set_index('SK_ID_CURR')
# Création list des clients 
clients_id = df_test_prod["SK_ID_CURR"].tolist() 

##################################################
# Chargement du modèle
model = joblib.load('model.pkl')
data = joblib.load('sample_test_set.pickle')
list_ID = data.index.tolist()
# Enregistrer le model
classifier = model.named_steps['classifier']
##########################################


# @app.post('/predict')
# def predict(data : fraudDetection):
                                                                                                                                                                                                                                
#     features = np.array([data.client_id])

# #     id = features[0]

#     client_id = features[0]
	
#     predictions = model.predict_proba(data).tolist()
#     predict_proba = []
#     for pred, ID in zip(predictions, list_ID):
#         if ID == client_id:
#             predict_proba.append(pred[1])
#     return predict_proba[0]

@app.post("/predict")
async def predict(data : fraudDetection):
    request_body = await data.json()
    client_id = request_body['client_id']
    predictions = model.predict_proba(data).tolist()
    predict_proba = []
    for pred, ID in zip(predictions, list_ID):
        if ID == client_id:
            predict_proba.append(pred[1])
    return predict_proba[0]

#     if id not in clients_id:
#         raise HTTPException(status_code=404, detail="client's id not found")
    
#     else:
        
        
#         pipe_prod = joblib.load('LGBM_pipe_version7.pkl')
    
#         values_id_client = df_test_prod_request.loc[[id]]
       
#         # Defining the best threshold
#         prob_preds = pipe_prod.predict_proba(values_id_client)
        
#         #Fast_API_prob_preds
#         threshold = 0.332 #threshold
#         y_test_prob = [1 if prob_preds[i][1]> threshold else 0 for i in range(len(prob_preds))]
        
       
#     return {"prediction": id}

