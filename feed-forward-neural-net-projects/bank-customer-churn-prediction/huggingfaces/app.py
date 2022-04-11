from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

saved_model = load_model('churn_model2.h5')
scaler = MinMaxScaler()

def churn_prediction(CreditScore,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Location):
    Geography_France,Geography_Germany,Geography_Spain=0,0,0
    df = pd.DataFrame.from_dict({
        "Credit score":[CreditScore],
        "Is female?":[1 if Gender=='Female' else 0],
        "Age":[Age],
        'Tenure':[Tenure],
        'Balance':[Balance],
        'Number of products':[NumOfProducts],
        'Has credit card?': [1 if HasCrCard=='Yes' else 0],
        'Is a active member?':[1 if IsActiveMember=='Yes' else 0],
        'Estimated Salary': [EstimatedSalary],
        'Geography_France': [1 if Location=='France' else 0],
        'Geography_Germany':[1 if Location=='Germany' else 0],
        'Geography_Spain':[1 if Location=='Spain' else 0]
        
    })
    cols_to_scale = ["Credit score",'Age','Tenure','Balance','Number of products','Estimated Salary']
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    pred=saved_model.predict(df)
    pred = pred[0][0]
    churn_prob=str(round(pred,2))
    churn_prob_d = round(round(pred,2) * 100)
    non_churn_prob_d = 100 - churn_prob_d 
    non_churn_prob = str(round(1-pred,2))
    return {f"probability customer will exit:  {churn_prob_d}%":churn_prob , f"probability customer will stay:  { non_churn_prob_d}%": non_churn_prob}


import gradio as gr

iface = gr.Interface(fn=churn_prediction,
                    inputs=['number',gr.inputs.Radio(['Female','Male']),'number','number','number','number',gr.inputs.Radio(['Yes','No']),gr.inputs.Radio(['Yes','No']),'number',gr.inputs.Radio(['France','Germany','Spain'])],
                    outputs=['label'])

iface.launch()