import pandas as pd
import streamlit as st
import requests
import json

df = pd.read_csv('selection_test.csv')
PORT = 5001
def score_model(pdf, params):
    headers = {"Content-Type": "application/json"}
    url = f"http://127.0.0.1:{PORT}/invocations"
    ds_dict = {"dataframe_split": pdf, "params": params}
    data_json = json.dumps(ds_dict, allow_nan=True)

    response = requests.request(method="POST", headers=headers, url=url, data=data_json)
    response.raise_for_status()

    return response.json()



def main():
    MLFLOW_URI = 'http://127.0.0.1:5001/invocations'
    CORTEX_URI = 'http://0.0.0.0:8890/'
    RAY_SERVE_URI = 'http://127.0.0.1:8000/regressor'

    api_choice = st.sidebar.selectbox(
        'Quelle API souhaitez vous utiliser',
        ['MLflow', 'Cortex', 'Ray Serve'])

    st.title('pret Prediction')

    numero_ligne = st.number_input('numero de ligne',
                                 min_value=-2, value=0, step=1)
    

    predict_btn = st.button('Prédire')
    if predict_btn:
        if 0 <= numero_ligne < len(df):
            data = df.iloc[[numero_ligne]].to_dict(orient='split')
            pred = None

            if api_choice == 'MLflow':
                pred = score_model(data, params={})  # Assurez-vous de passer les bons paramètres

            if pred and "predictions" in pred:
                prediction_value = pred["predictions"][0]
                st.write('Le risque de prêt est {:.2f}'.format(prediction_value))
        else:
            st.error("Numéro de ligne invalide. Veuillez entrer un numéro dans la plage valide.")




if __name__ == '__main__':
    main()
