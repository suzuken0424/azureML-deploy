import os
import json
import logging
from mlflow.pyfunc import load_model

def init():
    global model
    logging.info("AZUREML_MODEL_DIR: " + os.environ["AZUREML_MODEL_DIR"])
    model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "models")
    model = load_model(model_path)  
    logging.info("Init complete")

def run(mini_batch):
    logging.info(f"run method start: {__file__}, run({mini_batch})")

    input = json.loads(mini_batch)["data"]
    logging.info(f"input: {input}")
    predictions = model.predict(input)
    logging.info('Predictions:' + str(predictions))
    logging.info("Request processed")

    return predictions.tolist() 