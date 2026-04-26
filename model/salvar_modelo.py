from model.pagamento_model import treinar_modelo
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import joblib
import os


MODEL_PATH = "models/distilbert_pagamento"
ENCODER_PATH = "models/label_encoder.pkl"


def salvar_modelo():
    model, tokenizer, le = treinar_modelo()

    os.makedirs(MODEL_PATH, exist_ok=True)

    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)


def carregar_modelo():
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)

    return model, tokenizer, le
