from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.preprocessing import LabelEncoder


def criar_dataset():
    dados = [
        ("vou pagar no pix", "PIX"),
        ("pagamento em pix", "PIX"),
        ("pagar via pix", "PIX"),
        ("aceita pix?", "PIX"),

        ("vou pagar em dinheiro", "DINHEIRO"),
        ("pagar no dinheiro", "DINHEIRO"),
        ("pode ser dinheiro", "DINHEIRO"),

        ("cartao de credito", "CREDITO"),
        ("vou pagar no credito", "CREDITO"),
        ("passar no credito", "CREDITO"),

        ("cartao de debito", "DEBITO"),
        ("vou pagar no debito", "DEBITO"),
        ("pagar no cartao de debito", "DEBITO"),
    ]

    textos = [t[0] for t in dados]
    labels = [t[1] for t in dados]

    return textos, labels


def treinar_modelo():
    textos, labels = criar_dataset()

    le = LabelEncoder()
    y = le.fit_transform(labels)

    dataset = Dataset.from_dict({
        "text": textos,
        "label": y
    })

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length")

    dataset = dataset.map(tokenize)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(le.classes_)
    )

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        logging_steps=10,
        save_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    trainer.train()

    return model, tokenizer, le


def prever_pagamento(texto, model, tokenizer, label_encoder):
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits.detach().numpy()
    predicted_class_id = np.argmax(logits)
    return label_encoder.inverse_transform([predicted_class_id])[0]
