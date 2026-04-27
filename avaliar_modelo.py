from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from model.salvar_modelo import carregar_modelo
from model.pagamento_model import prever_pagamento


def criar_dataset_teste():
    return [
        ("quero pagar no pix", "PIX"),
        ("vou fazer pix", "PIX"),
        ("aceita pagamento via pix", "PIX"),

        ("vou pagar em dinheiro", "DINHEIRO"),
        ("pagamento será em dinheiro", "DINHEIRO"),
        ("pago com dinheiro na entrega", "DINHEIRO"),

        ("vou pagar no credito", "CREDITO"),
        ("passa no cartao de credito", "CREDITO"),
        ("credito por favor", "CREDITO"),

        ("vou pagar no debito", "DEBITO"),
        ("passa no cartao de debito", "DEBITO"),
        ("debito por favor", "DEBITO"),
    ]


def avaliar_modelo():
    model, tokenizer, le = carregar_modelo()

    dataset = criar_dataset_teste()

    y_true = []
    y_pred = []

    for texto, label_real in dataset:
        label_prevista = prever_pagamento(texto, model, tokenizer, le)

        y_true.append(label_real)
        y_pred.append(label_prevista)

    print("Acurácia:")
    print(accuracy_score(y_true, y_pred))

    print("\nRelatório de classificação:")
    print(classification_report(y_true, y_pred))

    print("\nMatriz de confusão:")
    print(confusion_matrix(y_true, y_pred, labels=le.classes_))

    print("\nClasses:")
    print(le.classes_)


if __name__ == "__main__":
    avaliar_modelo()
