from utils.extracao import extrair_itens, extrair_endereco
from utils.precos import precos
from model.salvar_modelo import carregar_modelo
from model.pagamento_model import prever_pagamento


model, tokenizer, le = carregar_modelo()


def calcular_total(itens):
    total = 0
    resumo = []

    for produto, qtd in itens:
        valor = precos[produto] * qtd
        total += valor
        resumo.append(f"{qtd}x {produto} (R${valor})")

    return total, resumo


def processar_pedido(texto):
    itens = extrair_itens(texto)
    endereco = extrair_endereco(texto)
    pagamento = prever_pagamento(texto, model, tokenizer, le)

    total, resumo = calcular_total(itens)

    return resumo, total, pagamento, endereco


if __name__ == "__main__":
    pedido = input("Digite o pedido: ")

    itens, total, pagamento, endereco = processar_pedido(pedido)

    print("\n--- PEDIDO ---")
    for item in itens:
        print(item)

    print(f"\nTotal: R${total}")
    print(f"Pagamento: {pagamento}")
    print(f"Endereço: {endereco}")
