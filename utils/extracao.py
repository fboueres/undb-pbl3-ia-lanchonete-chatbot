import re
from utils.precos import precos, sinonimos


def normalizar_texto(texto):
    texto = texto.lower()

    for base, variacoes in sinonimos.items():
        for termo in variacoes:
            texto = texto.replace(termo, base)

    return texto


def extrair_itens(texto):
    texto = normalizar_texto(texto)
    itens = []

    for produto in precos:
        if produto in texto:
            match = re.search(rf"(\d+)\s*{produto}", texto)

            if match:
                quantidade = int(match.group(1))
            else:
                quantidade = 1

            itens.append((produto, quantidade))

    return itens


def extrair_endereco(texto):
    texto = texto.lower()

    padrao = r"(rua|av|avenida)\s+[^\n,]+"
    match = re.search(padrao, texto)

    if match:
        return match.group(0)

    return "Não informado"
