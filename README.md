# 🍔 Lanchonete IA

Sistema inteligente de pedidos para lanchonete utilizando Processamento de Linguagem Natural (PLN) e modelo de IA baseado em DistilBERT.

---

# 🚀 Funcionalidades

- Registro de pedidos em linguagem natural
- Identificação automática de produtos
- Cálculo do valor total do pedido
- Previsão da forma de pagamento (PIX, DINHEIRO, CRÉDITO, DÉBITO) usando IA
- Extração de endereço de entrega

---

# 🧠 Tecnologias utilizadas

- Python 3.11+
- scikit-learn
- PyTorch
- Transformers (Hugging Face)
- DistilBERT
- NLP (Processamento de Linguagem Natural)

---

# 📦 Instalação

## 1. Clonar o repositório

```bash
git clone git@github.com:SEU_USUARIO/lanchonete-ia.git
cd lanchonete-ia
```

## 2. Criar ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate
```

## 3. Instalar dependências

```bash
pip install -r requirements.txt
```

---

# 🧠 Primeira execução

```bash
python
```

```python
from model.salvar_modelo import salvar_modelo
salvar_modelo()
```

# ▶ Executar sistema

```bash
python main.py
```
