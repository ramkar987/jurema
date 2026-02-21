# ⚖️ LAW RAG System — Demo Jurídico com IA

Sistema interativo de consulta em documentos jurídicos usando RAG
(Retrieval-Augmented Generation). Faça perguntas em linguagem natural
sobre PDFs, DOCX e TXT e receba respostas com **citações exatas das fontes**.

🌐 **Demo ao vivo:** [jurema.streamlit.app](https://jurema.streamlit.app)
📂 **Repositório:** [github.com/ramkar987/jurema](https://github.com/ramkar987/jurema)

---

## ✨ Funcionalidades

- 📤 Upload de PDFs, DOCX e TXT
- 🔍 Busca semântica com FAISS (vetores locais)
- 🤖 Respostas geradas por Llama 3 / Mixtral via Groq
- 📎 **Citações automáticas das fontes** (arquivo + trecho + score)
- ⏱️ Tempo de resposta + score de confiança em cada consulta
- 📜 Histórico da sessão clicável
- ⚙️ Configurações: modelo, temperatura, max tokens, trechos

---

## 🏗️ Arquitetura

```
PDF / DOCX / TXT
      ↓
DocumentProcessor (pypdf / python-docx)
      ↓
RecursiveCharacterTextSplitter (chunks de 1000 chars)
      ↓
FastEmbed ONNX — BAAI/bge-small-en-v1.5 (embeddings locais)
      ↓
FAISS VectorStore (busca semântica)
      ↓
Groq API — Llama 3.3 70B (geração da resposta)
      ↓
Streamlit UI com citações + métricas
```

---

## 📦 Stack

| Componente | Tecnologia | Custo |
|---|---|---|
| Interface | Streamlit | Grátis |
| LLM | Groq API (Llama 3 / Mixtral) | Grátis |
| Embeddings | FastEmbed ONNX (local) | Grátis |
| Vector Store | FAISS (local) | Grátis |
| Orquestração | LangChain 0.3 LCEL | Grátis |
| PDF Parser | pypdf | Grátis |
| Deploy | Streamlit Cloud | Grátis |

> 💡 **MVP 100% gratuito** — sem cartão de crédito necessário.

---

## 🚀 Instalação Local

### Pré-requisitos
- Python 3.10+
- Conta Groq (grátis): [console.groq.com](https://console.groq.com)

```bash
# 1. Clonar o repositório
git clone https://github.com/ramkar987/jurema
cd jurema

# 2. Criar ambiente virtual
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Configurar a API Key
cp .streamlit/secrets.toml.template .streamlit/secrets.toml
# Edite o arquivo e adicione sua GROQ_API_KEY

# 5. Rodar o app
streamlit run app.py
```

---

## ☁️ Deploy no Streamlit Cloud

1. Faça um **fork** deste repositório
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Clique em **New app** → selecione o repositório `jurema`
4. Main file path: `app.py`
5. Clique em **Advanced settings → Secrets** e adicione:

```toml
GROQ_API_KEY = "gsk_SuaChaveAqui"
```

6. Clique em **Deploy** — URL gerada automaticamente 🎉

> ⏳ Na primeira execução, o FastEmbed baixa o modelo de embeddings (~60MB).
> Isso leva ~30s apenas uma vez; depois fica em cache.

---

## 📁 Estrutura do Projeto

```
jurema/
├── app.py                        # App principal Streamlit
├── requirements.txt              # Dependências
├── README.md
├── .gitignore
├── .streamlit/
│   ├── config.toml               # Tema e configurações
│   └── secrets.toml.template     # Template da API Key
├── src/
│   ├── __init__.py
│   ├── rag_engine.py             # Motor RAG (Groq + FAISS)
│   ├── document_processor.py     # Leitura de PDF/DOCX/TXT
│   └── utils.py                  # Helpers
└── sample_documents/             # Documentos jurídicos de exemplo
    ├── contrato_venda_2024.txt
    ├── memo_legal_ip.txt
    └── case_law_summary.txt
```

---

## 🔑 Obter API Key Groq (Grátis)

1. Acesse [console.groq.com](https://console.groq.com)
2. Crie uma conta (pode usar Google)
3. Vá em **API Keys → Create API Key**
4. Copie a chave (começa com `gsk_`)

---

## 🤖 Modelos Disponíveis

| Modelo | Característica | Ideal para |
|---|---|---|
| `llama-3.3-70b-versatile` | Melhor qualidade | Demos e produção |
| `llama-3.1-8b-instant` | Mais rápido | Testes rápidos |
| `mixtral-8x7b-32768` | Contexto longo | Documentos grandes |

---

## 🗺️ Roadmap

- [x] MVP com upload de documentos
- [x] Busca semântica com FAISS
- [x] Respostas com citações de fonte
- [x] Histórico de sessão
- [x] Deploy no Streamlit Cloud
- [ ] Suporte a múltiplos usuários
- [ ] Histórico persistente (banco de dados)
- [ ] Autenticação por empresa/cliente
- [ ] Modelos privados (Llama local)
- [ ] Suporte a documentos escaneados (OCR)

---

## 👤 Autor

**Antônio Pinheiro** — Porto Alegre, RS
[LinkedIn](https://linkedin.com/in/seuperfil) | [GitHub](https://github.com/ramkar987)

---

## 📄 Licença

MIT License — livre para uso, modificação e distribuição.
