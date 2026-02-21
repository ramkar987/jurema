# ⚖️ LAW RAG Demo — Sistema de Consulta Jurídica com IA

Demo interativo de um sistema RAG (Retrieval-Augmented Generation) aplicado
ao contexto jurídico, desenvolvido com Streamlit + LangChain + FAISS + OpenAI.

🌐 **Demo ao vivo:** [jurema.streamlit.app](https://jurema.streamlit.app)

## ✨ Funcionalidades

- 📤 Upload de PDFs, DOCX e TXT
- 🔍 Busca semântica com FAISS
- 🤖 Respostas geradas por GPT-3.5 / GPT-4
- 📎 **Citações automáticas das fontes** (arquivo + trecho)
- 📊 Score de confiança e tempo de resposta
- 📜 Histórico da sessão clicável
- ⚙️ Configurações: modelo, temperatura, max tokens

## 🚀 Instalação Local

```bash
git clone https://github.com/seuusuario/jurema
cd jurema
pip install -r requirements.txt

# Configurar API Key
cp .streamlit/secrets.toml.template .streamlit/secrets.toml
# Edite o arquivo e adicione sua OPENAI_API_KEY

streamlit run app.py
```

## ☁️ Deploy Streamlit Cloud

1. Fork este repositório
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Conecte ao GitHub, selecione `jurema`
4. Em **Advanced Settings → Secrets**, adicione:
   ```toml
   OPENAI_API_KEY = "sk-sua-chave-aqui"
   ```
5. Clique em **Deploy** — pronto! 🎉

## 🏗️ Arquitetura

```
PDF/DOCX/TXT → PyPDF/docx → RecursiveTextSplitter (chunks 1000 chars)
    → OpenAI Embeddings → FAISS VectorStore
    → Semantic Search → LangChain LCEL Chain → GPT Response
    → Streamlit UI com citações
```

## 📦 Stack

| Componente | Tecnologia |
|---|---|
| UI | Streamlit |
| LLM | OpenAI GPT-3.5/4 |
| Embeddings | text-embedding-3-small |
| Vector Store | FAISS (local) |
| Orquestração | LangChain 0.3 LCEL |
| PDF Parser | pypdf |

## 👤 Autor

**Antônio Pinheiro** — Porto Alegre, RS  
[LinkedIn](https://linkedin.com/in/seuperfil) | [GitHub](https://github.com/seuusuario)
