"""
LAW RAG Demo — Sistema de Consulta Jurídica com IA
Autor: Antônio Pinheiro
Stack: Streamlit + LangChain 0.3 + FAISS + OpenAI GPT
Deploy: https://share.streamlit.io
"""

import time
from pathlib import Path

import streamlit as st

from src.document_processor import DocumentProcessor
from src.rag_engine import RAGEngine
from src.utils import format_confidence, format_elapsed, truncar

# ─────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA
# ─────────────────────────────────────────
st.set_page_config(
    page_title="LAW RAG Demo — Consulta Jurídica com IA",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado para visual profissional
st.markdown(
    """
    <style>
        /* Cabeçalho */
        .main-header {
            font-size: 2rem;
            font-weight: 700;
            color: #1E3A5F;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            color: #555;
            font-size: 1rem;
            margin-bottom: 1rem;
        }
        /* Box de resposta */
        .answer-box {
            background: linear-gradient(135deg, #f0f7ff 0%, #e8f4fd 100%);
            border-left: 5px solid #1E3A5F;
            padding: 1.2rem 1.5rem;
            border-radius: 8px;
            font-size: 1rem;
            line-height: 1.7;
            margin: 0.5rem 0;
        }
        /* Badge de confiança */
        .badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        /* Botão primário */
        .stButton > button[kind="primary"] {
            background-color: #1E3A5F !important;
            color: white !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────
# INICIALIZAÇÃO DO SESSION STATE
# ─────────────────────────────────────────
defaults = {
    "rag_engine": None,
    "document_list": [],      # lista de dicts {name, type, chunks}
    "search_history": [],     # últimas 10 perguntas
    "last_result": None,      # último resultado de query
    "prefill_query": "",      # pergunta do histórico clicada
    "api_key_ok": False,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ LAW RAG System")
    st.markdown("*Consulta jurídica inteligente com IA*")
    st.divider()

    # ── API KEY ──────────────────────────
    st.subheader("🔑 API Key")
    api_key = ""

# Tenta Streamlit Cloud Secrets primeiro
try:
    api_key = st.secrets["GROQ_API_KEY"]   # ← era OPENAI_API_KEY
    st.success("✅ API Key via Secrets")
except Exception:
    api_key = st.text_input(
        "Groq API Key:",                    # ← era "OpenAI API Key"
        type="password",
        placeholder="gsk_...",             # ← era "sk-..."
        help="Obtenha grátis em console.groq.com",
    )


    # Inicializa ou reinicializa o motor RAG quando a key muda
    if api_key and api_key != st.session_state.get("_last_api_key", ""):
        try:
            st.session_state.rag_engine = RAGEngine(api_key=api_key)
            st.session_state.api_key_ok = True
            st.session_state["_last_api_key"] = api_key
        except Exception as e:
            st.error(f"Erro ao inicializar motor: {e}")
            st.session_state.api_key_ok = False

    if not api_key:
        st.info("ℹ️ Insira sua API Key para habilitar o sistema")

    st.divider()

    # ── CONFIGURAÇÕES AVANÇADAS ──────────
    with st.expander("⚙️ Configurações Avançadas"):
modelo = st.selectbox(
    "Modelo LLM:",
    [
        "llama-3.3-70b-versatile",   # melhor qualidade
        "llama-3.1-8b-instant",       # mais rápido
        "mixtral-8x7b-32768",         # contexto longo
    ],
    index=0,
    help="Todos gratuitos no Groq 🆓",
)

        temperatura = st.slider(
            "Temperatura:", 0.0, 1.0, 0.1, 0.05,
            help="0 = determinístico | 1 = criativo"
        )
        max_tokens = st.slider("Max Tokens:", 200, 4000, 1500, 100)
        top_k = st.slider(
            "Trechos para recuperar:", 1, 5, 3,
            help="Quantos trechos do documento usar como contexto"
        )

        if st.session_state.rag_engine and st.button("✅ Aplicar"):
            st.session_state.rag_engine.update_config(
                model=modelo,
                temperature=temperatura,
                max_tokens=max_tokens,
                top_k=top_k,
            )
            st.success("Configurações aplicadas!")

    st.divider()

    # ── HISTÓRICO ────────────────────────
    st.subheader("📜 Histórico da Sessão")

    if st.session_state.search_history:
        historico_rev = list(reversed(st.session_state.search_history[-10:]))
        for i, pergunta in enumerate(historico_rev):
            label = truncar(pergunta, 38)
            if st.button(f"↩ {label}", key=f"hist_{i}", use_container_width=True):
                st.session_state.prefill_query = pergunta
                st.rerun()

        if st.button("🗑️ Limpar histórico", use_container_width=True):
            st.session_state.search_history = []
            st.rerun()
    else:
        st.caption("Nenhuma pesquisa realizada ainda.")

    st.divider()

    # ── LINKS ────────────────────────────
    st.markdown(
        "🔗 [GitHub](https://github.com/seuusuario/law-rag-demo)  |  "
        "[LinkedIn](https://linkedin.com/in/seuperfil)"
    )
    st.caption("Desenvolvido por **Antônio Pinheiro**")

    # ── INSTRUÇÕES ───────────────────────
    with st.expander("❓ Como usar"):
        st.markdown(
            """
            1. **Configure** sua OpenAI API Key acima
            2. **Carregue** documentos (PDF, DOCX, TXT) **ou** use os exemplos demo
            3. **Digite** sua pergunta no campo à direita
            4. **Pressione** *Pesquisar* e aguarde
            5. Veja a **resposta com citações** das fontes
            """
        )


# ─────────────────────────────────────────
# CONTEÚDO PRINCIPAL
# ─────────────────────────────────────────
st.markdown(
    '<h1 class="main-header">⚖️ LAW RAG System — Demo Jurídico</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-header">Consulta inteligente em documentos jurídicos • '
    'Powered by GPT + FAISS + LangChain</p>',
    unsafe_allow_html=True,
)
st.divider()

col_docs, col_query = st.columns([1, 1], gap="large")

# ─────────────────────────────────────────
# COLUNA ESQUERDA: DOCUMENTOS
# ─────────────────────────────────────────
with col_docs:
    st.subheader("📂 Documentos")

    btn_col1, btn_col2 = st.columns(2)

    # Botão: carregar documentos demo
    with btn_col1:
        demo_btn = st.button(
            "📁 Carregar Demo",
            use_container_width=True,
            help="Carrega 3 documentos jurídicos de exemplo",
        )

    # Botão: limpar tudo
    with btn_col2:
        clear_btn = st.button(
            "🗑️ Limpar Tudo",
            use_container_width=True,
        )

    if demo_btn:
        if not st.session_state.rag_engine:
            st.error("⚠️ Configure a API Key primeiro!")
        else:
            demo_dir = Path("sample_documents")
            if not demo_dir.exists():
                st.error("Pasta `sample_documents/` não encontrada.")
            else:
                processor = DocumentProcessor()
                novos = []
                nomes_existentes = {d["name"] for d in st.session_state.document_list}

                for arq in sorted(demo_dir.glob("*")):
                    if arq.suffix.lower() in [".pdf", ".txt", ".docx"]:
                        if arq.name not in nomes_existentes:
                            texto = processor.load_file(str(arq))
                            if texto:
                                novos.append((arq.name, texto))

                if not novos:
                    st.info("Todos os documentos demo já estão carregados.")
                else:
                    with st.spinner(f"Indexando {len(novos)} documento(s)..."):
                        chunks = st.session_state.rag_engine.add_documents_from_texts(novos)
                        for nome, _ in novos:
                            st.session_state.document_list.append(
                                {"name": nome, "type": "demo", "chunks": chunks // len(novos)}
                            )
                    st.success(f"✅ {len(novos)} documentos demo carregados! ({chunks} chunks)")

    if clear_btn and st.session_state.rag_engine:
        st.session_state.rag_engine.clear()
        st.session_state.document_list = []
        st.session_state.last_result = None
        st.success("✅ Todos os documentos foram removidos.")
        st.rerun()

    # Upload de arquivos
    arquivos = st.file_uploader(
        "📎 Carregar seus documentos:",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Arraste ou clique para selecionar arquivos",
    )

    if arquivos and st.session_state.rag_engine:
        nomes_existentes = {d["name"] for d in st.session_state.document_list}
        novos_uploads = [f for f in arquivos if f.name not in nomes_existentes]

        if novos_uploads:
            processor = DocumentProcessor()
            docs_para_indexar = []

            for arq in novos_uploads:
                tmp = f"/tmp/{arq.name}"
                with open(tmp, "wb") as f:
                    f.write(arq.getbuffer())
                texto = processor.load_file(tmp)
                if texto:
                    docs_para_indexar.append((arq.name, texto))

            if docs_para_indexar:
                with st.spinner(f"Processando {len(docs_para_indexar)} arquivo(s)..."):
                    total_chunks = st.session_state.rag_engine.add_documents_from_texts(
                        docs_para_indexar
                    )
                    for nome, _ in docs_para_indexar:
                        st.session_state.document_list.append(
                            {"name": nome, "type": "upload", "chunks": total_chunks // len(docs_para_indexar)}
                        )
                st.success(f"✅ {len(docs_para_indexar)} arquivo(s) indexado(s)!")

    # Painel de documentos indexados
    if st.session_state.document_list:
        total_docs = len(st.session_state.document_list)
        total_chunks = (
            st.session_state.rag_engine.total_chunks
            if st.session_state.rag_engine
            else 0
        )

        m1, m2 = st.columns(2)
        m1.metric("📄 Documentos", total_docs)
        m2.metric("🔢 Chunks", total_chunks)

        st.markdown("**Documentos em memória:**")
        for doc in st.session_state.document_list:
            icone = "📋" if doc["type"] == "demo" else "📄"
            st.markdown(f"{icone} `{doc['name']}`")
    else:
        st.info(
            "Nenhum documento carregado.\n\n"
            "Use **Carregar Demo** para testar com exemplos jurídicos."
        )


# ─────────────────────────────────────────
# COLUNA DIREITA: PERGUNTAS
# ─────────────────────────────────────────
with col_query:
    st.subheader("🔍 Faça uma Pergunta")

    # Preenche campo a partir do histórico
    valor_inicial = st.session_state.prefill_query
    if valor_inicial:
        st.session_state.prefill_query = ""

    pergunta = st.text_area(
        "Sua pergunta sobre os documentos:",
        value=valor_inicial,
        placeholder=(
            "Ex: Quais são as obrigações do comprador no contrato?\n"
            "Ex: O que diz o memorando sobre propriedade intelectual?"
        ),
        height=110,
    )

    pesquisar = st.button("🔍 Pesquisar", type="primary", use_container_width=True)

    # Exemplos rápidos de perguntas
    with st.expander("💡 Perguntas de exemplo"):
        exemplos = [
            "Quais são os direitos e obrigações das partes no contrato?",
            "Qual o prazo de vigência do contrato?",
            "O que o memorando diz sobre propriedade intelectual?",
            "Quais são as penalidades por descumprimento?",
            "Como é resolvida uma disputa entre as partes?",
        ]
        for ex in exemplos:
            if st.button(ex, key=f"ex_{ex[:20]}", use_container_width=True):
                st.session_state.prefill_query = ex
                st.rerun()

    # Executa a pesquisa
    if pesquisar and pergunta.strip():
        if not st.session_state.rag_engine:
            st.error("⚠️ Configure a API Key primeiro!")
        elif not st.session_state.document_list:
            st.error("⚠️ Carregue documentos antes de pesquisar!")
        else:
            with st.spinner("🤔 Consultando documentos e gerando resposta..."):
                t0 = time.time()
                try:
                    resultado = st.session_state.rag_engine.query(pergunta.strip())
                    resultado["elapsed"] = time.time() - t0

                    # Adiciona ao histórico (sem duplicatas)
                    if pergunta not in st.session_state.search_history:
                        st.session_state.search_history.append(pergunta)

                    st.session_state.last_result = resultado

                except Exception as e:
                    st.error(f"❌ Erro ao processar: {str(e)}")
                    st.info("Verifique sua API Key e tente novamente.")


# ─────────────────────────────────────────
# ÁREA DE RESULTADOS (largura total)
# ─────────────────────────────────────────
if st.session_state.last_result:
    st.divider()
    res = st.session_state.last_result

    st.subheader("✅ Resposta")

    # Box principal com a resposta
    st.markdown(
        f'<div class="answer-box">{res["answer"]}</div>',
        unsafe_allow_html=True,
    )

    # Métricas de desempenho
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("⏱️ Tempo", format_elapsed(res.get("elapsed", 0)))
    mc2.metric("🎯 Confiança", f"{res['confidence_score']:.0f}%")
    mc3.metric("📊 Trechos usados", res["chunks_used"])
    mc4.metric("🤖 Modelo", 
    st.session_state.rag_engine.model_name.split("-")[0].upper()   # ex: "LLAMA"
    if st.session_state.rag_engine else "-"
)


    # Fontes utilizadas
    if res.get("sources"):
        st.subheader("📎 Fontes Utilizadas")
        for i, fonte in enumerate(res["sources"], 1):
            confianca = format_confidence(fonte["score"])
            with st.expander(
                f"[{i}] 📄 {fonte['file']} — {fonte['location']} | {confianca}"
            ):
                st.markdown("**Trecho recuperado:**")
                st.markdown(f"> {fonte['excerpt']}")
                st.caption(f"Score de relevância: {fonte['score']:.1f}%")
