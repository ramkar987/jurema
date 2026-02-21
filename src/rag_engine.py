"""
Motor RAG (Retrieval-Augmented Generation) para consulta jurídica.
Usa LangChain 0.3+, FAISS e OpenAI GPT.
"""

from typing import List, Tuple, Dict, Optional
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Prompt jurídico especializado em português
PROMPT_JURIDICO = """Você é um assistente jurídico especializado e preciso.
Responda à pergunta do usuário usando EXCLUSIVAMENTE os documentos fornecidos como contexto.

DOCUMENTOS DE CONTEXTO:
{context}

REGRAS OBRIGATÓRIAS:
1. Cite SEMPRE o nome do arquivo fonte entre aspas duplas, ex: "contrato_venda_2024.txt"
2. Se houver número de página/trecho, mencione-o
3. Se a informação NÃO estiver nos documentos, diga: "Esta informação não foi encontrada nos documentos carregados."
4. Use linguagem jurídica precisa e formal
5. Seja completo mas objetivo

PERGUNTA DO USUÁRIO: {question}

RESPOSTA FUNDAMENTADA (com citações de fonte):"""


class RAGEngine:
    """
    Motor principal do sistema RAG.
    Gerencia embeddings, vector store FAISS e chamadas ao LLM.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.1,
        max_tokens: int = 1500,
        top_k: int = 3,
    ):
        self.api_key = api_key
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k

        # Embeddings: text-embedding-3-small é mais barato e rápido
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            model="text-embedding-3-small",
        )

        # LLM principal
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self.vector_store: Optional[FAISS] = None
        self.total_chunks: int = 0

    def update_config(
        self,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        top_k: int = None,
    ) -> None:
        """Atualiza configurações e recria o LLM se necessário."""
        changed = False

        if model and model != self.model_name:
            self.model_name = model
            changed = True
        if temperature is not None and temperature != self.temperature:
            self.temperature = temperature
            changed = True
        if max_tokens and max_tokens != self.max_tokens:
            self.max_tokens = max_tokens
            changed = True
        if top_k:
            self.top_k = top_k

        # Recria o LLM apenas se algum parâmetro mudou
        if changed:
            self.llm = ChatOpenAI(
                openai_api_key=self.api_key,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

    def add_documents_from_texts(self, docs: List[Tuple[str, str]]) -> int:
        """
        Indexa documentos no FAISS.

        Args:
            docs: Lista de tuplas (nome_arquivo, texto_extraido)

        Returns:
            Número de chunks criados
        """
        # Splitter: chunks de 1000 chars com 200 de overlap para não perder contexto
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        langchain_docs: List[Document] = []

        for filename, text in docs:
            if not text or not text.strip():
                continue

            chunks = splitter.split_text(text)

            for idx, chunk in enumerate(chunks):
                langchain_docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": filename,
                            "chunk_index": idx,
                            "total_chunks": len(chunks),
                        },
                    )
                )

        if not langchain_docs:
            return 0

        # Adiciona ao vector store existente ou cria novo
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(langchain_docs, self.embeddings)
        else:
            self.vector_store.add_documents(langchain_docs)

        self.total_chunks += len(langchain_docs)
        return len(langchain_docs)

    def query(self, question: str) -> Dict:
        """
        Executa consulta RAG: busca semântica + geração com LLM.

        Returns:
            Dict com: answer, sources, confidence_score, chunks_used
        """
        if self.vector_store is None:
            return {
                "answer": "⚠️ Nenhum documento indexado. Carregue documentos primeiro.",
                "sources": [],
                "confidence_score": 0,
                "chunks_used": 0,
            }

        # Busca semântica com scores de relevância (0-1, sendo 1 = mais relevante)
        try:
            docs_with_scores = self.vector_store.similarity_search_with_relevance_scores(
                question, k=self.top_k
            )
        except Exception:
            # Fallback: sem scores
            raw_docs = self.vector_store.similarity_search(question, k=self.top_k)
            docs_with_scores = [(doc, 0.8) for doc in raw_docs]

        if not docs_with_scores:
            return {
                "answer": "Não encontrei informações relevantes nos documentos carregados.",
                "sources": [],
                "confidence_score": 0,
                "chunks_used": 0,
            }

        # Monta contexto identificando cada fonte
        context_parts = []
        sources = []

        for i, (doc, score) in enumerate(docs_with_scores, 1):
            filename = doc.metadata.get("source", "Documento desconhecido")
            chunk_idx = doc.metadata.get("chunk_index", 0)
            total_chunks = doc.metadata.get("total_chunks", "?")
            confidence_pct = round(score * 100, 1)

            # Contexto passado ao LLM com identificação clara de fonte
            context_parts.append(
                f"[FONTE {i}: arquivo='{filename}', trecho={chunk_idx + 1}/{total_chunks}]\n"
                f"{doc.page_content}"
            )

            sources.append({
                "file": filename,
                "location": f"Trecho {chunk_idx + 1} de {total_chunks}",
                "excerpt": (
                    doc.page_content[:300] + "..."
                    if len(doc.page_content) > 300
                    else doc.page_content
                ),
                "score": confidence_pct,
            })

        context = "\n\n---\n\n".join(context_parts)
        avg_confidence = sum(s["score"] for s in sources) / len(sources)

        # Prompt + LLM + Parser em cadeia LCEL
        prompt = ChatPromptTemplate.from_template(PROMPT_JURIDICO)
        chain = prompt | self.llm | StrOutputParser()

        answer = chain.invoke({"context": context, "question": question})

        return {
            "answer": answer,
            "sources": sources,
            "confidence_score": round(avg_confidence, 1),
            "chunks_used": len(docs_with_scores),
        }

    def clear(self) -> None:
        """Limpa o vector store e reseta o contador de chunks."""
        self.vector_store = None
        self.total_chunks = 0
