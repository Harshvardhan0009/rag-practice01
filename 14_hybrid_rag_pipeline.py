import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

# 1. Load environment variables
load_dotenv()

print("=== 🔀 Hybrid Search (Dense Vector + BM25 Keyword) RAG Pipeline ===\n")

# 2. Custom Hybrid Retriever with Reciprocal Rank Fusion (RRF)
class HybridRetriever:
    def __init__(self, vector_retriever, bm25_retriever, weights=[0.5, 0.5], top_k=5):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.weights = weights
        self.top_k = top_k

    def invoke(self, query: str) -> List[Document]:
        # Retrieve candidates from both dense vector and sparse BM25
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)
        
        # Combine using Reciprocal Rank Fusion (RRF) algorithm
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        c = 60  # RRF constant smoothing factor
        
        for rank, doc in enumerate(vector_docs):
            content = doc.page_content
            doc_map[content] = doc
            rrf_scores[content] = rrf_scores.get(content, 0.0) + self.weights[0] * (1.0 / (c + rank + 1))
            
        for rank, doc in enumerate(bm25_docs):
            content = doc.page_content
            doc_map[content] = doc
            rrf_scores[content] = rrf_scores.get(content, 0.0) + self.weights[1] * (1.0 / (c + rank + 1))
            
        # Sort documents by total RRF score
        sorted_contents = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return [doc_map[content] for content in sorted_contents[:self.top_k]]

# 3. Setup paths and models
docs_path = "docs"
persistent_directory = "db/chroma_db"

embedding_model = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2", temperature=0)

# 4. Load persistent Chroma Vector Database (Dense Retriever)
print("📦 Loading ChromaDB vector store (Dense Retriever)...")
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)
vector_retriever = db.as_retriever(search_kwargs={"k": 5})

# 5. Create BM25 Keyword Retriever (Sparse Retriever)
print("🔍 Building BM25 index on documents (Sparse Retriever)...")
loader = DirectoryLoader(path=docs_path, glob="*.txt", loader_cls=TextLoader)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks = text_splitter.split_documents(documents)

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5

# 6. Combine Vector & BM25 into HybridRetriever
print("⚡ Combining Vector (Dense) + BM25 (Sparse) into HybridRetriever...")
hybrid_retriever = HybridRetriever(
    vector_retriever=vector_retriever,
    bm25_retriever=bm25_retriever,
    weights=[0.5, 0.5],
    top_k=5
)

# 7. Execute Query
query = "How much did Microsoft pay to acquire GitHub?"
print(f"\n❓ User Query: '{query}'\n")

print("--- 🔍 Retrieving top 5 relevant documents via Hybrid Search ---")
relevant_docs = hybrid_retriever.invoke(query)

for i, doc in enumerate(relevant_docs, 1):
    source = doc.metadata.get("source", "Unknown")
    print(f"\nDocument {i} (Source: {source}):")
    print("-" * 50)
    print(doc.page_content.strip())
    print("-" * 50)

# 8. Generate Answer using LLM
combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, concise answer using only the information from these documents.
"""

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

print("\n🤖 Generating answer with local Llama 3.2...")
result = llm.invoke(messages)

print("\n--- 💡 Generated Answer ---")
print(result.content)
