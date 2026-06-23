import os
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class DocumentLoader:
    def load(self, directory: str) -> List[str]:
        documents = []

        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                filepath = os.path.join(directory, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    documents.append(f.read())

        return documents


class TextSplitter:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, documents: List[str]) -> List[str]:
        chunks = []

        for doc in documents:
            doc_chunks = self._split_single(doc)
            chunks.extend(doc_chunks)

        return chunks

    def _split_single(self, text: str) -> List[str]:
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.overlap

        return chunks


class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return np.array(embeddings)


class VectorStore:
    def __init__(self, embedding_dim: int = 384):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.chunks: List[str] = []

    def add(self, embeddings: np.ndarray, chunks: List[str]):
        self.index.add(embeddings) 
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[Tuple[str, float]]: # list of (text,dist)
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(dist)))

        return results


class MedicalRAG:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.loader = DocumentLoader()
        self.splitter = TextSplitter()
        self.embedder = EmbeddingModel(model_name)
        self.vector_store = None
        self.embedding_dim = 384

    def build_index(self, documents_dir: str):
        print(f"Loading documents from {documents_dir}...")
        documents = self.loader.load(documents_dir)
        print(f"Loaded {len(documents)} documents")

        print("Splitting into chunks...")
        chunks = self.splitter.split(documents)
        print(f"Created {len(chunks)} chunks")

        print("Generating embeddings...")
        embeddings = self.embedder.embed(chunks)

        print("Building FAISS index...")
        self.vector_store = VectorStore(self.embedding_dim)
        self.vector_store.add(embeddings, chunks)
        print("Index built successfully!")

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        if self.vector_store is None:
            raise ValueError("Index not built. Call build_index() first.")

        query_embedding = self.embedder.embed([query])
        results = self.vector_store.search(query_embedding, k)

        return [chunk for chunk, distance in results]

    def build_prompt(self, query: str, chunks: List[str]) -> str:
        context = "\n\n".join(chunks)

        prompt = f"""Based on the following medical context, answer the question.
                Context: {context}
                Question: {query}
                Answer:"""

        return prompt
