import os
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.api.types import EmbeddingFunction
from sentence_transformers import SentenceTransformer
import tiktoken



class STEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            input,
            normalize_embeddings=True
        )
        return embeddings.tolist() 


class RAGTool:

    def __init__(
        self,
        persist_dir: str = "./data/chroma_db",
        collection_name: str = "safar_docs",
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):


        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)


        self.client = chromadb.PersistentClient(path=self.persist_dir)


        self.embedding_model_name = embedding_model_name


        self.tokenizer = tiktoken.get_encoding("cl100k_base")


        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap


        self.collection_name = collection_name
        self.embedding_fn = STEmbeddingFunction(embedding_model_name)

        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception:
            
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn
            )


    def _chunk_text(self, text: str) -> List[str]:
        tokens = self.tokenizer.encode(text)
        chunks = []

        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

        return chunks
 
    def ingest_folder(self, folder: str, extensions: Optional[List[str]] = None) -> Dict[str, Any]:

        if extensions is None:
            extensions = [".txt", ".md"]

        folder_path = Path(folder)
        if not folder_path.exists():
            return {"error": "folder_not_found", "folder": folder}

        added = 0
        skipped = []

        for file_path in folder_path.rglob("*"):
            if file_path.suffix.lower() not in extensions:
                continue

            try:
                text = file_path.read_text(encoding="utf-8").strip()
                if not text:
                    skipped.append(str(file_path))
                    continue

                chunks = self._chunk_text(text)

                ids, documents, metadatas = [], [], []

                for idx, chunk in enumerate(chunks):
                    chunk_id = f"{file_path.name}_chunk_{idx}_{uuid.uuid4().hex[:8]}"
                    ids.append(chunk_id)
                    documents.append(chunk)
                    metadatas.append({
                        "source": file_path.name,
                        "path": str(file_path),
                        "chunk_index": idx,
                        "total_chunks": len(chunks)
                    })

 
                embeddings = self.embedding_fn(documents)

                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings
                )

                added += len(ids)

            except Exception:
                skipped.append(str(file_path))

        try:
            self.client.persist()
        except Exception:
            pass

        return {"added_chunks": added, "skipped_files": skipped}
 
    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:

        if not query_text:
            return []

        query_embedding = self.embedding_fn([query_text])
        res = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        ) 
        results = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        distances = res.get("distances", [[]])[0]
        ids = res.get("ids", [[]])[0]

        for i in range(len(docs)):
            results.append({
                "id": ids[i],
                "document": docs[i],
                "metadata": metas[i],
                "distance": distances[i] if i < len(distances) else None
            })

        return results
 
    def clear_collection(self):
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )
 
    def get_stats(self) -> Dict[str, Any]:
        try:
            count = self.collection.count()
        except Exception:
            count = None

        return {"name": self.collection_name, "count": count}
