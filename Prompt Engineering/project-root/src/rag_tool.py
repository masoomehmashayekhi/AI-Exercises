
import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path


try:
    import chromadb
    from chromadb.utils import embedding_functions
except Exception as e:
    chromadb = None
    embedding_functions = None


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = max(end - overlap, end)
    return chunks

class RAGTool:
    

    def __init__(
        self,
        persist_dir: str = "chroma_persist",
        collection_name: str = "safar_docs",
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ):
        if chromadb is None or embedding_functions is None:
            raise RuntimeError(
                "chromadb or chromadb.utils.embedding_functions not available. "
                "Install chromadb and sentence-transformers: pip install chromadb sentence-transformers"
            )

        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap


        self._emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.model_name
        )


        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.Client(
            chromadb.config.Settings(chroma_db_impl="duckdb+parquet", persist_directory=self.persist_dir)
        )


        try:
            self.collection = self.client.get_collection(self.collection_name)
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name, embedding_function=self._emb_fn
            )

    def ingest_folder(self, folder: str = "data", extensions: Optional[List[str]] = None) -> Dict[str, Any]:
        
        if extensions is None:
            extensions = [".txt", ".md"]

        folder = Path(folder)
        if not folder.exists():
            return {"error": "folder_not_found", "folder": str(folder)}

        added = 0
        skipped = []
        for file_path in folder.rglob("*"):
            if file_path.suffix.lower() not in extensions:
                continue
            try:
                text = file_path.read_text(encoding="utf-8")
                
                text = text.strip()
                if not text:
                    skipped.append(str(file_path))
                    continue
                chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
                ids = []
                documents = []
                metadatas = []
                for i, c in enumerate(chunks):
                    doc_id = f"{file_path.name}::{i}"
                    ids.append(doc_id)
                    documents.append(c)
                    metadatas.append({"source": file_path.name, "chunk_index": i, "path": str(file_path)})
                    
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                added += len(ids)
            except Exception as e:
                skipped.append(str(file_path))
                
        try:
            self.client.persist()
        except Exception:
            pass

        return {"added_chunks": added, "skipped_files": skipped}

    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        
        if not query_text:
            return []

        res = self.collection.query(query_texts=[query_text], n_results=top_k, include=["documents", "metadatas", "distances", "ids"])
        
        results = []
        try:
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
        except Exception:
            
            pass

        return results

    def clear_collection(self):
        """Remove all items from collection (useful for dev)."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        
        self.collection = self.client.create_collection(name=self.collection_name, embedding_function=self._emb_fn)

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return simple stats about the collection."""
        try:
            return {
                "name": self.collection_name,
                "count": self.collection.count()
            }
        except Exception:
            return {"name": self.collection_name, "count": None}
