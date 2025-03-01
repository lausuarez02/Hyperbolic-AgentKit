from typing import List, Dict
import chromadb
from datetime import datetime
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import os
from utils import print_system, print_error

class NFTMetadata(BaseModel):
    collection_name: str
    token_id: str
    name: str
    description: str
    attributes: List[Dict[str, str]]
    image_url: str
    created_at: str

class NFTKnowledgeBase:
    def __init__(self, collection_name: str = "nft_knowledge"):
        print_system("Initializing NFTKnowledgeBase...")
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
        os.makedirs(data_dir, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        
        class EmbeddingFunction:
            def __init__(self, model):
                self.model = model
            
            def __call__(self, input: List[str]) -> List[List[float]]:
                embeddings = self.model.encode(input)
                return embeddings.tolist()
        
        embedding_func = EmbeddingFunction(self.embedding_model)
        
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_func
            )
        except Exception as e:
            print_error(f"Error initializing collection: {e}")
            raise

    def add_nft(self, nft: NFTMetadata):
        """Add NFT metadata to the knowledge base."""
        document = f"{nft.name}\n{nft.description}\n{str(nft.attributes)}"
        
        try:
            self.collection.add(
                documents=[document],
                ids=[f"{nft.collection_name}_{nft.token_id}"],
                metadatas=[{
                    "collection_name": nft.collection_name,
                    "token_id": nft.token_id,
                    "name": nft.name,
                    "image_url": nft.image_url,
                    "created_at": nft.created_at
                }]
            )
        except Exception as e:
            print_error(f"Error adding NFT: {e}")
            raise

    def query_knowledge_base(self, query: str, n_results: int = 5) -> List[Dict]:
        """Query the knowledge base for relevant NFTs."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            if not results['documents'][0]:
                return []
                
            formatted_results = []
            for doc, metadata, distance in zip(
                results['documents'][0], 
                results['metadatas'][0],
                results['distances'][0]
            ):
                formatted_results.append({
                    "content": doc,
                    "metadata": metadata,
                    "relevance_score": 1 - distance
                })
            
            return formatted_results
            
        except Exception as e:
            print_error(f"Error querying knowledge base: {e}")
            return [] 