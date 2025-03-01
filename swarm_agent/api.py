from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import os
from dotenv import load_dotenv

from .nft_knowledge_base import NFTKnowledgeBase, NFTMetadata
from .nft_generator import NFTGenerator

load_dotenv()

app = FastAPI(title="SwarmFund NFT Agent API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

kb = NFTKnowledgeBase()

# Create a global variable for the generator
nft_generator = None

@app.on_event("startup")
async def startup_event():
    """Initialize NFTGenerator on startup"""
    global nft_generator
    nft_generator = await NFTGenerator().initialize()

class NFTGenerationRequest(BaseModel):
    collection_name: str
    description: str
    trait_preferences: Optional[List[dict]] = None
    style_preferences: Optional[List[str]] = None

class NFTResponse(BaseModel):
    collection_name: str
    token_id: str
    name: str
    description: str
    image_url: str
    attributes: List[dict]
    transaction_hash: Optional[str] = None

@app.post("/generate-nft")
async def generate_nft(request: NFTGenerationRequest):
    try:
        # First create the collection
        collection_address = await nft_generator.create_collection(
            name=request.collection_name,
            symbol=request.collection_name[:5].upper()  # Use first 5 chars as symbol
        )
        
        # Then generate the NFT in that collection
        nft_data = await nft_generator.generate(
            collection_address=collection_address,
            description=request.description,
            trait_preferences=request.trait_preferences,
            style_preferences=request.style_preferences
        )
        
        # Store in knowledge base
        nft_metadata = NFTMetadata(
            collection_name=request.collection_name,
            token_id=str(nft_data["token_id"]),
            name=nft_data["name"],
            description=nft_data["description"],
            attributes=nft_data["attributes"],
            image_url=nft_data["image_url"],
            created_at=datetime.now().isoformat()
        )
        kb.add_nft(nft_metadata)
        
        return {
            "success": True,
            "data": nft_data,
            "collection_tx": collection_address.get("transaction_hash") if isinstance(collection_address, dict) else None,
            "mint_tx": nft_data.get("transaction_hash"),
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "data": None,
            "collection_tx": None,
            "mint_tx": None,
            "error": str(e)
        }

@app.get("/query-nfts/{query}")
async def query_nfts(query: str, limit: int = 5):
    results = kb.query_knowledge_base(query, n_results=limit)
    return {"results": results}

@app.get("/wallet-address")
async def get_wallet_address():
    """Get the CDP wallet address to fund"""
    if not nft_generator:
        raise HTTPException(status_code=500, detail="NFTGenerator not initialized")
    address = await nft_generator.get_wallet_address()
    print(f"CDP Wallet Address: {address}")  # Print for server logs
    return {"address": address} 