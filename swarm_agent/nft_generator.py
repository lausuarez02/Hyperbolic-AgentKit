from typing import List, Dict, Optional
import os
# from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
import json
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import RunnableConfig
from langchain_community.tools import Tool

# Import CDP components
from coinbase_agentkit import (
    AgentKit,
    AgentKitConfig,
    CdpWalletProvider,
    CdpWalletProviderConfig,
    cdp_api_action_provider,
    cdp_wallet_action_provider,
    erc20_action_provider
)
from coinbase_agentkit_langchain import get_langchain_tools

load_dotenv()

# Disable tracing explicitly
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""

class NFTGenerator:
    def __init__(self):
        # Disable tracing explicitly
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        
        # Initialize CDP wallet provider
        wallet_data = None
        if os.path.exists("wallet_data.txt"):
            with open("wallet_data.txt") as f:
                wallet_data = f.read()

        self.wallet_provider = CdpWalletProvider(CdpWalletProviderConfig(
            api_key_name=os.getenv("CDP_API_KEY_NAME"),
            api_key_private=os.getenv("CDP_API_KEY_PRIVATE"),
            network_id="base-sepolia",
            wallet_data=wallet_data
        ))

        # Initialize AgentKit with all needed providers
        self.agent_kit = AgentKit(AgentKitConfig(
            wallet_provider=self.wallet_provider,
            action_providers=[
                cdp_api_action_provider(),
                cdp_wallet_action_provider(),
                erc20_action_provider()
            ]
        ))

        # Get CDP tools
        self.tools = get_langchain_tools(self.agent_kit)

        # Create ReAct agent
        self.agent_executor = create_react_agent(
            self.llm,
            tools=self.tools
        )

        # Save wallet data if it's new
        if not wallet_data:
            wallet_data = json.dumps(self.wallet_provider.export_wallet().to_dict())
            with open("wallet_data.txt", "w") as f:
                f.write(wallet_data)

    async def create_collection(self, name: str, symbol: str) -> str:
        """Create a new NFT collection using the factory"""
        try:
            # Create the prompt for the agent
            prompt = f"""Create a new NFT collection using the factory contract at {os.getenv('NFT_FACTORY_ADDRESS')}
            with the following parameters:
            - Name: {name}
            - Symbol: {symbol}
            - Base URI: ipfs://  # This will be overridden by our on-chain metadata
            
            Use the contract_call tool to call the createCollection function with these exact parameters.
            Return only the deployed collection address."""

            # Run the agent
            result = await self.agent_executor.ainvoke(
                {"messages": [HumanMessage(content=prompt)]},
                RunnableConfig(
                    configurable={
                        "thread_id": "nft_generator",
                        "checkpoint_ns": "create_collection",
                    }
                )
            )
            
            # Extract collection address from result
            if isinstance(result, dict):
                if 'output' in result:
                    return result['output']
                elif 'messages' in result:
                    # Get the last non-error message
                    for msg in reversed(result['messages']):
                        if isinstance(msg, AIMessage) and msg.content:
                            return msg.content.strip()
            elif isinstance(result, str):
                return result
            
            # Print result for debugging
            print("Agent result:", result)
            raise Exception(f"Unexpected result format: {result}")

        except Exception as e:
            raise Exception(f"Failed to create collection: {str(e)}")

    async def generate(
        self,
        collection_address: str,
        description: str,
        trait_preferences: Optional[List[dict]] = None,
        style_preferences: Optional[List[str]] = None,
        reference_nfts: Optional[List[Dict]] = None
    ):
        """Generate and mint a new NFT with the given parameters.
        
        Args:
            collection_address: Address of the NFT collection
            description: Description of the NFT to generate
            trait_preferences: List of preferred traits and their probabilities
            style_preferences: List of artistic style preferences
            reference_nfts: List of reference NFTs to inspire the generation
            
        Returns:
            Dict containing the NFT metadata and transaction details
        """
        try:
            # Generate NFT metadata using LLM
            generation_prompt = self._create_generation_prompt(
                description, 
                trait_preferences, 
                style_preferences, 
                reference_nfts
            )
            
            metadata_result = await self.llm.ainvoke(
                [HumanMessage(content=generation_prompt)]
            )
            
            try:
                nft_data = json.loads(metadata_result.content)
            except json.JSONDecodeError:
                raise Exception("Failed to generate valid NFT metadata")

            # Generate image using the metadata
            image_url = await self._generate_image(nft_data)
            nft_data["image"] = image_url
            
            # Create the prompt for minting
            mint_prompt = f"""Mint a new NFT in the collection at {collection_address} with these parameters:
            - Token ID: {nft_data['token_id']}
            - Name: {nft_data['name']}
            - Description: {nft_data['description']}
            - Image URL: {image_url}
            - Attributes: {json.dumps(nft_data['attributes'])}
            
            Use the contract_call tool to call the mint function with these exact parameters.
            Return the transaction hash."""

            # Run the agent to mint the NFT
            result = await self.agent_executor.ainvoke(
                {"messages": [HumanMessage(content=mint_prompt)]},
                RunnableConfig(
                    configurable={
                        "thread_id": "nft_generator",
                        "checkpoint_ns": "mint_nft",
                    }
                )
            )
            
            # Extract transaction hash
            tx_hash = None
            if isinstance(result, dict):
                tx_hash = result.get('output') or next(
                    (msg.content.strip() for msg in reversed(result.get('messages', []))
                    if isinstance(msg, AIMessage) and msg.content),
                    None
                )
            
            if not tx_hash:
                raise Exception("Failed to extract transaction hash from mint result")

            return {
                "collection_address": collection_address,
                "token_id": nft_data["token_id"],
                "name": nft_data["name"],
                "description": nft_data["description"],
                "attributes": nft_data["attributes"],
                "image_url": image_url,
                "transaction_hash": tx_hash
            }

        except Exception as e:
            raise Exception(f"Failed to generate and mint NFT: {str(e)}")

    def _create_generation_prompt(self, description, traits, style, references):
        """Create a detailed prompt for NFT metadata generation."""
        reference_text = ""
        if references:
            reference_text = "\nReference NFTs for inspiration:\n" + "\n".join(
                f"- {ref.get('name', 'Unnamed')}: {ref.get('description', 'No description')}"
                for ref in references
            )

        trait_text = ""
        if traits:
            trait_text = "\nRequired traits:\n" + "\n".join(
                f"- {trait['type']}: {trait['value']} (probability: {trait.get('probability', 'not specified')})"
                for trait in traits
            )

        style_text = f"\nArtistic style preferences: {', '.join(style)}" if style else ""

        return f"""Generate unique and creative NFT metadata for a new digital artwork.

Core Description: {description}
{style_text}
{trait_text}
{reference_text}

Generate a JSON object with the following structure:
{{
    "token_id": "<unique numeric identifier>",
    "name": "<creative and memorable name>",
    "description": "<detailed artistic description>",
    "attributes": [
        {{
            "trait_type": "<trait category>",
            "value": "<trait value>"
        }}
        ...
    ]
}}

Ensure the metadata is unique, creative, and aligns with the provided preferences.
Return only valid JSON without any additional text."""

    async def _generate_image(self, nft_data: Dict) -> str:
        """Generate an image based on NFT metadata.
        
        This is a placeholder that should be implemented with your preferred
        image generation service (e.g., DALL-E, Stable Diffusion, Midjourney).
        """
        raise NotImplementedError(
            "Image generation must be implemented before using this feature"
        )

    # Add a method to get the address
    async def get_wallet_address(self):
        """Get the CDP wallet address"""
        # get_address is not async, so call it directly
        return self.wallet_provider.get_address()

    async def initialize(self):
        """Initialize async components"""
        address = self.wallet_provider.get_address()  # Removed await
        print(f"\nCDP Wallet Address: {address}\n")
        return self 