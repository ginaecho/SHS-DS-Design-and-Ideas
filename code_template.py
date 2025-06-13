# install semantic-kernel if not installed:
# pip install semantic-kernel

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.memory.azure_cognitive_search import AzureCognitiveSearchMemoryStore

# ========== CONFIGURATION ==========
AZURE_OPENAI_DEPLOYMENT = "gpt-4-deployment"
AZURE_OPENAI_ENDPOINT = "https://your-openai-endpoint.openai.azure.com/"
AZURE_OPENAI_KEY = "your-openai-key"

AZURE_SEARCH_ENDPOINT = "https://your-search-service.search.windows.net/"
AZURE_SEARCH_KEY = "your-search-key"
AZURE_SEARCH_INDEX_NAME = "teamplay-index"

# ========== SETUP KERNEL ==========
kernel = sk.Kernel()

# Add AI Service
kernel.add_chat_service(
    "chat_completion",
    AzureChatCompletion(
        deployment_name=AZURE_OPENAI_DEPLOYMENT,
        endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY
    )
)

# Add Semantic Memory (RAG)
kernel.register_memory_store(
    memory_store=AzureCognitiveSearchMemoryStore(
        endpoint=AZURE_SEARCH_ENDPOINT,
        api_key=AZURE_SEARCH_KEY,
        index_name=AZURE_SEARCH_INDEX_NAME
    )
)

# ========== FUNCTION: QUERY ==========
async def answer_user_query(domain: str, user_input: str):
    memory = kernel.memory
    retrieved_docs = await memory.search_async(domain, user_input, limit=3)

    context_snippets = "\n\n".join([doc.text for doc in retrieved_docs])
    prompt = f"""You are a helpful onboarding assistant for Teamplay app. The user is currently exploring the function: {domain}.
    
Based on the following context and user input, provide a clear, easy-to-understand answer for non-experts. Include useful tips or links if appropriate.

Context:
{context_snippets}

User Question:
{user_input}

Answer:"""

    chat = kernel.get_service("chat_completion")
    result = await chat.complete(prompt)
    return result.text

# ========== USAGE EXAMPLE ==========
import asyncio

async def main():
    domain = "UtilizationPerformance"
    question = "What does scan utilization mean in Teamplay?"
    answer = await answer_user_query(domain, question)
    print("Answer:\n", answer)

asyncio.run(main())
