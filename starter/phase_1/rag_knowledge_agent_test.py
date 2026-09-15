import os

from dotenv import load_dotenv

from workflow_agents.base_agents import RAGKnowledgePromptAgent


load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY is not set.")


agent = RAGKnowledgePromptAgent(
    openai_api_key,
    "a technical project manager",
)


knowledge = """
The Email Router automatically classifies incoming emails,
generates responses for routine inquiries, and routes complex
communications to appropriate subject matter experts.
"""


prompt = "How should complex emails be handled?"


print("=" * 70)
print("RAG KNOWLEDGE PROMPT AGENT TEST")
print("=" * 70)

print("\nScript: rag_knowledge_prompt_agent.py")
print("Test: rag_knowledge_prompt_agent_test.py")

print("\nPrompt:")
print(prompt)

print(
    "\nKnowledge source: The agent retrieves relevant knowledge "
    "from the provided knowledge using embedding-based semantic search."
)


# Split the knowledge into chunks
agent.chunk_text(knowledge)

# Generate embeddings for the chunks
agent.calculate_embeddings()


# Retrieve the most relevant knowledge and generate an answer
response = agent.find_prompt_in_knowledge(prompt)


print("\nRetrieved/generated answer:")
print(response)

print("\nRAGKnowledgePromptAgent test completed successfully.")