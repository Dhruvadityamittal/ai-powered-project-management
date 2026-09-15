import os

from dotenv import load_dotenv

from workflow_agents.base_agents import (
    RAGKnowledgePromptAgent,
)


load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set."
    )

RAG_knowledge_prompt_agent = RAGKnowledgePromptAgent(
    openai_api_key,
    "a knowledge retrieval assistant",
)

knowledge_text = """
Clara was born in Boston and became interested in
science, engineering, culture, and ethics.

Clara created a podcast called "Crosscurrents".

The podcast explored the intersection of science,
culture, and ethics.

She interviewed researchers, engineers, artists,
and activists.

Clara also worked with retrieval-augmented generation
and experimented with semantic search, vector databases,
and multimodal embeddings.

Her podcast often explored emerging technology,
science, cultural preservation, and ethical questions.
"""

chunks = RAG_knowledge_prompt_agent.chunk_text(
    knowledge_text
)

embeddings = (
    RAG_knowledge_prompt_agent.calculate_embeddings()
)

prompt = (
    "What is the podcast that Clara hosts about?"
)

print(prompt)

prompt_answer = (
    RAG_knowledge_prompt_agent.find_prompt_in_knowledge(
        prompt
    )
)

print(prompt_answer)