import os
from dotenv import load_dotenv

from workflow_agents.base_agents import (
    KnowledgeAugmentedPromptAgent,
)


load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY is not set.")

persona = "a product manager"

knowledge = """
A user story should follow this format:

As a [type of user],
I want [an action],
so that [a benefit].
"""

agent = KnowledgeAugmentedPromptAgent(
    openai_api_key,
    persona,
    knowledge,
)

prompt = (
    "Create two user stories for an email router."
)

print("=" * 70)
print("KNOWLEDGE AUGMENTED PROMPT AGENT TEST")
print("=" * 70)

print("\nScript: knowledge_augmented_prompt_agent_test.py")

print("\nPersona:")
print(persona)

print("\nPrompt:")
print(prompt)

print("\nSupplied knowledge:")
print(knowledge)

print(
    "\nKnowledge source confirmation: "
    "KnowledgeAugmentedPromptAgent uses the supplied knowledge "
    "as its permitted knowledge source. The system prompt instructs "
    "the agent to use this knowledge and not rely on previous "
    "conversational context."
)

response = agent.respond(prompt)

print("\nAgent response:")
print(response)

print(
    "\nKnowledgeAugmentedPromptAgent test completed successfully."
)