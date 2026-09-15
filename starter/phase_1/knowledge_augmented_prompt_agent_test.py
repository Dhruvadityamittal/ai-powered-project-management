import os
from dotenv import load_dotenv

from workflow_agents.base_agents import (
    KnowledgeAugmentedPromptAgent,
)


load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set."
    )

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

print(
    "Testing KnowledgeAugmentedPromptAgent"
)

print(f"Prompt: {prompt}")

response = agent.respond(prompt)

print("\nResponse:")
print(response)

print(
    "\nKnowledgeAugmentedPromptAgent "
    "test completed successfully."
)