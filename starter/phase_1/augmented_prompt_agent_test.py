import os
from dotenv import load_dotenv

from workflow_agents.base_agents import AugmentedPromptAgent


load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY is not set.")

persona = "a senior technical project manager"

agent = AugmentedPromptAgent(
    openai_api_key,
    persona,
)

prompt = (
    "Explain the importance of clear acceptance criteria."
)

print("=" * 70)
print("AUGMENTED PROMPT AGENT TEST")
print("=" * 70)

print("\nScript: augmented_prompt_agent_test.py")

print("\nPersona:")
print(persona)

print("\nPrompt:")
print(prompt)

print(
    "\nKnowledge source: AugmentedPromptAgent uses the LLM's "
    "general knowledge. No external knowledge or retrieval source "
    "is provided."
)

print(
    "\nPersona impact: The system prompt establishes the agent as "
    "a senior technical project manager, which guides the response "
    "to emphasize professional project-management practices, "
    "perspective, and terminology."
)

response = agent.respond(prompt)

print("\nAgent response:")
print(response)

print("\nAugmentedPromptAgent test completed successfully.")