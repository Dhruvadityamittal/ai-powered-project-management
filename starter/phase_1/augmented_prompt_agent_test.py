import os
from dotenv import load_dotenv

from workflow_agents.base_agents import AugmentedPromptAgent


load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set."
    )

persona = (
    "a senior technical project manager"
)

agent = AugmentedPromptAgent(
    openai_api_key,
    persona,
)

prompt = (
    "Explain the importance of clear acceptance criteria."
)

print("Testing AugmentedPromptAgent")
print(f"Persona: {persona}")
print(f"Prompt: {prompt}")

response = agent.respond(prompt)

print("\nResponse:")
print(response)

print(
    "\nAugmentedPromptAgent test completed successfully."
)