import os
from dotenv import load_dotenv

from workflow_agents.base_agents import DirectPromptAgent


load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set."
    )

agent = DirectPromptAgent(openai_api_key)

prompt = (
    "Explain what a technical project manager does "
    "in one short paragraph."
)

response = agent.respond(prompt)

print("\nAgent response:")
print(response)