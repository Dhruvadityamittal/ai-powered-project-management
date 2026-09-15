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

print("Testing DirectPromptAgent")
print(f"Script: direct_prompt_agent_test.py")
print(f"Prompt: {prompt}")

print("\nAgent response:")
print(response)

print(
    "\nKnowledge source: DirectPromptAgent used the general knowledge "
    "encoded in the selected LLM model because no persona, external "
    "knowledge, retrieval step, or system prompt was provided."
)

print("\nDirectPromptAgent test completed successfully.")