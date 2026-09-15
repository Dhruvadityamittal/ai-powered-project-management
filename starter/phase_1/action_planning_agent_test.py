import os
from dotenv import load_dotenv

from workflow_agents.base_agents import (
    ActionPlanningAgent,
)


load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set."
    )

knowledge = """
A development plan contains:

- User stories
- Product features
- Engineering tasks

User stories describe what users need.
Features group related stories.
Engineering tasks describe the technical
work needed to implement them.
"""

agent = ActionPlanningAgent(
    openai_api_key,
    knowledge,
)

prompt = (
    "Plan the work required to turn a product "
    "specification into an actionable development plan."
)

print("Testing ActionPlanningAgent")
print(f"Prompt: {prompt}")

steps = agent.extract_steps_from_prompt(prompt)

print("\nGenerated workflow steps:")

for index, step in enumerate(
    steps,
    start=1,
):
    print(f"{index}. {step}")

print(
    "\nActionPlanningAgent test completed successfully."
)