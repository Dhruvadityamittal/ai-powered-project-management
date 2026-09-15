import os
from dotenv import load_dotenv

from workflow_agents.base_agents import RoutingAgent


load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set."
    )


def product_manager(query):
    return (
        "Product Manager selected.\n"
        f"Task: {query}"
    )


def program_manager(query):
    return (
        "Program Manager selected.\n"
        f"Task: {query}"
    )


def development_engineer(query):
    return (
        "Development Engineer selected.\n"
        f"Task: {query}"
    )


agent = RoutingAgent(openai_api_key)

agent.agents = [
    {
        "name": "Product Manager",
        "description": (
            "Defines user personas and user stories."
        ),
        "func": product_manager,
    },
    {
        "name": "Program Manager",
        "description": (
            "Defines product features."
        ),
        "func": program_manager,
    },
    {
        "name": "Development Engineer",
        "description": (
            "Defines detailed engineering tasks."
        ),
        "func": development_engineer,
    },
]

query = (
    "Define the technical development tasks "
    "required to implement the email router."
)

print("Testing RoutingAgent")
print(f"Query: {query}")

result = agent.route(query)

print("\nResult:")
print(result)

print(
    "\nRoutingAgent test completed successfully."
)