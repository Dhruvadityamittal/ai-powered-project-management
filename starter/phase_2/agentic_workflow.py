# agentic_workflow.py

"""General-purpose TPM workflow demonstrated with the Email Router specification."""

import os

from dotenv import load_dotenv

from workflow_agents.base_agents import (
    ActionPlanningAgent,
    EvaluationAgent,
    KnowledgeAugmentedPromptAgent,
    RoutingAgent,
)


load_dotenv()

openai_api_key = os.getenv(
    "OPENAI_API_KEY"
)

if not openai_api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. "
        "Add it to .env or your environment."
    )


# ---------------------------------------------------------------------------
# Load the product specification
# ---------------------------------------------------------------------------

spec_path = os.path.join(
    os.path.dirname(__file__),
    "Product-Spec-Email-Router.txt",
)

with open(
    spec_path,
    "r",
    encoding="utf-8",
) as f:
    product_spec = f.read()


# ---------------------------------------------------------------------------
# Action Planning Agent
# ---------------------------------------------------------------------------

knowledge_action_planning = """
Stories are defined from a product specification
by identifying:

- a persona
- an action
- a desired outcome

Each story represents a specific functionality
of the product described in the specification.

Features are defined by grouping related user stories.

Tasks are defined for each story and represent
the engineering work required to develop the product.

A development plan for a product contains all
these components.
"""

action_planning_agent = ActionPlanningAgent(
    openai_api_key,
    knowledge_action_planning,
)


# ---------------------------------------------------------------------------
# Product Manager Agent
# ---------------------------------------------------------------------------

persona_product_manager = (
    "You are a Product Manager. "
    "You are responsible for defining "
    "the user stories for a product."
)

knowledge_product_manager = (
    "Stories are defined by writing sentences with "
    "a persona, an action, and a desired outcome. "
    "The sentences always start with: As a. "
    "Write several stories for the product specification "
    "below, where the personas are the different users "
    "of the product.\n\n"
    "PRODUCT SPECIFICATION:\n"
    f"{product_spec}"
)

product_manager_knowledge_agent = (
    KnowledgeAugmentedPromptAgent(
        openai_api_key,
        persona_product_manager,
        knowledge_product_manager,
    )
)


# ---------------------------------------------------------------------------
# Product Manager Evaluation Agent
# ---------------------------------------------------------------------------

persona_product_manager_eval = (
    "You are an evaluation agent that checks "
    "the answers of other worker agents."
)

product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_product_manager_eval,
    """
    The answer should be stories that follow
    this structure:

    As a [type of user],
    I want [an action or feature],
    so that [benefit/value].
    """,
    product_manager_knowledge_agent,
    3,
)


# ---------------------------------------------------------------------------
# Program Manager Agent
# ---------------------------------------------------------------------------

persona_program_manager = (
    "You are a Program Manager. "
    "You are responsible for defining "
    "the features for a product."
)

knowledge_program_manager = (
    "Features of a product are defined by organizing "
    "similar user stories into cohesive groups.\n\n"
    "PRODUCT SPECIFICATION:\n"
    f"{product_spec}"
)

program_manager_knowledge_agent = (
    KnowledgeAugmentedPromptAgent(
        openai_api_key,
        persona_program_manager,
        knowledge_program_manager,
    )
)


# ---------------------------------------------------------------------------
# Program Manager Evaluation Agent
# ---------------------------------------------------------------------------

persona_program_manager_eval = (
    "You are an evaluation agent that checks "
    "the answers of other worker agents."
)

program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_program_manager_eval,
    """
    The answer should be product features that
    follow the following structure:

    Feature Name:
    A clear, concise title that identifies
    the capability.

    Description:
    A brief explanation of what the feature does
    and its purpose.

    Key Functionality:
    The specific capabilities or actions the feature
    provides.

    User Benefit:
    How this feature creates value for the user.
    """,
    program_manager_knowledge_agent,
    3,
)


# ---------------------------------------------------------------------------
# Development Engineer Agent
# ---------------------------------------------------------------------------

persona_dev_engineer = (
    "You are a Development Engineer. "
    "You are responsible for defining "
    "the development tasks for a product."
)

knowledge_dev_engineer = (
    "Development tasks are defined by identifying "
    "what needs to be built to implement each user story.\n\n"
    "PRODUCT SPECIFICATION:\n"
    f"{product_spec}"
)

development_engineer_knowledge_agent = (
    KnowledgeAugmentedPromptAgent(
        openai_api_key,
        persona_dev_engineer,
        knowledge_dev_engineer,
    )
)


# ---------------------------------------------------------------------------
# Development Engineer Evaluation Agent
# ---------------------------------------------------------------------------

persona_dev_engineer_eval = (
    "You are an evaluation agent that checks "
    "the answers of other worker agents."
)

development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_dev_engineer_eval,
    """
    The answer should be tasks following this structure:

    Task ID:
    A unique identifier for tracking purposes.

    Task Title:
    Brief description of the specific development work.

    Related User Story:
    Reference to the parent user story.

    Description:
    Detailed explanation of the technical work required.

    Acceptance Criteria:
    Specific requirements that must be met for completion.

    Estimated Effort:
    Time or complexity estimation.

    Dependencies:
    Any tasks that must be completed first.
    """,
    development_engineer_knowledge_agent,
    3,
)


# ---------------------------------------------------------------------------
# Support functions
# ---------------------------------------------------------------------------

def product_manager_support_function(query):
    result = (
        product_manager_evaluation_agent.evaluate(
            query
        )
    )

    return result["final_response"]


def program_manager_support_function(query):
    result = (
        program_manager_evaluation_agent.evaluate(
            query
        )
    )

    return result["final_response"]


def development_engineer_support_function(query):
    result = (
        development_engineer_evaluation_agent.evaluate(
            query
        )
    )

    return result["final_response"]


# ---------------------------------------------------------------------------
# Routing Agent
# ---------------------------------------------------------------------------

routing_agent = RoutingAgent(
    openai_api_key
)

routing_agent.agents = [
    {
        "name": "Product Manager",
        "description": (
            "Responsible for defining product personas "
            "and user stories only. Produces stories "
            "using the format: "
            "As a [user], I want [action] so that "
            "[benefit]. Does not define features or "
            "engineering tasks."
        ),
        "func": product_manager_support_function,
    },
    {
        "name": "Program Manager",
        "description": (
            "Responsible for defining product features "
            "by grouping related user stories into "
            "cohesive feature areas. Does not define "
            "engineering tasks."
        ),
        "func": program_manager_support_function,
    },
    {
        "name": "Development Engineer",
        "description": (
            "Responsible for defining detailed engineering "
            "development tasks for user stories, including "
            "implementation work, acceptance criteria, "
            "effort, and dependencies."
        ),
        "func": development_engineer_support_function,
    },
]


# ---------------------------------------------------------------------------
# Execute Workflow
# ---------------------------------------------------------------------------

print(
    "\n*** Workflow execution started ***\n"
)


workflow_prompt = (
    "What would the development tasks "
    "for this product be?"
)

print(
    "Task to complete in this workflow:"
)

print(
    f"workflow prompt = {workflow_prompt}"
)

print(
    "\nDefining workflow steps "
    "from the workflow prompt"
)


workflow_steps = (
    action_planning_agent.extract_steps_from_prompt(
        workflow_prompt
    )
)

if not workflow_steps:
    raise RuntimeError(
        "Action Planning Agent returned "
        "no workflow steps."
    )


completed_steps = []


for step_number, step in enumerate(
    workflow_steps,
    start=1,
):

    print(
        f"\n=== Processing workflow step "
        f"{step_number}: {step} ==="
    )

    result = routing_agent.route(
        step
    )

    completed_steps.append(
        result
    )

    print(
        f"\n=== Result for step "
        f"{step_number} ==="
    )

    print(result)


# ---------------------------------------------------------------------------
# Final Output
# ---------------------------------------------------------------------------

print(
    "\n*** Final workflow output ***\n"
)

if completed_steps:
    print(
        completed_steps[-1]
    )
else:
    print(
        "No workflow results were produced."
    )