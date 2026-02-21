# agents.py
import pulp
from typing import Dict

# Agent registry: agent_id -> {'skills': Dict[str, float], 'capacity': int, 'current_load': int}
agents: Dict[int, Dict] = {
    1: {'skills': {'Technical': 0.9, 'Billing': 0.1, 'Legal': 0.0}, 'capacity': 5, 'current_load': 0},
    2: {'skills': {'Technical': 0.1, 'Billing': 0.9, 'Legal': 0.0}, 'capacity': 5, 'current_load': 0},
    3: {'skills': {'Technical': 0.0, 'Billing': 0.0, 'Legal': 0.9}, 'capacity': 5, 'current_load': 0},
}

# agents.py
import pulp
from typing import Dict

agents: Dict[int, Dict] = {
    1: {'skills': {'Technical': 0.9, 'Billing': 0.1, 'Legal': 0.0}, 'capacity': 5, 'current_load': 0},
    2: {'skills': {'Technical': 0.1, 'Billing': 0.9, 'Legal': 0.0}, 'capacity': 5, 'current_load': 0},
    3: {'skills': {'Technical': 0.0, 'Billing': 0.0, 'Legal': 0.9}, 'capacity': 5, 'current_load': 0},
}


def route_to_agent(ticket_cat: str, urgency: float) -> int:

    available_agents = [
        a for a in agents
        if agents[a]['current_load'] < agents[a]['capacity']
    ]

    if not available_agents:
        return -1

    prob = pulp.LpProblem("Agent_Assignment", pulp.LpMaximize)

    assignment = pulp.LpVariable.dicts(
        "assign",
        available_agents,
        cat=pulp.LpBinary
    )

    # 🔥 Multi-objective score
    objective_terms = []

    for a in available_agents:
        skill = agents[a]['skills'].get(ticket_cat, 0)

        load_ratio = agents[a]['current_load'] / agents[a]['capacity']
        availability = 1 - load_ratio  # fairness factor

        # Agentic weighted objective
        score = (
            0.6 * skill +
            0.3 * availability +
            0.1 * (skill * urgency)
        )

        objective_terms.append(assignment[a] * score)

    prob += pulp.lpSum(objective_terms)

    # Must assign exactly one
    prob += pulp.lpSum(assignment[a] for a in available_agents) == 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != "Optimal":
        return -1

    for a in available_agents:
        if pulp.value(assignment[a]) == 1:
            agents[a]['current_load'] += 1
            return a

    return -1