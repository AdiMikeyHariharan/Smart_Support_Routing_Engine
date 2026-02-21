# agents.py
import pulp
from typing import Dict

# Agent registry: agent_id -> {'skills': Dict[str, float], 'capacity': int, 'current_load': int}
agents: Dict[int, Dict] = {
    1: {'skills': {'Technical': 0.9, 'Billing': 0.1, 'Legal': 0.0}, 'capacity': 5, 'current_load': 0},
    2: {'skills': {'Technical': 0.1, 'Billing': 0.9, 'Legal': 0.0}, 'capacity': 5, 'current_load': 0},
    3: {'skills': {'Technical': 0.0, 'Billing': 0.0, 'Legal': 0.9}, 'capacity': 5, 'current_load': 0},
}

def route_to_agent(ticket_cat: str, urgency: float) -> int:
    prob = pulp.LpProblem("Agent_Assignment", pulp.LpMaximize)
    
    # Variables: assignment to each agent
    assignment = pulp.LpVariable.dicts("assign", agents.keys(), cat=pulp.LpBinary)
    
    # Objective: maximize skill match weighted by urgency
    prob += pulp.lpSum([assignment[a] * agents[a]['skills'].get(ticket_cat, 0) * (1 + urgency) for a in agents])
    
    # Constraints
    prob += pulp.lpSum(assignment) == 1  # Assign to exactly one agent
    for a in agents:
        if agents[a]['current_load'] >= agents[a]['capacity']:
            prob += assignment[a] == 0
    
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    for a in agents:
        if pulp.value(assignment[a]) == 1:
            agents[a]['current_load'] += 1
            return a
    return -1  # No agent available