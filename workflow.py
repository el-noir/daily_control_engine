from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END

class DailyState(TypedDict):
    energy_level: int
    sleep_hours: float
    tasks: List[str]
    selected_tasks: List[str]
    completed_tasks: List[str]
    distractions: List[str]
    score: int
    suggestion: str

def score_tasks(state: DailyState) -> DailyState:
    energy = state['energy_level']
    tasks = state['tasks']

    if energy < 5:
        state['selected_tasks'] = tasks[:2]
    else:
        state['selected_tasks'] = tasks[:5]
    
    return state

def limit_to_3_tasks(state: DailyState) -> DailyState:
    state['selected_tasks'] = state['selected_tasks'][:3]
    return state

def generate_plan(state: DailyState) -> DailyState:
    
    state['suggestion'] = (
        "Focus deeply on: " + ", ".join(state['selected_tasks'])
    )
    return state

builder = StateGraph(DailyState)

builder.add_node("score_tasks", score_tasks)
builder.add_node("limit_to_3_tasks", limit_to_3_tasks)
builder.add_node("generate_plan", generate_plan)

builder.add_edge(START, "score_tasks")

builder.add_edge("score_tasks", "limit_to_3_tasks")
builder.add_edge("limit_to_3_tasks", "generate_plan")

builder.add_edge("generate_plan", END)

morning_graph = builder.compile()

def analyze_performance(state: DailyState) -> DailyState:
    
    if len(state["selected_tasks"]) == 0:
        state["score"] =0
        return state

    completion_rate = len(state["completed_tasks"]) / len(state["selected_tasks"])

    state["score"] = round(completion_rate * 100)

    return state

def suggest_improvement(state: DailyState) -> DailyState:
    
    score = state["score"]

    if score == 100:
        state["suggestion"] = "Increase difficulty tommorow."
    elif score >= 60:
        state["suggestion"] = "Maintain pace but reduce distractions."
    else:
        state["suggestion"] = "Reduce workload and eliminate distractions."

    return state

night_builder = StateGraph(DailyState)

night_builder.add_node("analyze_performance", analyze_performance)
night_builder.add_node("suggest_improvement", suggest_improvement)

night_builder.add_edge(START, "analyze_performance")
night_builder.add_edge("analyze_performance", "suggest_improvement")
night_builder.add_edge("suggest_improvement", END)

night_graph = night_builder.compile()

if __name__ == "__main__":
    initial_state: DailyState = {
        "energy_level": 7,
        "sleep_hours": 7.5,
        "tasks": [
            "Work on backend API",
            "Study distributed systems",
            "Gym",
            "Fix deployment bug",
            "Read research paper"
        ],
        "selected_tasks": [],
        "completed_tasks": [],
        "distractions": [],
        "score": 0,
        "suggestion": ""
    }

    result = morning_graph.invoke(initial_state)

    print(result["suggestion"])

    # Simulate evening
    night_state = result.copy()
    night_state["completed_tasks"] = [
        "Work on backend API",
        "Gym"
    ]

    night_result = night_graph.invoke(night_state)

    print("Score:", night_result["score"])
    print("Suggestion:", night_result["suggestion"])
