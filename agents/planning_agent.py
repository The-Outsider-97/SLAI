from typing import List
import networkx as nx

class PlanningAgent:
    def __init__(self):
        self.plan_graph = nx.DiGraph()

    def decompose_goal(self, goal: str) -> List[str]:
        # Example decomposition - this should use NLP model in production
        return [f"Subtask {i+1} for {goal}" for i in range(3)]

    def build_plan(self, goal: str):
        subtasks = self.decompose_goal(goal)
        self.plan_graph.clear()
        self.plan_graph.add_node(goal)
        for sub in subtasks:
            self.plan_graph.add_node(sub)
            self.plan_graph.add_edge(goal, sub)
        return subtasks

    def visualize_plan(self):
        import matplotlib.pyplot as plt
        nx.draw(self.plan_graph, with_labels=True, node_color='lightblue', font_size=10)
        plt.show()
