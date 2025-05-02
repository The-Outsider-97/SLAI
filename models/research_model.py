"""
slai_lm: Summarizes input and manages context for response.
browser_agent: Searches the web, parses content, and extracts relevant facts.
Output: Text, APA-style citations, time spent, and a follow-up query.
"""

import time

from src.agents.browser_agent import BrowserAgent
from models.slai_lm import get_shared_slailm
from src.utils.agent_factory import AgentFactory
from src.collaborative.shared_memory import SharedMemory

class ResearchModel:
    def __init__(self, shared_memory=None, agent_factory=None):
        self.shared_memory = shared_memory or SharedMemory()
        self.factory = agent_factory or AgentFactory(shared_resources={"shared_memory": self.shared_memory})
        self.slai_lm = get_shared_slailm(self.shared_memory, agent_factory=self.factory)
        self.browser = BrowserAgent(self.shared_memory, agent_factory=self.factory, slai_lm=get_shared_slailm)

    def run(self, user_prompt: str) -> dict:
        start_time = time.time()
        trace_log = {"input": user_prompt, "steps": []}

        summary = self.slai_lm.grammar_processor.summarize(user_prompt)
        trace_log["steps"].append({"stage": "summary", "value": summary})

        # Internet-assisted search & processing
        result_facts = self.browser.process_page(user_prompt)
        trace_log["steps"].append({"stage": "web_results", "value": result_facts})

        # Build response text
        response = self.slai_lm.dialogue_context.compose_response(summary)
        trace_log["steps"].append({"stage": "response_generation", "value": response})

        # Convert results to APA sources
        sources = self._format_sources(result_facts)

        end_time = time.time()
        duration = round(end_time - start_time, 2)

        return {
            "summary": summary,
            "response": response,
            "sources": sources,
            "time_taken": f"{duration} seconds",
            "follow_up_question": self._generate_follow_up(summary, response),
            "agent_trace": trace_log
        }

    def _format_sources(self, facts: list) -> list:
        citations = []
        for i, fact in enumerate(facts):
            try:
                title = fact.get('text')[:50].strip(". ") + "..."
                url = fact.get('link')
                citations.append(f"{i+1}. {title} Retrieved from {url}")
            except Exception:
                continue
        return citations

    def _generate_follow_up(self, summary, response) -> str:
        return f"What would be a deeper question related to: “{summary[:60]}...”? Discuss implications."

# === Example Usage ===
if __name__ == "__main__":
    model = ResearchModel()
    output = model.run("What are the latest advancements in Mars colonization?")
    for key, val in output.items():
        print(f"\n--- {key.upper()} ---")
        print(val if not isinstance(val, list) else "\n".join(val))
