import os, sys
import time
import json
import logging
import argparse
import importlib
import subprocess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from recursive_improvement.rewriter import Rewriter
from typing import Optional
from dotenv import load_dotenv
from string import Template
from rnd_loop.evaluator import Evaluator

try:
    from collaborative.shared_memory import SharedMemory
    shared_memory = SharedMemory()
except ImportError:
    shared_memory = None
    print("[WARN] SharedMemory not available. Proceeding without logging.")

TEMPLATES = {
    "agent": Template("""
class ${AgentName}:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = ${learning_rate}
        self.epsilon = ${epsilon}

    def act(self, state):
        # TODO: Add decision logic
        pass

    def train(self, experience):
        # TODO: Implement training step
        pass
    """),

    "evaluator": Template("""
class ${EvaluatorName}:
    def __init__(self):
        self.history = []

    def evaluate(self, agent, environment):
        result = agent.act(environment.reset())
        self.history.append(result)
        return result
    """),

    "runner": Template("""
if __name__ == "__main__":
    from agents.${agent_module} import ${AgentName}

    agent = ${AgentName}(state_size=${state_size}, action_size=${action_size})
    print("Initialized agent with state size ${state_size} and action size ${action_size}")
    """)
}

def generate_code(template_type: str, context: dict) -> str:
    if template_type not in TEMPLATES:
        raise ValueError(f"Unknown template_type: {template_type}")

    template = TEMPLATES[template_type]
    try:
        return template.substitute(context)
    except KeyError as e:
        raise KeyError(f"Missing key for template: {e}")

def save_code_to_file(code_str: str, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(code_str)

    SharedMemory().set(f"codegen_{os.path.basename(filepath)}", {
        "path": filepath,
        "template": "generated",
        "status": "written"
    })

def load_vars_from_file(path: str) -> dict:
    if path.endswith(".json"):
        with open(path) as f:
            return json.load(f)
    elif path.endswith(".yaml") or path.endswith(".yml"):
        with open(path) as f:
            return yaml.safe_load(f)
    else:
        raise ValueError("Unsupported config file format. Use .json or .yaml")

def rewrite_and_reload_if_needed(agent_path: str, config_path: str, reward: float):
    rewriter = Rewriter(agent_path, config_path)
    if rewriter.trigger_recursive_improvement(reward):
        module_name = agent_path.replace("/", ".").replace(".py", "")
        agent_class_name = os.path.splitext(os.path.basename(agent_path))[0]
        updated_cls = rewriter.reload_agent(module_name, agent_class_name)
        print(f"\n✓ Agent reloaded with updates from {agent_path}")
        return updated_cls
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate code from SLAI templates")
    parser.add_argument("--template", required=False, help="Template type [agent, evaluator, runner]")
    parser.add_argument("--output", required=False, help="Output path for generated code")
    parser.add_argument("--vars", nargs="*", help="Key=Value pairs or a path to .json/.yaml config")
    parser.add_argument("--auto-eval", action="store_true", help="Automatically evaluate generated agent")
    parser.add_argument("--rewrite-check", action="store_true", help="Apply rewrite logic after generation")
    parser.add_argument("--task-data", type=str, help="Path to JSON file containing test task data")

    args = parser.parse_args()

    if not args.template or not args.output:
        print("\n[Interactive Mode] Fill in required details below:")
        args.template = input("Template type (agent/evaluator/runner): ")
        args.output = input("Output file path: ")

    if not args.vars:
        print("No --vars provided. Use --vars AgentName=Foo learning_rate=0.001 or path to config file.")
        sys.exit(1)

    if len(args.vars) == 1 and os.path.isfile(args.vars[0]):
        context = load_vars_from_file(args.vars[0])
    else:
        context = dict(var.split("=") for var in args.vars)

    print(f"\n>>> Generating {args.template} code")
    code = generate_code(args.template, context)
    save_code_to_file(code, args.output)
    print(f"\n✓ Code saved to {args.output}")

    if args.rewrite_check:
        print("\n> Checking if rewrite is necessary...")
        reward = float(input("Current reward: "))
        rewrite_and_reload_if_needed(args.output, "config/agent_config.yaml", reward)

    if args.auto_eval and args.template == "runner":
        print("\n> Launching generated runner...")
        subprocess.run(["python", args.output])

if __name__ == "__main__" and not sys.argv[1:]:
    print("\n>>> Generating example agent, evaluator and runner")

    a_code = generate_code("agent", {
        "AgentName": "MyAgent",
        "learning_rate": 0.001,
        "epsilon": 1.0
    })
    save_code_to_file(a_code, "generated/MyAgent.py")

    e_code = generate_code("evaluator", {
        "EvaluatorName": "SimpleEvaluator"
    })
    save_code_to_file(e_code, "generated/SimpleEvaluator.py")

    r_code = generate_code("runner", {
        "AgentName": "MyAgent",
        "agent_module": "MyAgent",
        "state_size": 4,
        "action_size": 2
    })
    save_code_to_file(r_code, "generated/run_agent.py")

    print("\n>>> Summary of generated files:")
    for f in ["MyAgent.py", "SimpleEvaluator.py", "run_agent.py"]:
        print(f"- generated/{f} {'(exists)' if os.path.exists('generated/' + f) else '(missing)'}")

    print("\n>>> Done generating example components.")
