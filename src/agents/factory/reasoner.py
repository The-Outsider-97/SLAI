import ast
import re
import json
import psutil
import io
import traceback
import signal # For basic timeout on Unix-like systems
import time
import copy

from collections import deque
from typing import List, Dict, Tuple, Any, Optional
from contextlib import redirect_stdout

from logs.logger import get_logger

logger = get_logger("Basic Zero Reasoner")

PROMPT_TEMPLATES = ""

# --- Configuration Constants ---
DEFAULT_BANNED_KEYWORDS = [
    "random", "multiprocessing", "subprocess", "threading", "socket",
    "os", "sys", "gc", "pickle", "marshal", "importlib", "eval", "exec" 
    # More restrictive for a basic, safer version
]
DEFAULT_BANNED_ASSERTION_KEYWORDS = ["assert", "try", "raise", "except"] # Example

RUN_CODE_TEMPLATE = """{code}
__bzs_result__ = repr(f({inputs}))"""

EVAL_INPUT_PREDICTION_TEMPLATE = """{code}
__bzs_result__ = repr({gold_output}) == repr(f({agent_input}))"""

EVAL_OUTPUT_PREDICTION_TEMPLATE = """{code}
__bzs_result__ = repr({gold_output}) == repr({agent_output})"""


# --- Helper Functions: Parsers & Formatters ---
def parse_imports(code_snippet: str) -> List[str]:
    """Extracts import statements from a code snippet."""
    imports = []
    try:
        tree = ast.parse(code_snippet)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(ast.unparse(node))
    except SyntaxError: # Fallback for snippets that might not be full modules
        import_pattern = r"^\s*(from\s+[\w.]+\s+import\s+[\w,\s]+|import\s+[\w.,\s]+)"
        for line in code_snippet.splitlines():
            if re.match(import_pattern, line.strip()):
                imports.append(line.strip())
    return imports

def parse_error_message(error_message: str) -> str:
    """Extracts the error type from a traceback message."""
    return error_message.split(':')[0].strip()

def remove_comments_and_docstrings(code: str) -> str:
    """Removes comments and docstrings from Python code."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef, ast.ClassDef, ast.Module)):
                if node.body and isinstance(node.body[0], ast.Expr):
                    if isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                        node.body = node.body[1:]
                    elif isinstance(node.body[0].value, ast.Str): # Python < 3.8
                        node.body = node.body[1:]
        return ast.unparse(tree)
    except Exception: # Fallback if parsing fails
        return remove_comments_and_docstrings_fallback(code)
    
def remove_comments_and_docstrings_fallback(code: str) -> str:
    """
    Removes comments (inline and full-line) and block docstrings from Python code.
    Only use this if AST parsing fails.
    """
    # Remove all triple-quoted strings (docstrings or multiline strings)
    code_no_block_strings = re.sub(r"('{3}|\"{3})(?s).*?\1", "", code)

    cleaned_lines = []
    for line in code_no_block_strings.splitlines():
        # Remove inline comments while preserving code
        line_no_comment = re.sub(r"#.*", "", line).rstrip()
        if line_no_comment.strip():  # Skip empty lines
            cleaned_lines.append(line_no_comment)

    return "\n".join(cleaned_lines)

def format_python_code_basic(code: str) -> str:
    """A very basic code formatter. Relies on ast.unparse for consistency."""
    try:
        return ast.unparse(ast.parse(code))
    except SyntaxError:
        return code # Return original if it can't be parsed

# --- Helper Functions: Checkers ---
def contains_banned_keywords(code: str, banned_keywords: List[str],
                               banned_assertion_keywords: Optional[List[str]] = None) -> bool:
    """Checks if the code contains any banned keywords or constructs."""
    if banned_assertion_keywords is None:
        banned_assertion_keywords = []
    
    # Check for banned keywords in the code string itself (covers comments, strings)
    for keyword in banned_keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', code):
            return True
            
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            # Check for banned imports (more precise than string search for imports)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for bk in banned_keywords: # Check parts of module path
                        if bk in alias.name.split('.'): return True
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for bk in banned_keywords:
                        if bk in node.module.split('.'): return True
                for alias in node.names:
                    for bk in banned_keywords:
                         if bk in alias.name.split('.'): return True
            
            # Check for banned assertion/error handling keywords using AST
            if isinstance(node, ast.Assert) and 'assert' in banned_assertion_keywords: return True
            if isinstance(node, ast.Raise) and 'raise' in banned_assertion_keywords: return True
            if isinstance(node, ast.Try) and 'try' in banned_assertion_keywords: return True
            if isinstance(node, ast.Name) and node.id == 'input' and 'input' in banned_keywords : return True
            if isinstance(node, ast.Name) and node.id == 'eval' and 'eval' in banned_keywords : return True
            if isinstance(node, ast.Name) and node.id == 'exec' and 'exec' in banned_keywords : return True


    except SyntaxError: # If code can't be parsed, rely on the initial string search
        pass # Already covered by the initial loop
    return False

# ---Python Code Executor ---
class PythonExecutor:
    """Executes Python code snippets with basic timeout and output capturing."""
    class _ExecutionTimeout(Exception):
        pass

    def __init__(self, timeout_length: int = 5):
        self.timeout_length = timeout_length

    def _timeout_handler(self, signum, frame):
        raise PythonExecutor._ExecutionTimeout("Execution timed out")

    def _execute_code_unsafe(self, code_to_run: str) -> Tuple[Any, str]:
        """
        Executes code using exec. Captures stdout and the result of the last expression.
        IMPORTANT: This is fundamentally unsafe for arbitrary code.
        """
        # Create a restricted global scope
        # Allow only a very limited set of builtins for safety.
        restricted_globals = {
            "__builtins__": {
                # Safe built-ins only
                "print": print, "repr": repr, "str": str, "int": int, "float": float,
                "list": list, "dict": dict, "tuple": tuple, "set": set, "len": len,
                "range": range, "abs": abs, "round": round, "min": min, "max": max,
                "sum": sum, "sorted": sorted, "zip": zip, "enumerate": enumerate,
                "map": map, "filter": filter, "any": any, "all": all,
        
                # Type guards
                "isinstance": isinstance, "hasattr": hasattr, "getattr": getattr,
        
                # Constants
                "True": True, "False": False, "None": None,
        
                # Errors (limited set)
                "ValueError": ValueError, "TypeError": TypeError,
                "IndexError": IndexError, "KeyError": KeyError,
                "NameError": NameError, "ZeroDivisionError": ZeroDivisionError,
                "Exception": Exception
            },
            # ONLY explicitly allow safe modules
            "copy": copy,  # If required
        }
        # Local scope for execution
        local_scope = {}
        
        stdout_capture = io.StringIO()
        status = "Done"
        result_val = None # This will hold the value of `__bzs_result__`

        old_handler = None
        if hasattr(signal, 'SIGALRM'): # signal.alarm is Unix-specific
            old_handler = signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(self.timeout_length)
        else: # Basic time tracking for non-Unix or when signal is unavailable
            start_time = time.time()

        try:
            with redirect_stdout(stdout_capture):
                exec(code_to_run, restricted_globals, local_scope)
            
            # The execution templates assign their result to `__bzs_result__`
            if "__bzs_result__" in local_scope:
                result_val = local_scope["__bzs_result__"]
            else: # Fallback if template wasn't used or __bzs_result__ wasn't set
                result_val = stdout_capture.getvalue().strip()
                if not result_val: # if stdout is empty
                    status = "Error: No result variable __bzs_result__ found and stdout was empty."


        except PythonExecutor._ExecutionTimeout:
            status = "Timeout Error"
        except Exception:
            status = traceback.format_exc().splitlines()[-1]
        finally:
            if hasattr(signal, 'SIGALRM') and old_handler is not None:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            elif not hasattr(signal, 'SIGALRM'): # Check non-Unix timeout
                if (time.time() - start_time) > self.timeout_length:
                    status = "Timeout Error (fallback timer)"
        
        if status == "Done":
            try: # Ensure result is serializable for practical purposes (e.g. JSON)
                json.dumps(result_val) 
            except (TypeError, OverflowError):
                # If not directly JSON serializable, keep its string representation
                # This aligns with how results are often handled (repr strings)
                if not isinstance(result_val, str): # if it's not already a string (e.g. from repr)
                    try:
                        result_val = repr(result_val)
                    except Exception:
                         status = "Error: Result could not be represented as a string."
                         result_val = None

        return result_val, status

    def run_code(self, code_snippet_def_f: str, problem_input_args_str: str,
                 imports: Optional[List[str]] = None) -> Tuple[Any, str]:
        """
        Runs the given code snippet (defining function 'f') with the provided input arguments.
        Returns the execution result (as a string from repr()) and status.
        """
        full_code_parts = []
        if imports:
            full_code_parts.extend(imports)
        full_code_parts.append(code_snippet_def_f)
        
        code_to_execute = RUN_CODE_TEMPLATE.format(
            code="\n".join(full_code_parts), 
            inputs=problem_input_args_str
        )
        
        try:
            ast.parse(code_to_execute) # Basic syntax check before execution
        except SyntaxError as e:
            return None, f"SyntaxError in prepared code: {e}"

        return self._execute_code_unsafe(code_to_execute)

    def check_determinism(self, code_snippet_def_f: str, problem_input_args_str: str,
                          imports: Optional[List[str]] = None, n_runs: int = 2) -> bool:
        """Checks if the code produces the same output over multiple runs."""
        outputs_set = set()
        for _ in range(n_runs):
            output, status = self.run_code(code_snippet_def_f, problem_input_args_str, imports)
            if "error" in status.lower() or "timeout" in status.lower():
                return False # Execution failure implies non-determinism for this check
            outputs_set.add(output) # Output is already a string (from repr)
        return len(outputs_set) == 1

    # --- Core Parsing Logic for LLM-like responses
    def parse_llm_code_input_output(
        llm_content: str,
        expect_input: bool = True,
        expect_output_block: bool = False # True for gen_code_o/e to parse the LLM's predicted output/error
        ) -> Tuple[bool, Dict[str, Any]]:
        """Parses python..., input..., and optionally output... blocks."""
        code_match = re.search(r"python\s*\n?(.*?)\n?", llm_content, re.DOTALL)
        if not code_match: return False, {"error": "No Python code block found."}
        code_snippet = code_match.group(1).strip()
        code_snippet = format_python_code_basic(code_snippet) # Basic formatting
        parsed_data = {"code": code_snippet, "imports": parse_imports(code_snippet)}
        
        if expect_input:
            input_match = re.search(r"```input\s*\n?(.*?)\n?```", llm_content, re.DOTALL)
            if not input_match: return False, {"error": "Expected input block not found."}
            parsed_data["input"] = input_match.group(1).strip()
        
        if expect_output_block:
            output_match = re.search(r"```output\s*\n?(.*?)\n?```", llm_content, re.DOTALL)
            if not output_match: return False, {"error": "Expected output block not found."}
            parsed_data["output_from_llm"] = output_match.group(1).strip()
            
        # Standardize main function name to 'f' if not already
        try:
            tree = ast.parse(parsed_data["code"])
            func_defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
            if func_defs and func_defs[0].name != 'f': # Simple: rename first function
                original_name = func_defs[0].name
                class RenameTransformer(ast.NodeTransformer):
                    def visit_FunctionDef(self, node):
                        if node.name == original_name: node.name = 'f'
                        return self.generic_visit(node)
                    def visit_Call(self, node):
                        if isinstance(node.func, ast.Name) and node.func.id == original_name:
                            node.func.id = 'f'
                        return self.generic_visit(node)
                tree = RenameTransformer().visit(tree)
                parsed_data["code"] = ast.unparse(tree)
        except SyntaxError:
            pass # If syntax error, caught later by executor
        
        return True, parsed_data

    def parse_llm_inputs_message_for_code_f(
        llm_content: str, num_expected_inputs: int
        ) -> Tuple[bool, Dict[str, Any]]:
        """Parses multiple input... blocks and one message... block."""
        inputs = re.findall(r"input\s*\n?(.*?)\n?", llm_content, re.DOTALL)
        message_match = re.search(r"message\s*\n?(.*?)\n?", llm_content, re.DOTALL)
        if len(inputs) < num_expected_inputs:
            return False, {"error": f"Found {len(inputs)} inputs, expected {num_expected_inputs}."}
        if not message_match:
            return False, {"error": "No message block found."}
        
        return True, {"inputs": [i.strip() for i in inputs[-num_expected_inputs:]],
                    "message": message_match.group(1).strip()}
    
    def parse_llm_predicted_code_for_pred_f(llm_content: str) -> Tuple[bool, str]:
        """Parses a single python... block for pred_code_f task."""
        code_match = re.search(r"python\s*\n?(.*?)\n?", llm_content, re.DOTALL)
        if not code_match:
            return False, ""
        code_snippet = code_match.group(1).strip()
        try:
            ast.parse(code_snippet)
            return True, format_python_code_basic(code_snippet)
        except SyntaxError:
            return False, ""
        
class BasicZeroReasoner:
    """
    A basic, single-file Python module for managing code reasoning tasks,
    simulating parts of the Absolute Zero Reasoner workflow without an LLM.
    """
    def init(self):
        self.executor_timeout: int = 5
        self.banned_keywords: Optional[List[str]] = None
        self.banned_assertion_keywords: Optional[List[str]] = None
        self.executor = PythonExecutor(timeout_length=self.executor_timeout)
        self.generated_data = {
        "gen_code_i": {'snippet', 'input', 'target_output', 'imports', 'executed_output'},
        "gen_code_o": [], # Successful items: {'snippet', 'input', 'llm_predicted_output', 'imports', 'executed_output'},
        "gen_code_e": [], # Successful items: {'snippet', 'input', 'llm_predicted_error', 'imports', 'actual_error'},
        "gen_code_f": {'original_snippet', 'generated_inputs', 'generated_message', 'imports', 'executed_outputs_for_gen_inputs'}
        }
        self.banned_keywords = self.banned_keywords if self.banned_keywords is not None else DEFAULT_BANNED_KEYWORDS
        self.banned_assertion_keywords = self.banned_assertion_keywords if self.banned_assertion_keywords is not None else DEFAULT_BANNED_ASSERTION_KEYWORDS
    def get_prompt(self, problem_type: str, **kwargs) -> str:
        """
        Generates a prompt for the specified problem type.
        kwargs: Parameters needed by the specific prompt template.
        """
        template = PROMPT_TEMPLATES.get(problem_type)
        if not template:
            raise ValueError(f"Unknown problem type for prompting: {problem_type}")
        
        # Common arguments for templates
        kwargs.setdefault("banned_keywords_str", ", ".join(self.banned_keywords))
        kwargs.setdefault("reference_snippets_str", self._format_reference_snippets(kwargs.get("reference_snippets")))
        
        if "io_pairs" in kwargs and problem_type == "pred_code_f": # Special formatting for pred_code_f
            kwargs["io_pairs_str"] = "\n".join([f"Input: {io[0]}\nOutput: {io[1]}" for io in kwargs["io_pairs"]])
            
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing argument for prompt template {problem_type}: {e}")

    def _format_reference_snippets(self, snippets: Optional[List[Dict]]) -> str:
        if not snippets: return "N/A"
        formatted = []
        for i, snip in enumerate(snippets):
            # Example snip: {'snippet': 'def f()...', 'input': '...', 'output': '...'}
            s = f"<Reference {i+1}>\n```python\n{snip.get('snippet', '')}\n```\n"
            if 'input' in snip: s += f"```input\n{snip.get('input')}\n```\n"
            if 'output' in snip: s += f"```output\n{snip.get('output')}\n```\n" # or 'error' for code_e
            elif 'error' in snip: s += f"```output\n{snip.get('error')}\n```\n"
            s += f"</Reference {i+1}>\n"
            formatted.append(s)
        return "\n".join(formatted)

    def process_llm_generation(self, problem_type: str, llm_content: str,
                            # Args specific to task types
                            target_output_for_gen_i: Optional[str] = None,
                            original_snippet_for_gen_f: Optional[str] = None,
                            original_imports_for_gen_f: Optional[List[str]] = None,
                            num_expected_inputs_for_gen_f: int = 3
                            ) -> Dict:
        """
        Processes content (simulated as from an LLM) for "generation" tasks.
        Validates, executes, and stores successful results.
        """
        if problem_type in ["gen_code_i", "gen_code_o", "gen_code_e"]:
            expect_llm_output_block = problem_type in ["gen_code_o", "gen_code_e"]
            success, parsed = PythonExecutor.parse_llm_code_input_output(llm_content, expect_output_block=expect_llm_output_block)
            if not success: return {"success": False, "error": parsed.get("error")}

            code, p_input, imports = parsed["code"], parsed["input"], parsed["imports"]
            llm_predicted_val = parsed.get("output_from_llm") # For gen_code_o (output) or gen_code_e (error type)

            if contains_banned_keywords(code, self.banned_keywords, 
                                    self.banned_assertion_keywords if problem_type == "gen_code_e" else []):
                return {"success": False, "error": "Code contains banned keywords."}

            exec_output, exec_status = self.executor.run_code(code, p_input, imports)
            
            item_to_store = {"snippet": code, "input": p_input, "imports": imports}
            is_valid_for_storage = False

            if problem_type == "gen_code_i":
                item_to_store["target_output"] = target_output_for_gen_i
                item_to_store["executed_output"] = exec_output
                # Stored if execution is clean. Evaluation vs target_output is separate.
                if "error" not in exec_status.lower() and "timeout" not in exec_status.lower():
                    is_valid_for_storage = True
            elif problem_type == "gen_code_o":
                item_to_store["llm_predicted_output"] = llm_predicted_val
                item_to_store["executed_output"] = exec_output
                if "error" not in exec_status.lower() and "timeout" not in exec_status.lower():
                    is_valid_for_storage = True
            elif problem_type == "gen_code_e":
                item_to_store["llm_predicted_error"] = llm_predicted_val
                item_to_store["actual_error"] = parse_error_message(exec_status) if "error" in exec_status.lower() or "timeout" in exec_status.lower() else "NoError"
                is_valid_for_storage = True # Store all attempts for gen_code_e

            if is_valid_for_storage:
                self.generated_data[problem_type].append(item_to_store)
                return {"success": True, "data": item_to_store, "execution_status": exec_status}
            else:
                return {"success": False, "error": f"Execution failed or invalid for {problem_type}: {exec_status}", "parsed": parsed}

        elif problem_type == "gen_code_f":
            if not original_snippet_for_gen_f:
                return {"success": False, "error": "Original snippet required for gen_code_f."}
            success, parsed = PythonExecutor.parse_llm_inputs_message_for_code_f(llm_content, num_expected_inputs_for_gen_f)
            if not success: return {"success": False, "error": parsed.get("error")}

            gen_inputs, gen_msg = parsed["inputs"], parsed["message"]
            exec_outputs_list = []
            all_ok = True
            for g_input in gen_inputs:
                out, status = self.executor.run_code(original_snippet_for_gen_f, g_input, original_imports_for_gen_f)
                if "error" in status.lower() or "timeout" in status.lower(): all_ok = False; break
                exec_outputs_list.append(out)
            
            if all_ok:
                item_to_store = {
                    "original_snippet": original_snippet_for_gen_f, "imports": original_imports_for_gen_f,
                    "generated_inputs": gen_inputs, "generated_message": gen_msg,
                    "executed_outputs_for_gen_inputs": exec_outputs_list
                }
                self.generated_data[problem_type].append(item_to_store)
                return {"success": True, "data": item_to_store}
            else:
                return {"success": False, "error": "Execution of original snippet failed for one or more generated inputs."}
        else:
            return {"success": False, "error": f"Unknown problem type for generation: {problem_type}"}

    def evaluate_llm_prediction(self, problem_type: str, llm_prediction_content: str,
                                # Args specific to task types
                                code_snippet_for_problem: Optional[str] = None,
                                imports_for_problem: Optional[List[str]] = None,
                                input_for_problem: Optional[str] = None, # For pred_code_o/e
                                output_for_problem: Optional[str] = None, # For pred_code_i (expected output repr)
                                io_pairs_for_pred_f: Optional[List[Tuple[str,str]]] = None # For pred_code_f [(input_repr, output_repr)]
                                ) -> Dict:
        """Evaluates LLM's prediction content against the ground truth for "prediction" tasks."""
        
        # For pred_code_i, _o, _e, the LLM prediction is the content of the ```input/output...``` block
        # For pred_code_f, it's the content of the ```python...``` block
        
        if problem_type == "pred_code_i":
            # llm_prediction_content is the raw predicted input string arguments
            if not code_snippet_for_problem or output_for_problem is None:
                return {"success": False, "error": "Code snippet and target output required for pred_code_i."}
            
            actual_exec_output, exec_status = self.executor.run_code(code_snippet_for_problem, llm_prediction_content, imports_for_problem)
            if "error" in exec_status.lower() or "timeout" in exec_status.lower():
                return {"match": False, "reason": f"Execution failed: {exec_status}"}
            
            # output_for_problem and actual_exec_output are expected to be repr() strings
            match = (actual_exec_output == output_for_problem)
            return {"match": match, "predicted_input": llm_prediction_content, "executed_output": actual_exec_output, "expected_output": output_for_problem}

        elif problem_type == "pred_code_o":
            # llm_prediction_content is the raw predicted output string (repr)
            if not code_snippet_for_problem or input_for_problem is None:
                return {"success": False, "error": "Code snippet and input required for pred_code_o."}
            # We need the true output of code_snippet_for_problem(input_for_problem) to compare.
            # Let's assume output_for_problem IS this true output repr string for simplicity here.
            # Or, we execute to get it. Let's assume it's provided as output_for_problem.
            if output_for_problem is None: # If not provided, execute to get it.
                true_exec_output, status = self.executor.run_code(code_snippet_for_problem, input_for_problem, imports_for_problem)
                if "error" in status.lower() or "timeout" in status.lower():
                    return {"success": False, "error": f"Failed to get true output for comparison: {status}"}
                output_for_problem = true_exec_output

            match = (llm_prediction_content == output_for_problem)
            return {"match": match, "predicted_output": llm_prediction_content, "true_output": output_for_problem}

        elif problem_type == "pred_code_e":
            # llm_prediction_content is the predicted error type string
            if not code_snippet_for_problem or input_for_problem is None:
                return {"success": False, "error": "Code snippet and input required for pred_code_e."}
            _, actual_status = self.executor.run_code(code_snippet_for_problem, input_for_problem, imports_for_problem)
            actual_error = parse_error_message(actual_status) if "error" in actual_status.lower() or "timeout" in actual_status.lower() else "NoError"
            match = (llm_prediction_content.strip().lower() == actual_error.lower())
            return {"match": match, "predicted_error": llm_prediction_content.strip(), "actual_error": actual_error}

        elif problem_type == "pred_code_f":
            success_parse, predicted_func_code = PythonExecutor.parse_llm_predicted_code_for_pred_f(llm_prediction_content)
            if not success_parse: return {"success": False, "error": "Failed to parse predicted function."}
            if not io_pairs_for_pred_f: return {"success": False, "error": "I/O pairs required for pred_code_f."}

            if contains_banned_keywords(predicted_func_code, self.banned_keywords):
                return {"success": False, "error": "Predicted function contains banned keywords."}

            correct_count = 0
            details = []
            for p_input, p_expected_output_repr in io_pairs_for_pred_f:
                exec_out, status = self.executor.run_code(predicted_func_code, p_input, imports_for_problem)
                current_match = False
                if "error" not in status.lower() and "timeout" not in status.lower() and exec_out == p_expected_output_repr:
                    correct_count += 1
                    current_match = True
                details.append({"input":p_input, "expected":p_expected_output_repr, "got":exec_out, "status":status, "match":current_match})
            
            accuracy = correct_count / len(io_pairs_for_pred_f)
            return {"success": True, "accuracy": accuracy, "all_match": accuracy == 1.0, "details": details, "predicted_code": predicted_func_code}
        else:
            return {"success": False, "error": f"Unknown problem type for prediction: {problem_type}"}

    def add_seed_data(self, task_type: str, data_item: Dict):
        """Adds a pre-validated item to the internal data store."""
        if task_type in self.generated_data:
            self.generated_data[task_type].append(data_item)
        else:
            print(f"Warning: Unknown task_type '{task_type}' for seed data.")

    def get_reference_snippets(self, task_type: str, count: int = 1) -> List[Dict]:
        """Retrieves random reference snippets for prompting."""
        import random
        if task_type in self.generated_data and self.generated_data[task_type]:
            return random.sample(self.generated_data[task_type], min(count, len(self.generated_data[task_type])))
        # Fallback seed if no data exists for this type yet.
        if task_type == "gen_code_i":
            return [{'snippet': 'def f(a):\n  return a*2', 'input': '3', 'target_output': '6'}]
        if task_type == "gen_code_o":
            return [{'snippet': 'def f(s):\n  return s.upper()', 'input': "'hello'", 'llm_predicted_output': "'HELLO'"}]
        return []
    
class ContinuousImprovementSystem:
    def __init__(self):
        from src.agents.collaborative.shared_memory import SharedMemory
        from src.agents.learning._agent import SLAIEnv
        from src.agents.learning_agent import LearningAgent
        from src.agents.reasoning_agent import ReasoningAgent
        from src.agents.knowledge.knowledge_sync import KnowledgeSynchronizer
        self.shared_memory = SharedMemory()
        
        # Core Components
        self.reasoner = BasicZeroReasoner(
            executor_timeout=10,
            banned_keywords=DEFAULT_BANNED_KEYWORDS
        )
        
        self.reasoning_agent = ReasoningAgent(
            shared_memory=self.shared_memory,
            rule_validation={'contradiction_threshold': 0.3}
        )
        
        self.learning_agent = LearningAgent(
            env=SLAIEnv(state_dim=256),
            strategy_weights=[0.4, 0.3, 0.2, 0.1]  # RL, DQN, MAML, RSI
        )

        # Coordination Mechanisms
        self.feedback_queue = deque(maxlen=1000)
        self.knowledge_sync = KnowledgeSynchronizer()

    def improvement_cycle(self, iterations=100):
        for _ in range(iterations):
            # 1. Collect Experiences
            transitions = self.collect_interaction_data()
            
            # 2. Parallel Learning
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self.learning_agent.meta_learn, tasks=3),
                    executor.submit(self.reasoning_agent.forward_chaining),
                    executor.submit(self.process_feedback)
                ]
                
            # 3. Knowledge Fusion
            self.knowledge_sync.synchronize(
                reasoning_kb=self.reasoning_agent.knowledge_base,
                learning_data=self.learning_agent.memory,
                reasoner_data=self.reasoner.generated_data
            )
            
            # 4. Validation & Debugging
            self.run_consistency_checks()
            self.log_performance_metrics()

    def execute_safe_reasoning(self, code_snippet):
        validation_steps = [
            self.reasoner.contains_banned_keywords,
            self.reasoning_agent.validate_fact,
            self.learning_agent.validate_action
        ]
        
        for step in validation_steps:
            result = step(code_snippet)
            if not result["valid"]:
                return self.handle_validation_failure(result)
        
        return PythonExecutor().run_code(code_snippet)
    
    def process_feedback(self):
        while self.feedback_queue:
            feedback = self.feedback_queue.popleft()
            
            # Update Learning Agent
            self.learning_agent.learn_from_interaction(
                states=feedback['states'],
                actions=feedback['actions'],
                rewards=self.calculate_adaptive_reward(feedback)
            )
            
            # Update Reasoning KB
            self.reasoning_agent.stream_update(
                new_facts=extract_facts(feedback),
                confidence=calculate_confidence(feedback)
            )
            
            # Update Zero Reasoner
            self.reasoner.add_seed_data(
                task_type="gen_code_f",
                data_item=format_as_code_problem(feedback)
            )

    def run_consistency_checks(self):
        pass

    def log_performance_metrics(self):
        metrics = {
            "reasoning": {
                "inference_speed": self.reasoning_agent.forward_chaining_speed,
                "kb_consistency": self.reasoning_agent.check_consistency()
            },
            "learning": {
                "meta_loss": self.learning_agent.meta_loss,
                "strategy_effectiveness": self.learning_agent.strategy_weights
            },
            "safety": {
                "rejected_code": self.reasoner.security_log,
                "conflict_resolutions": self.reasoning_agent.conflict_count
            }
        }
        
        self.shared_memory.set('performance_metrics', metrics)
        visualize_metrics(metrics)  # Real-time dashboard updates

    def enhance_code_generation(self):
        problematic_cases = self.reasoner.generated_data["gen_code_e"]
        training_data = create_finetuning_dataset(problematic_cases)
        
        self.learning_agent.perform_task({
            "task_type": "code_refinement",
            "dataset": training_data,
            "fine_tune_steps": 1000
        })
        
        updated_rules = self.learning_agent.extract_new_rules()
        self.reasoning_agent.add_rule(updated_rules)

    def adapt_security_rules(self):
        attack_patterns = analyze_security_logs(
            self.reasoner.security_log,
            self.learning_agent.threat_model
        )
        
        new_rules = generate_mitigation_rules(attack_patterns)
        self.reasoner.update_banned_keywords(new_rules)
        
        self.reasoning_agent.validation_engine.update_rules(
            conflict_threshold=calculate_new_threshold(),
            redundancy_margin=0.15
        )

class ResourceGovernor:
    MAX_CPU_USAGE = 85  # %
    MAX_MEMORY = 4096  # MB
    
    def check_resources(self):
        if psutil.cpu_percent() > self.MAX_CPU_USAGE:
            self.learning_agent.pause_training()
            self.reasoning_agent.throttle_inference()
            
        if psutil.virtual_memory().used > self.MAX_MEMORY:
            self.reasoner.clean_cache()
            self.shared_memory.compress_storage()

    def rollback_system(self, checkpoint):
        self.reasoning_agent.load_knowledge(checkpoint['reasoning_kb'])
        self.learning_agent.load_model(checkpoint['learning_state'])
        self.reasoner.generated_data = checkpoint['reasoner_data']
        
        logger.info(f"System rolled back to {checkpoint['timestamp']}")
