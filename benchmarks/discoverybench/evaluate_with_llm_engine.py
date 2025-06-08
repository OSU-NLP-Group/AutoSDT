#!/usr/bin/env python3
import os
import glob
import json
import csv
import re
import threading
import queue
import time
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
# 引入LLMEngine
from engine.base_engine import LLMEngine
from litellm import model_cost
from litellm.utils import trim_messages
from string import Template

MODEL_NAME = "qwen32b_0shot_0516_run8"

# CODER_NAME = "vllm_saves/qwen2.5-coder-14b-instruct/full/sft_expanded_bs8_ep1_nowmup"  
# 14B gpu0
# CODER_NAME = "vllm_saves/qwen2.5-coder-7b-instruct/full/sft_expanded_bs8_ep1_nowmup"
CODER_NAME = "vllm_Qwen/Qwen2.5-Coder-32B-Instruct"
# 4k gpu2
# CODER_NAME = "vllm_saves/qwen2.5-coder-32b-instruct/full/sft_full/checkpoint-500"
# 5.1kwmup gpu3
# CODER_NAME = "vllm_saves/qwen2.5-coder-32b-instruct/full/sft_full_expanded_cardinal_bs8_warmup/checkpoint-644"
DEFAULT_PORT = 8003
USE_PYTHON_CODE_REGEX = True

class UsageCounter:
    def __init__(self):
        self.lock = threading.Lock()
        self.default_code_count = 0
        self.default_hypo_count = 0
        self.default_workflow_count = 0
    
    def increment_default_code(self):
        with self.lock:
            self.default_code_count += 1
    
    def increment_default_hypo(self):
        with self.lock:
            self.default_hypo_count += 1
    
    def increment_default_workflow(self):
        with self.lock:
            self.default_workflow_count += 1
    
    def get_stats(self):
        with self.lock:
            return {
                "default_code_count": self.default_code_count,
                "default_hypo_count": self.default_hypo_count,
                "default_workflow_count": self.default_workflow_count
            }

usage_counter = UsageCounter()

TEST_ROOT = "discoverybench/discoverybench/real/test"
ANSWER_KEY = "discoverybench/eval/answer_key_real.csv"
INFER_LOG_DIR = f"inference_logs_{MODEL_NAME}"
PROGRAM_OUTPUT_DIR = f"program_outputs_{MODEL_NAME}"
PROGRAM_RESULT_DIR = f"program_results_{MODEL_NAME}"
EVAL_RESULT_DIR = f"eval_results_{MODEL_NAME}"

for dir_path in [INFER_LOG_DIR, PROGRAM_OUTPUT_DIR, PROGRAM_RESULT_DIR, EVAL_RESULT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

CODE_SYSTEM_PROMPT = """You are a python programming agent who can generate a python code to answer a query based on one or more datasets. Please use print statement to view the result instead of directly using the variable name."""

CODE_USER_PROMPT = Template("""$query_content""")

# # Qwen-SFT
# CODE_SYSTEM_PROMPT = "You are an expert Python programming assistant that helps scientist users to write high-quality code to solve their tasks.\nGiven a user request, you are expected to write a complete program that accomplishes the requested task.\n"

# # CODE_USER_PROMPT = Template("""You'll be given one or more dataset files, including their paths, names, column descriptions, and finally a scientific query. You must carefully understand the previewed data, strictly follow the paths of these dataset files given in this instruction, and finally write a complete python code that can return certain values from the dataset file(s) to help answer the query. Please pay extreme attention to the data keys and value types and avoid keyerror or valueerrors. Finally, print all the results to console. $query_content""")

# CODE_USER_PROMPT = Template("""Please generate python code to help answer a scientific query based on the provided one or more dataset files. Please strictly follow the paths of the dataset files given in this instruction. Please carefully read and understand the data description and preview before writing the code. Please use print statement to view the result instead of directly using the variable name. $query_content""")

# # added absolute path
# CODE_USER_PROMPT = Template("""Please generate python code to help answer a scientific query based on the provided one or more dataset files. Please strictly follow the absolute paths of the dataset files given in this instruction. Please carefully read and understand the data description and preview before writing the code. Please use print statement to view the result instead of directly using the variable name. $query_content""")

# HYPO_SYSTEM_PROMPT = "You are a scientific hypothesis generator who can interpret data analysis results and generate meaningful scientific hypotheses."
HYPO_SYSTEM_PROMPT = """You are a scientific hypothesis generator who can interpret data analysis results and generate meaningful scientific hypotheses.
Use the following format:

Question: the input question you must answer
Action Input: the input to the action
Observation: the result of the action
Final Answer: the final answer to the original input question. In the final answer, please write down a **scientific hypothesis** in natural language, derived from the provided dataset, clearly stating the context of hypothesis (if any), variables chosen (if any) and relationship between those variables (if any) including any statistical significance. Also generate a **summary of the full workflow** starting from data loading that led to the final answer as **WORKFLOW SUMMARY:**

Here are several example hypotheses:

Scientific Hypothesis Example1: The rate of maximum body length evolution emerged as the most influential factor explaining spatial variation in speciation rates. The relationship is positive with linear coefficient 0.82.

Scientific Hypothesis Example2: Per unit increased ease of immigration reduces 0.1059 unit of the share of offshore employment.

Scientific Hypothesis Example3: Higher time preference associated with higher BMI for 1989 data. BMI is postively related with if person spent more than their saving with a coefficient 0.3596. BMI is also positively correlated with if the savings of a person remained unchaged with a coefficient 0.4858.

Begin!
"""

HYPO_USER_PROMPT = Template("""Question:
$question
Action Input:
$program_code
Observation:
$program_result
""")

DEFAULT_CODE = """import pandas as pd
import os
import glob

# Get dataset paths
dataset_paths = os.environ.get("DATASET_PATHS", "").split(";")
print("Available datasets:", dataset_paths)

# Try to load datasets
for path in dataset_paths:
    if path and os.path.exists(path):
        try:
            df = pd.read_csv(path)
            print(f"Successfully loaded {path}")
            print(f"Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"Sample data:\\n{df.head(3)}")
            print("-" * 50)
        except Exception as e:
            print(f"Failed to load {path}: {e}")

# Basic statistical analysis
if 'df' in locals():
    print("\\nBasic statistics:")
    try:
        print(df.describe())
    except Exception as e:
        print(f"Failed to generate statistics: {e}")
"""

DEFAULT_HYPOTHESIS = """Based on the analyzed datasets, I observe patterns indicating relationships between various variables within the data.

SCIENTIFIC HYPOTHESIS: There is a significant correlation between key variables in the dataset, with certain factors showing stronger influence on the outcome variables than others. This relationship suggests a causal mechanism that could explain the observed patterns in the data.

WORKFLOW SUMMARY:
1. Loaded the provided datasets and examined their structure
2. Performed exploratory data analysis to understand the distribution of variables
3. Attempted to identify relationships between key variables
4. Generated a hypothesis based on the observed patterns in the data
5. The analysis suggests further investigation would be valuable to confirm these initial findings"""

def load_answer_key(path):
    answer_dict = {}
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # 以 (dataset, metadataid, query_id) 为 key
            key = (row['dataset'], row['metadataid'], row['query_id'])
            answer_dict[key] = row['gold_hypo']
    return answer_dict

sci_hypo_pattern = re.compile(
    r"^#*\s*\*{0,2}\s*SCI(ENTIFIC)?\s+HYPOTHESIS\s*\*{0,2}\s*:?", re.IGNORECASE
)
workflow_pattern = re.compile(
    r"^#*\s*\*{0,2}\s*WORKFLOW\s+SUMMARY\s*\*{0,2}\s*:?", re.IGNORECASE
)

def extract_hypothesis(log_content):
    """Extract scientific hypothesis part from log content"""
    try:
        log_data = json.loads(log_content)
        if "message" in log_data:
            message_data = json.loads(log_data["message"])
            if "response" in message_data:
                response_text = message_data["response"]
                
                parts = response_text.split("\n\n")
                if parts and len(parts) > 0:
                    first_part = parts[0]
                    
                    hypothesis_match = re.search(
                        r"(?:Final Answer:\s*)?(?:\*{0,4}|#{0,4})\s*(?:Scientific?|Scienc?e)\s*(?:Hypothes[ie]s?)(?:\*{0,4}|#{0,4})\s*:?\s*(.*?)(?=$|\n\n|\*\*WORKFLOW|\#\#\#\s*WORKFLOW)",
                        first_part,
                        re.IGNORECASE | re.DOTALL
                    )
                    
                    if hypothesis_match:
                        return hypothesis_match.group(1).strip()
        
        return ""
        
    except Exception as e:
        print(f"Failed to extract hypothesis: {e}")
        return ""
    
def extract_hypo_and_workflow(log_file):
    gen_hypo = ""
    gen_workflow = ""
    
    if not os.path.exists(log_file):
        return DEFAULT_HYPOTHESIS.split("SCIENTIFIC HYPOTHESIS:")[1].split("WORKFLOW SUMMARY:")[0].strip(), ""
    
    with open(log_file, "r") as lf:
        content = lf.read()
        
        if content.strip().startswith('{"timestamp":'):
            try:
                log_data = json.loads(content)
                if "message" in log_data:
                    message_data = json.loads(log_data["message"])
                    if "response" in message_data:
                        response_text = message_data["response"]
                        
                        parts = response_text.split("\n\n")
                        if parts and len(parts) > 0:
                            first_part = parts[0]
                            
                            hypothesis_match = re.search(
                                r"(?:Final Answer:\s*)?(?:\*{0,4}|#{0,4})\s*(?:Scientific?|Scienc?e)\s*(?:Hypothes[ie]s?)(?:\*{0,4}|#{0,4})\s*:?\s*(.*?)(?=$|\n\n|\*\*WORKFLOW|\#\#\#\s*WORKFLOW)",
                                first_part,
                                re.IGNORECASE | re.DOTALL
                            )
                            
                            if hypothesis_match:
                                gen_hypo = hypothesis_match.group(1).strip()
                            
                            if not gen_hypo:
                                hypothesis_match = re.search(
                                    r"(?:Final Answer:\s*)?(?:\*{0,4}|#{0,4})\s*(?:Scientific?|Scienc?e)\s*(?:Hypothes[ie]s?)(?:\*{0,4}|#{0,4})\s*:?\s*(.*?)(?=\*\*WORKFLOW|\#\#\#\s*WORKFLOW|$)",
                                    response_text,
                                    re.IGNORECASE | re.DOTALL
                                )
                                if hypothesis_match:
                                    gen_hypo = hypothesis_match.group(1).strip()
            except Exception as e:
                print(f"Error parsing JSON: {e}")
        
        if not gen_hypo:
            hypothesis_match = re.search(
                r"(?:Final Answer:\s*)?(?:\*{0,4}|#{0,4})\s*(?:Scientific?|Scienc?e)\s*(?:Hypothes[ie]s?)(?:\*{0,4}|#{0,4})\s*:?\s*(.*?)(?=\*\*WORKFLOW|\#\#\#\s*WORKFLOW|WORKFLOW SUMMARY|$)",
                content,
                re.IGNORECASE | re.DOTALL
            )
            if hypothesis_match:
                gen_hypo = hypothesis_match.group(1).strip()
    
    if not gen_hypo:
        usage_counter.increment_default_hypo()
        gen_hypo = DEFAULT_HYPOTHESIS.split("SCIENTIFIC HYPOTHESIS:")[1].split("WORKFLOW SUMMARY:")[0].strip()
    
    return gen_hypo, ""

def extract_hypo_and_workflow_old(log_file):
    gen_hypo = ""
    gen_workflow = ""
    
    if not os.path.exists(log_file):
        return DEFAULT_HYPOTHESIS.split("SCIENTIFIC HYPOTHESIS:")[1].split("WORKFLOW SUMMARY:")[0].strip(), \
               DEFAULT_HYPOTHESIS.split("WORKFLOW SUMMARY:")[1].strip()
    
    with open(log_file, "r") as lf:
        content = lf.read()
        
        if content.strip().startswith('{"timestamp":'):
            try:
                log_data = json.loads(content)
                if "message" in log_data:
                    message_data = json.loads(log_data["message"])
                    if "response" in message_data:
                        response_text = message_data["response"]
                        lines = response_text.split('\n')
                        
                        sections = []
                        current_section = []
                        current_type = None
                        
                        for line in lines:
                            line = line.strip()
                            if not line:
                                continue
                                
                            if sci_hypo_pattern.match(line):
                                if current_section:
                                    sections.append((current_type, current_section))
                                current_type = "hypo"
                                current_section = [sci_hypo_pattern.sub("", line, count=1).strip()]
                            elif workflow_pattern.match(line):
                                if current_section:
                                    sections.append((current_type, current_section))
                                current_type = "workflow"
                                current_section = [workflow_pattern.sub("", line, count=1).strip()]
                            elif current_type:
                                if line.startswith("```") or line.startswith("###"):
                                    sections.append((current_type, current_section))
                                    current_type = None
                                    current_section = []
                                else:
                                    current_section.append(line)
                        
                        if current_section:
                            sections.append((current_type, current_section))
                        
                        for section_type, section_lines in sections:
                            if section_type == "hypo":
                                gen_hypo = " ".join(section_lines).strip()
                            elif section_type == "workflow":
                                gen_workflow = "\n".join(section_lines).strip()
                        
                        if gen_hypo and gen_workflow:
                            return gen_hypo, gen_workflow
            except Exception as e:
                print(f"Error parsing JSON: {e}")
        
        lines = content.split('\n')
        hypo_lines = []
        in_hypo = False
        for line in lines:
            if sci_hypo_pattern.match(line.strip()):
                in_hypo = True
                content = sci_hypo_pattern.sub("", line, count=1)
                if content.strip():
                    hypo_lines.append(content.strip())
                continue
            if in_hypo:
                if (line.strip() == "" or
                    workflow_pattern.match(line.strip()) or
                    line.strip().startswith("###") or line.strip().startswith(">")):
                    break
                hypo_lines.append(line.strip())
        gen_hypo = " ".join(hypo_lines).strip()
        
        workflow_lines = []
        in_workflow = False
        for line in lines:
            if workflow_pattern.match(line.strip()):
                in_workflow = True
                content = workflow_pattern.sub("", line, count=1)
                if content.strip():
                    workflow_lines.append(content.strip())
                continue
            if in_workflow:
                if (line.strip() == "" or
                    sci_hypo_pattern.match(line.strip()) or
                    line.strip().startswith("###") or line.strip().startswith(">")):
                    break
                workflow_lines.append(line.strip())
        gen_workflow = "\n".join(workflow_lines).strip()
    
    if not gen_hypo:
        usage_counter.increment_default_hypo()
        gen_hypo = DEFAULT_HYPOTHESIS.split("SCIENTIFIC HYPOTHESIS:")[1].split("WORKFLOW SUMMARY:")[0].strip()
    if not gen_workflow:
        usage_counter.increment_default_workflow()
        gen_workflow = DEFAULT_HYPOTHESIS.split("WORKFLOW SUMMARY:")[1].strip()
        
    # return gen_hypo, gen_workflow
    return gen_hypo, ""

def extract_code_from_response(response_text):
    """Extract Python code from response"""
    if USE_PYTHON_CODE_REGEX:
        code_pattern = re.compile(r"```(?:python)?(.*?)```", re.DOTALL)
        matches = code_pattern.findall(response_text)
        if matches:
            return matches[0].strip()
        else:
            return DEFAULT_CODE
    else:
        return response_text.strip()

def execute_python_program(program_file, dataset_paths):
    """Execute Python program and return result"""
    result = ""
    try:
        env = os.environ.copy()
        env["DATASET_PATHS"] = ";".join(dataset_paths)
        
        result = subprocess.check_output(
            ["python", program_file], 
            stderr=subprocess.STDOUT,
            env=env,
            universal_newlines=True,
            timeout=300
        )
    except subprocess.CalledProcessError as e:
        result = f"Execution error (return code {e.returncode}):\n{e.output}"
    except subprocess.TimeoutExpired:
        result = "Execution timeout (over 5 minutes)"
    except Exception as e:
        result = f"Execution exception: {str(e)}"
    
    return result

class CodeGenerator:
    def __init__(self, model_name=CODER_NAME, max_workers=10, context_cutoff=28000, port=DEFAULT_PORT):
        self.model_name = model_name
        self.max_workers = max_workers
        self.context_cutoff = context_cutoff
        self.port = port
        self.llm_engine = LLMEngine(model_name, api_key=os.environ.get('AZURE_OPENAI_KEY', ''), api_version=os.environ.get('AZURE_OPENAI_API_VERSION', ''), azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT', ''), port=port)
        self.llm_cost = model_cost[model_name] if "vllm" not in model_name and "azure" not in model_name else model_cost["gpt-4o"]
        self.task_queue = queue.Queue()
        self.results = {}
        
    def create_system_prompt(self):
        """Create system prompt"""
        return {"role": "system", "content": CODE_SYSTEM_PROMPT}
    
    def create_user_prompt(self, dataset_paths, query_content):
        """Create user prompt"""
        return {"role": "user", "content": CODE_USER_PROMPT.substitute(query_content=query_content)}
    
    def generate_code(self, task):
        """Generate code for a single task"""
        key, meta_path, dataset_paths, query_content, log_file = task
        
        dataset_name, meta_id, query_id = key
        program_file = f"{PROGRAM_OUTPUT_DIR}/{dataset_name}_{meta_id}_{query_id}.py"
        
        if os.path.exists(program_file) and os.path.exists(log_file):
            print(f"Program file already exists: {program_file}, skipping code generation")
            try:
                with open(log_file, "r") as f:
                    log_data = json.loads(f.read())
                    message_data = json.loads(log_data.get("message", "{}"))
                    response = message_data.get("response", "")
                    cost = 0.0
                with open(program_file, "r") as f:
                    program_code = f.read()
                return key, response, cost, program_file, program_code
            except Exception as e:
                print(f"Error reading existing log file: {str(e)}")
        
        system_message = self.create_system_prompt()
        user_message = self.create_user_prompt(dataset_paths, query_content)
        messages = [system_message, user_message]
        try:
            response, prompt_tokens, completion_tokens = self.llm_engine.respond(
                messages, 
                temperature=0.2,
                top_p=0.95,
                max_tokens=1024,
            )
            
            cost = (
                self.llm_cost["input_cost_per_token"] * prompt_tokens +
                self.llm_cost["output_cost_per_token"] * completion_tokens
            )

            log_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "message": json.dumps({"response": response})
            }
            
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(log_data))
                
            program_code = extract_code_from_response(response)
            with open(program_file, "w", encoding="utf-8") as f:
                f.write(program_code)
                    
            return key, response, cost, program_file, program_code
            
        except Exception as e:
            print(f"Error generating code for task {key}: {str(e)}")
            with open(program_file, "w", encoding="utf-8") as f:
                f.write(DEFAULT_CODE)
            
            log_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "message": json.dumps({"response": "ERROR: " + str(e) + "\n\nDefault code was used."})
            }
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(log_data))
                
            return key, str(e), 0.0, program_file, DEFAULT_CODE
    
    def worker(self):
        """Worker thread processing function"""
        while True:
            try:
                task = self.task_queue.get(block=False)
                if task is None:
                    break
                
                key, response, cost, program_file, program_code = self.generate_code(task)
                self.results[key] = (response, cost, program_file, program_code)
                
                print(f"Finished code generation task: {key}")
                self.task_queue.task_done()
                
            except queue.Empty:
                break
            except Exception as e:
                print(f"Error generating code for task {key}: {str(e)}")
                self.task_queue.task_done()
    
    def run_generation(self, metadata_tasks):
        """Run code generation program"""
        for task in metadata_tasks:
            self.task_queue.put(task)
        
        progress_bar = tqdm(total=len(metadata_tasks), desc="Code generation progress", unit="task")
        
        def update_progress():
            last_size = self.task_queue.qsize()
            while not self.task_queue.empty():
                current_size = self.task_queue.qsize()
                if current_size < last_size:
                    progress_bar.update(last_size - current_size)
                    last_size = current_size
                time.sleep(0.1)
        
        progress_thread = threading.Thread(target=update_progress)
        progress_thread.daemon = True
        progress_thread.start()
        
        threads = []
        for _ in range(min(self.max_workers, len(metadata_tasks))):
            thread = threading.Thread(target=self.worker)
            thread.start()
            threads.append(thread)
        
        self.task_queue.join()
        
        for _ in range(len(threads)):
            self.task_queue.put(None)
        for thread in threads:
            thread.join()
        
        progress_bar.n = len(metadata_tasks)
        progress_bar.refresh()
        progress_bar.close()
    
        return self.results

class HypothesisGenerator:
    def __init__(self, model_name=CODER_NAME, max_workers=10, context_cutoff=28000, port=DEFAULT_PORT):
        self.model_name = model_name
        self.max_workers = max_workers
        self.context_cutoff = context_cutoff
        self.port = port
        self.llm_engine = LLMEngine(model_name, api_key=os.environ.get('AZURE_OPENAI_KEY', ''), api_version=os.environ.get('AZURE_OPENAI_API_VERSION', ''), azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT', ''), port=port)
        self.llm_cost = model_cost[model_name] if "vllm" not in model_name and "azure" not in model_name else model_cost["gpt-4o"]
        self.task_queue = queue.Queue()
        self.results = {}
        
    def create_system_prompt(self):
        """Create system prompt"""
        return {"role": "system", "content": HYPO_SYSTEM_PROMPT}
    
    def create_user_prompt(self, query_content, program_code, program_result):
        """Create user prompt, including program execution result"""
        prompt = HYPO_USER_PROMPT.substitute(question=query_content, program_code=program_code, program_result=program_result)
        # prompt = f"{query_content}\n\nProgram execution result:\n{program_result}\n\n"
        # prompt += "In the final answer, please write down a scientific hypothesis as SCIENTIFIC HYPOTHESIS: in natural language, derived from the provided dataset, clearly stating the context of hypothesis (if any), variables chosen (if any) and relationship between those variables (if any) including any statistical significance. Also generate a summary of the full workflow starting from data loading that led to the final answer as WORKFLOW SUMMARY:"
        return {"role": "user", "content": prompt}
    
    def generate_hypothesis(self, task):
        """Generate scientific hypothesis and workflow for a single task"""
        key, meta_path, query_content, program_code, program_result, log_file = task
        
        if os.path.exists(log_file):
            print(f"Hypothesis log file already exists: {log_file}, skipping hypothesis generation")
            try:
                with open(log_file, "r") as f:
                    log_data = json.loads(f.read())
                    message_data = json.loads(log_data.get("message", "{}"))
                    response = message_data.get("response", DEFAULT_HYPOTHESIS)
                    cost = 0.0
                return key, response, cost
            except Exception as e:
                print(f"Error reading existing hypothesis log file: {str(e)}")
        
        system_message = self.create_system_prompt()
        user_message = self.create_user_prompt(query_content, program_code, program_result)
        
        messages = [system_message, user_message]
        
        try:
            response, prompt_tokens, completion_tokens = self.llm_engine.respond(
                messages, 
                temperature=0.2,
                top_p=0.95,
                max_tokens=1024
            )
            
            cost = (
                self.llm_cost["input_cost_per_token"] * prompt_tokens +
                self.llm_cost["output_cost_per_token"] * completion_tokens
            )
            
            log_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "message": json.dumps({"response": response})
            }
            
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(log_data))
                    
            return key, response, cost
            
        except Exception as e:
            print(f"Error generating hypothesis for task {key}: {str(e)}")
            
            log_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "message": json.dumps({"response": "ERROR: " + str(e) + "\n\n" + DEFAULT_HYPOTHESIS})
            }
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(log_data))
                
            return key, DEFAULT_HYPOTHESIS, 0.0
    
    def worker(self):
        """Worker thread processing function"""
        while True:
            try:
                task = self.task_queue.get(block=False)
                if task is None:
                    break
                
                key, response, cost = self.generate_hypothesis(task)
                self.results[key] = (response, cost)
                
                print(f"Finished hypothesis generation task: {key}")
                self.task_queue.task_done()
                
            except queue.Empty:
                break
            except Exception as e:
                print(f"Error at hypothesis generation task {key}: {str(e)}")
                self.task_queue.task_done()
    
    def run_generation(self, hypothesis_tasks):
        """Run hypothesis generation program"""
        for task in hypothesis_tasks:
            self.task_queue.put(task)
        
        progress_bar = tqdm(total=len(hypothesis_tasks), desc="hypothesis generation progress", unit="task")
        
        def update_progress():
            last_size = self.task_queue.qsize()
            while not self.task_queue.empty():
                current_size = self.task_queue.qsize()
                if current_size < last_size:
                    progress_bar.update(last_size - current_size)
                    last_size = current_size
                time.sleep(0.1)
        
        progress_thread = threading.Thread(target=update_progress)
        progress_thread.daemon = True
        progress_thread.start()
        
        threads = []
        for _ in range(min(self.max_workers, len(hypothesis_tasks))):
            thread = threading.Thread(target=self.worker)
            thread.start()
            threads.append(thread)
        
        self.task_queue.join()
        
        for _ in range(len(threads)):
            self.task_queue.put(None)
        for thread in threads:
            thread.join()
        
        progress_bar.n = len(hypothesis_tasks)
        progress_bar.refresh()
        progress_bar.close()
        
        return self.results

class Evaluator:
    def __init__(self, max_workers=10):
        self.max_workers = max_workers
        self.task_queue = queue.Queue()
        self.results = {}
        
    def evaluate_single_task(self, task):
        """Evaluate a single task"""
        key, meta_path, gold_hypo, gen_hypo, gold_workflow, gen_workflow, eval_out, query_text, cost = task
        
        if os.path.exists(eval_out):
            print(f"Evaluation result file already exists: {eval_out}, skipping evaluation")
            try:
                with open(eval_out, "r") as ef:
                    eval_result = json.load(ef)
                return key, eval_result
            except Exception as e:
                print(f"Error reading existing evaluation result file: {str(e)}, will re-evaluate")
                if os.path.exists(eval_out):
                    os.remove(eval_out)
        
        if os.path.exists(eval_out):
            os.remove(eval_out)
        
        eval_cmd = [
            'python3', 'discovery_eval.py',
            '--gold_hypo', gold_hypo,
            '--pred_hypo', gen_hypo,
            # '--gold_workflow', gold_workflow,
            '--gold_workflow', "",
            # '--pred_workflow', gen_workflow,
            '--pred_workflow', "",
            '--metadata_path', meta_path,
            '--metadata_type', 'real',
            '--eval_output_path', eval_out,
            query_text
        ]
        
        print(f"Evaluating task {key} ...")
        try:
            subprocess.run(eval_cmd, check=True)
            
            # Load evaluation result
            with open(eval_out, "r") as ef:
                eval_result = json.load(ef)
                
            # Add cost information
            eval_result["cost"] = cost
            
            # Update evaluation result file
            with open(eval_out, "w") as ef:
                json.dump(eval_result, ef, indent=2)
                
            return key, eval_result
            
        except Exception as e:
            print(f"Error evaluating task {key}: {str(e)}")
            
            default_result = {
                "final_score": 0.0,
                "hypo_score": 0.0,
                "workflow_score": 0.0,
                "gold_hypo": gold_hypo,
                "pred_hypo": gen_hypo,
                "gold_workflow": gold_workflow,
                "pred_workflow": gen_workflow,
                "cost": cost,
                "error": str(e)
            }
            
            with open(eval_out, "w") as ef:
                json.dump(default_result, ef, indent=2)
                
            return key, default_result
    
    def worker(self):
        """Worker thread processing function"""
        while True:
            try:
                task = self.task_queue.get(block=False)
                if task is None:
                    break
                
                key, result = self.evaluate_single_task(task)
                self.results[key] = result
                
                print(f"Finished evaluation task: {key}")
                self.task_queue.task_done()
                
            except queue.Empty:
                break
            except Exception as e:
                print(f"Error at evaluation task {key}: {str(e)}")
                self.task_queue.task_done()
    
    def run_evaluation(self, eval_tasks):
        """Run evaluation program"""
        for task in eval_tasks:
            self.task_queue.put(task)
        
        progress_bar = tqdm(total=len(eval_tasks), desc="hypothesis generation progress", unit="task")

        def update_progress():
            last_size = self.task_queue.qsize()
            while not self.task_queue.empty():
                current_size = self.task_queue.qsize()
                if current_size < last_size:
                    progress_bar.update(last_size - current_size)
                    last_size = current_size
                time.sleep(0.1)
        
        progress_thread = threading.Thread(target=update_progress)
        progress_thread.daemon = True
        progress_thread.start()
        
        threads = []
        for _ in range(min(self.max_workers, len(eval_tasks))):
            thread = threading.Thread(target=self.worker)
            thread.start()
            threads.append(thread)
        
        self.task_queue.join()
        
        for _ in range(len(threads)):
            self.task_queue.put(None)
        for thread in threads:
            thread.join()
        
        progress_bar.n = len(eval_tasks)
        progress_bar.refresh()
        progress_bar.close()
        
        return self.results

def get_dv_query_for_real(meta_path, metadata, provide_domain_knowledge=False, provide_workflow_tags=False, nl_query=""):
    dataset_meta = ""
    for dataset_metadata in metadata['datasets']:
        dataset_name = os.path.join(os.path.dirname(meta_path), dataset_metadata['name'])
        dataset_meta += "Dataset name: " + dataset_name
        # dataset_meta += "Dataset name: " + dataset_metadata['name']
        dataset_meta += "Dataset description: " + dataset_metadata['description']
        dataset_meta += "\nBrief description of columns: "
        for col in dataset_metadata['columns']['raw']:
            dataset_meta += col['name'] + ": " + col['description'] + ", "

    query_to_dv = dataset_meta

    for int_hypo in metadata['hypotheses']['intermediate']:
        query_to_dv += int_hypo['text'] + ",\n "

    query_to_dv += f"\nQuery: {nl_query}"

    if provide_domain_knowledge and 'domain_knowledge' in metadata:
        query_to_dv += "\nAdditionally, we provide some hints that might be useful to solve the task. Domain Knowledge: \n" + metadata['domain_knowledge']+".\n"

    if provide_workflow_tags and 'workflow_tags' in metadata:
        query_to_dv += "The meta tags are: " + metadata['workflow_tags'] + ".\n"


    # print(f"Code generation query: {query_to_dv}")
    return query_to_dv, dataset_meta

def get_dv_query_for_real_hypo(meta_path, metadata, provide_domain_knowledge=False, provide_workflow_tags=False, nl_query=""):
    dataset_meta = ""
    for dataset_metadata in metadata['datasets']:
        dataset_name = os.path.join(os.path.dirname(meta_path), dataset_metadata['name'])
        dataset_meta += "Dataset name: " + dataset_name
        # dataset_meta += "Dataset name: " + dataset_metadata['name']
        dataset_meta += "Dataset description: " + dataset_metadata['description']
        dataset_meta += "\nBrief description of columns: "
        for col in dataset_metadata['columns']['raw']:
            dataset_meta += col['name'] + ": " + col['description'] + ", "

    query_to_dv = dataset_meta

    for int_hypo in metadata['hypotheses']['intermediate']:
        query_to_dv += int_hypo['text'] + ",\n "

    query_to_dv += f"\nQuery: {nl_query}"

    if provide_domain_knowledge and 'domain_knowledge' in metadata:
        query_to_dv += "\nAdditionally, we provide some hints that might be useful to solve the task. Domain Knowledge: \n" + metadata['domain_knowledge']+".\n"

    if provide_workflow_tags and 'workflow_tags' in metadata:
        query_to_dv += "The meta tags are: " + metadata['workflow_tags'] + ".\n"

    # print(f"Hypothesis generation query: {query_to_dv}")
    return query_to_dv, dataset_meta

# detect existing code, hypothesis, and evaluation results
def check_existing_results():
    """Check existing code, hypothesis, and evaluation results"""
    existing_code = {}
    existing_hypo = {}
    existing_eval = {}
    
    # Detect existing code files
    for code_file in glob.glob(f"{PROGRAM_OUTPUT_DIR}/*.py"):
        file_name = os.path.basename(code_file)
        parts = file_name.replace('.py', '').split('_')
        if len(parts) >= 3:
            dataset_name = parts[0]
            meta_id = parts[1]
            query_id = parts[2]
            key = (dataset_name, meta_id, query_id)
            existing_code[key] = code_file
    
    # Detect existing hypothesis logs
    for hypo_log in glob.glob(f"{INFER_LOG_DIR}/*_hypo.log"):
        file_name = os.path.basename(hypo_log)
        parts = file_name.replace('_hypo.log', '').split('_')
        if len(parts) >= 3:
            dataset_name = parts[0]
            meta_id = parts[1]
            query_id = parts[2]
            key = (dataset_name, meta_id, query_id)
            existing_hypo[key] = hypo_log
    
    # Detect existing evaluation results
    for eval_file in glob.glob(f"{EVAL_RESULT_DIR}/eval_*.json"):
        file_name = os.path.basename(eval_file)
        parts = file_name.replace('eval_', '').replace('.json', '').split('_')
        if len(parts) >= 3:
            dataset_name = parts[0]
            meta_id = parts[1]
            query_id = parts[2]
            key = (dataset_name, meta_id, query_id)
            existing_eval[key] = eval_file
    
    print(f"Detected {len(existing_code)} existing code files")
    print(f"Detected {len(existing_hypo)} existing hypothesis logs")
    print(f"Detected {len(existing_eval)} existing evaluation results")
    
    return existing_code, existing_hypo, existing_eval

def main():
    answer_dict = load_answer_key(ANSWER_KEY)
    
    existing_code, existing_hypo, existing_eval = check_existing_results()
    
    code_tasks = []
    all_tasks = []
    
    for dataset_dir in glob.glob(f"{TEST_ROOT}/*"):
        if not os.path.isdir(dataset_dir):
            continue
        dataset_name = os.path.basename(dataset_dir)
        for meta_path in glob.glob(f"{dataset_dir}/metadata_*.json"):
            meta_id = Path(meta_path).stem.split("_")[-1]
            with open(meta_path, "r") as f:
                meta = json.load(f)
            
            # # Get dataset paths
            # dataset_paths = [
            #     f"{dataset_dir}/{dataset['name']}"
            #     for dataset in meta.get("datasets", [])
            # ]
            dataset_paths = [
                os.path.abspath(os.path.join(dataset_dir, dataset["name"]))
                for dataset in meta.get("datasets", [])
            ]
            
            # Compatible with queries structure
            queries = []
            if "queries" in meta and isinstance(meta["queries"], list):
                for qlist in meta["queries"]:
                    for q in qlist:
                        queries.append(q)
            
            # Traverse each query
            for q in queries:
                query_id = str(q.get("qid", 0))
                query_text = q.get("question", "")
                
                # All task keys
                key = (dataset_name, str(meta_id), query_id)
                all_tasks.append(key)
                
                # If there is already an evaluation result, skip the entire task
                if key in existing_eval:
                    print(f"task {key} has evaluation result, skipping")
                    continue
                
                # gold_hypo
                # gold_hypo
                gold_hypo = answer_dict.get(key, "")
                if not gold_hypo:
                    print(f"not found gold_hypo: {key}, using default value")
                
                # Create query content for code generation
                query_to_dv, dataset_meta = get_dv_query_for_real(
                    meta_path, 
                    meta, 
                    provide_domain_knowledge=True,
                    provide_workflow_tags=False, 
                    nl_query=query_text
                )
                
                # Create log file path
                code_log_file = f"{INFER_LOG_DIR}/{dataset_name}_{meta_id}_{query_id}_code.log"
                
                # If there is already a code file but no evaluation result, still add to the task list
                if key in existing_code and not key in existing_eval:
                    print(f"Task {key} has code file but missing evaluation result, adding to task list")
                
                # Add to code generation task list
                code_tasks.append((key, meta_path, dataset_paths, query_to_dv, code_log_file))
    
    print(f"Collected {len(code_tasks)} code generation tasks")
    # Step 1: Generate code
    code_generator = CodeGenerator(model_name=CODER_NAME, max_workers=10, port=DEFAULT_PORT)
    code_results = code_generator.run_generation(code_tasks)
    # import pdb; pdb.set_trace()
    print("Code generation completed, preparing to execute program and generate hypothesis...")
    
    # Step 2: Execute code, then generate hypothesis
    hypothesis_tasks = []
    
    for key, (response, cost, program_file, program_code) in code_results.items():
        dataset_name, meta_id, query_id = key
        meta_path = f"{TEST_ROOT}/{dataset_name}/metadata_{meta_id}.json"
        
        # Ensure program file exists
        if not program_file or not os.path.exists(program_file):
            print(f"Program file does not exist: {key}, using default code")
            program_file = f"{PROGRAM_OUTPUT_DIR}/{dataset_name}_{meta_id}_{query_id}.py"
            with open(program_file, "w", encoding="utf-8") as f:
                f.write(DEFAULT_CODE)
            
        # Read metadata
        with open(meta_path, "r") as f:
            meta = json.load(f)
        
        # Get query text
        query_text = ""
        if "queries" in meta and isinstance(meta["queries"], list):
            for qlist in meta["queries"]:
                for q in qlist:
                    if str(q.get("qid", 0)) == query_id:
                        query_text = q.get("question", "")
                        break
        
        # Get absolute dataset paths
        dataset_paths = [
            os.path.abspath(os.path.join(dataset_dir, dataset["name"]))
            for dataset in meta.get("datasets", [])
        ]
        
        # Check if execution result file already exists
        result_file = f"{PROGRAM_RESULT_DIR}/{dataset_name}_{meta_id}_{query_id}_result.txt"
        if os.path.exists(result_file):
            print(f"Execution result file already exists: {result_file}, skipping execution")
            with open(result_file, "r", encoding="utf-8") as f:
                program_result = f.read()
        else:
            # Execute program
            print(f"Executing program: {program_file}")
            program_result = execute_python_program(program_file, dataset_paths)
            
            # Save program execution result
            with open(result_file, "w", encoding="utf-8") as f:
                f.write(program_result)
        
        # Create query content for hypothesis generation
        query_to_dv_hypo, _ = get_dv_query_for_real_hypo(
            meta_path, 
            meta, 
            provide_domain_knowledge=True,
            provide_workflow_tags=True, 
            nl_query=query_text
        )
        
        # Create log file path
        hypo_log_file = f"{INFER_LOG_DIR}/{dataset_name}_{meta_id}_{query_id}_hypo.log"
        
        # Add to hypothesis generation task list
        hypothesis_tasks.append((key, meta_path, query_to_dv_hypo, program_code, program_result, hypo_log_file))
    
    print(f"Collected {len(hypothesis_tasks)} hypothesis generation tasks")
    # import pdb; pdb.set_trace()
    # Generate hypothesis
    hypo_generator = HypothesisGenerator(model_name="azure_gpt-4o", max_workers=10, port=8000)
    hypo_results = hypo_generator.run_generation(hypothesis_tasks)
    
    # Build all hypothesis result dict, including previously existing ones
    all_hypo_results = {}
    # Add newly generated results
    for key, (response, cost) in hypo_results.items():
        all_hypo_results[key] = (response, cost)
    
    # Add existing results
    for key in all_tasks:
        if key not in all_hypo_results and key in existing_hypo:
            dataset_name, meta_id, query_id = key
            hypo_log_file = f"{INFER_LOG_DIR}/{dataset_name}_{meta_id}_{query_id}_hypo.log"
            if os.path.exists(hypo_log_file):
                try:
                    with open(hypo_log_file, "r") as f:
                        log_data = json.loads(f.read())
                        message_data = json.loads(log_data.get("message", "{}"))
                        response = message_data.get("response", DEFAULT_HYPOTHESIS)
                        all_hypo_results[key] = (response, 0.0)
                except Exception as e:
                    print(f"Load existing hypothesis log file failed: {str(e)}")
                    all_hypo_results[key] = (DEFAULT_HYPOTHESIS, 0.0)
    
    # Ensure all tasks have results before processing hypothesis generation results
    # Check if any tasks are missing
    missing_tasks = set(all_tasks) - set(all_hypo_results.keys())
    if missing_tasks:
        print(f"Found {len(missing_tasks)} tasks that did not generate hypothesis, using default hypothesis")
        for key in missing_tasks:
            dataset_name, meta_id, query_id = key
            # Create log file
            hypo_log_file = f"{INFER_LOG_DIR}/{dataset_name}_{meta_id}_{query_id}_hypo.log"
            # Save default hypothesis log
            log_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "message": json.dumps({"response": DEFAULT_HYPOTHESIS})
            }
            with open(hypo_log_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(log_data))
            # Add to results
            all_hypo_results[key] = (DEFAULT_HYPOTHESIS, 0.0)
    
    # Collect evaluation tasks
    eval_tasks = []
    
    for key in all_tasks:
        # If there is already an evaluation result, skip
        if key in existing_eval:
            continue
            
        dataset_name, meta_id, query_id = key
        
        # Get or create hypothesis log file
        hypo_log_file = f"{INFER_LOG_DIR}/{dataset_name}_{meta_id}_{query_id}_hypo.log"
        if not os.path.exists(hypo_log_file):
            # Ensure log file exists
            log_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "message": json.dumps({"response": DEFAULT_HYPOTHESIS})
            }
            with open(hypo_log_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(log_data))
        
        # Extract hypothesis and workflow
        gen_hypo, gen_workflow = extract_hypo_and_workflow(hypo_log_file)
        # If the extracted hypothesis is too short, use the default hypothesis
        if not gen_hypo or len(gen_hypo) < 50:
            gen_hypo = DEFAULT_HYPOTHESIS.split("SCIENTIFIC HYPOTHESIS:")[1].split("WORKFLOW SUMMARY:")[0].strip()
        if not gen_workflow or len(gen_workflow) < 50:
            gen_workflow = DEFAULT_HYPOTHESIS.split("WORKFLOW SUMMARY:")[1].strip()
        gen_workflow = ""
        # Evaluation output file
        eval_out = f"{EVAL_RESULT_DIR}/eval_{dataset_name}_{meta_id}_{query_id}.json"
        
        # Find the corresponding gold_hypo
        gold_hypo = answer_dict.get(key, "")
        if not gold_hypo:
            print(f"Warning: No gold_hypo found, key: {key}")
            gold_hypo = "default gold hypothesis"
        
        # Read metadata to get gold_workflow
        meta_path = f"{TEST_ROOT}/{dataset_name}/metadata_{meta_id}.json"
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            
            # Get workflow_tags as gold_workflow
            gold_workflow = meta.get("workflow_tags", "")
            items = [item.strip() for item in gold_workflow.split(",") if item.strip()]
            gold_workflow = "\n".join(f"{i+1}. {item}." for i, item in enumerate(items))
            if not gold_workflow:
                print(f"Warning: No workflow_tags found, key: {key}")
                gold_workflow = "default gold workflow"

            # added
            gold_workflow = ""
            # added end
            
            # Get query text
            query_text = ""
            if "queries" in meta and isinstance(meta["queries"], list):
                for qlist in meta["queries"]:
                    for q in qlist:
                        if str(q.get("qid", 0)) == query_id:
                            query_text = q.get("question", "")
                            break
                
        except Exception as e:
            print(f"Failed to read metadata file: {meta_path}, error: {str(e)}")
            gold_workflow = "default gold workflow"
            gold_workflow = ""
            query_text = "default query"
        
        # Get cost information
        cost = all_hypo_results.get(key, (None, 0.0))[1]
        
        # Add to evaluation task list
        eval_tasks.append((key, meta_path, gold_hypo, gen_hypo, gold_workflow, gen_workflow, eval_out, query_text, cost))
    
    print(f"Collected {len(eval_tasks)} evaluation tasks")
    
    # Run evaluation in parallel
    evaluator = Evaluator(max_workers=10)
    eval_results = evaluator.run_evaluation(eval_tasks)
    
    # Count the number of evaluation results
    eval_count = len(glob.glob(f"{EVAL_RESULT_DIR}/eval_*.json"))
    print(f"Generated {eval_count} in total, original task number is {len(all_tasks)}")

    # Display default value usage statistics
    usage_stats = usage_counter.get_stats()
    print("\Default value usage statistics:")
    print(f"Default code usage: {usage_stats['default_code_count']}")
    print(f"Default hypothesis usage: {usage_stats['default_hypo_count']}")
    print(f"Default workflow usage: {usage_stats['default_workflow_count']}")
    print(f"Default code usage: {usage_stats['default_code_count'] / len(all_tasks):.2%}")
    print(f"Default hypothesis usage: {usage_stats['default_hypo_count'] / len(all_tasks):.2%}")
    print(f"Default workflow usage: {usage_stats['default_workflow_count'] / len(all_tasks):.2%}")
    

    if eval_count < len(all_tasks):
        print("Warning: Evaluation result number is less than task number!")
        # Find missing tasks
        existing_eval_files = glob.glob(f"{EVAL_RESULT_DIR}/eval_*.json")
        existing_eval_keys = set()
        for eval_file in existing_eval_files:
            file_name = os.path.basename(eval_file)
            parts = file_name.replace('eval_', '').replace('.json', '').split('_')
            if len(parts) >= 3:
                dataset_name = parts[0]
                meta_id = parts[1]
                query_id = parts[2]
                key = (dataset_name, meta_id, query_id)
                existing_eval_keys.add(key)
        
        missing_eval_keys = set(all_tasks) - existing_eval_keys
        print(f"missing eval keys: {missing_eval_keys}")
        
        # Create default evaluation results for missing tasks
        for key in missing_eval_keys:
            dataset_name, meta_id, query_id = key
            
            # Get hypothesis
            hypo_log_file = f"{INFER_LOG_DIR}/{dataset_name}_{meta_id}_{query_id}_hypo.log"
            gen_hypo, gen_workflow = extract_hypo_and_workflow(hypo_log_file)
            
            # Get gold_hypo
            gold_hypo = answer_dict.get(key, "default gold hypothesis")
            
            # Read metadata to get gold_workflow
            meta_path = f"{TEST_ROOT}/{dataset_name}/metadata_{meta_id}.json"
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                gold_workflow = meta.get("workflow_tags", "default gold workflow")
            except Exception:
                gold_workflow = "default gold workflow"
            gold_workflow = ""
            
            # Create default evaluation result
            eval_out = f"{EVAL_RESULT_DIR}/eval_{dataset_name}_{meta_id}_{query_id}.json"
            default_result = {
                "final_score": 0.0,
                "hypo_score": 0.0,
                "workflow_score": 0.0,
                "gold_hypo": gold_hypo,
                "pred_hypo": gen_hypo,
                "gold_workflow": gold_workflow,
                "pred_workflow": gen_workflow,
                "cost": 0.0,
                "error": "Create default evaluation result for missing tasks"
            }
            
            # Save default evaluation result
            with open(eval_out, "w") as ef:
                json.dump(default_result, ef, indent=2)
        
        # Count the number of evaluation results again
        eval_count = len(glob.glob(f"{EVAL_RESULT_DIR}/eval_*.json"))
        print(f"After adding missing tasks, a total of {eval_count} evaluation results were generated")
            
    elif eval_count > len(all_tasks):
        print("Warning: Evaluation result number is greater than task number!")
    else:
        print("All tasks have been successfully evaluated!")

# Call main function when script is executed
if __name__ == "__main__":
    main()