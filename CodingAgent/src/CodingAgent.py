import argparse
import threading
from queue import Queue
from pathlib import Path
import json
import os
import subprocess
from tqdm import tqdm
from string import Template
from engine.base_engine import LLMEngine
from shutil import copyfile, rmtree
from pydantic import BaseModel
import time
import re
import pdb


SELF_DEBUG_PROMPT = """The user may execute your code and report any exceptions and error messages.
You should address the reported issues and respond with a fixed, complete program.
Note that, when addressing bugs, you should ONLY focus on addressing the errors and exceptions and MUST NOT change or delete the main functionality and logic of the original program just to make it executable.
You should always include the full code in your response. You cannot generate things like "# Implementation remains unchanged" or "pass" or "TODO" or "mock" or "dummy" or "empty" or "not implemented" or "not available" or "not provided" or "not specified" or "not defined" or "not found" in your response.
"""

FORMAT_PROMPT = """You should keep your response concise and do not use a code block if it's not intended to be executed.
Please do not suggest a few line changes, incomplete program outline, or partial code that requires the user to modify. Your response should include a complete, standalone, executable program. 
Please do not use any interactive Python commands in your program, such as `!pip install numpy`, which will cause execution errors.
Regardless of the iterations of self-debugging, make sure to follow the structured output format."""

FORMAT_PROMPT_REGULAR = """Please keep your response concise and do not use a code block if it's not intended to be executed.
Please do not suggest a few line changes, incomplete program outline, or partial code that requires the user to modify. Your response should include a complete, standalone, executable program. 
Please do not use any interactive Python commands in your program, such as `!pip install numpy`, which will cause execution errors.
Regardless of the iterations of self-debugging, make sure to wrap your program in a code block that specifies the script type, python. For example:
```python
print("Hello World!")
```"""

template = Template("""You will be given a code file from a github repo. Your task is to modify the code into a self-contained program that can be run locally and separately.
Please do not change the original functionality of the code. YOU MUST keep the original logic and functionality of the code as much as possible. YOU SHOULD NEVER include dummy/pass statements or empty/mock functions in your response.
You need to slightly modify the source code's input/output logistics and intermediate steps to make it a stand-alone program that can be executed locally.
The modified code will then be executed in a local environment. If there are errors, you need to debug the code based on the execution feedback.
All the datasets and dependency files are located at $dataset_path. If the original code has imported modules from local files, you can assume they exist and do the same imports in your modified code. Here is the directory structure of the dataset and dependency files:
```
$dataset_structure
```
Make sure that the code you generate uses the same input files as the original code. Do not generate dummy input files or input data.
                                          
For the output of the programs, your code should save the results to a file named "pred_results/pred_[#code_file_name].[#extension]", depending on the type of data such as csv, txt or jsonl. ALL outputs of the program should be saved in the directory pred_results/. You should never create new folders or files outside of the specified directory.

## Code to be modified: $code_file_name
$code
## Code to be modified end

$self_debug_prompt
$format_prompt
""")

def print_dir(startpath):
    result = ""
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = "|--" + '-' * 2 * level + " "
        result += '{}{}/'.format(indent, os.path.basename(root)) + "\n"
        subindent = "|--" + '-' * 2 * (level + 1) + " "
        for f in files:
            result += '{}{}'.format(subindent, f) + "\n"
    return result

class CodingAgent():
    def __init__(self, llm_engine_name, context_cutoff, api_version="2024-10-21", use_self_debug=False, use_structured_output=False, max_debug_attempts=3):
        api_key = os.environ.get("AZURE_API_KEY")
        azure_endpoint = os.environ.get("AZURE_ENDPOINT")
        api_version = api_version
        self.llm_engine = LLMEngine(llm_engine_name, api_key=api_key, azure_endpoint=azure_endpoint, api_version=api_version)
        self.context_cutoff = context_cutoff
        self.use_self_debug = use_self_debug
        self.use_structured_output = use_structured_output
        self.max_debug_attempts = max_debug_attempts
        self.history = []
        self.sys_msg = ""

    def write_program(self, assistant_output, out_fname):
        with open(out_fname, "w", encoding="utf-8") as f:
            f.write(assistant_output)
        return assistant_output

    def install_dependencies(self, out_fname, conda_env):
        thread_id = int(conda_env.split('-')[-1])
        test_path = Path(f"program_to_eval_{thread_id}")
        if test_path.exists():
            rmtree(test_path)
        os.mkdir(test_path)
        copyfile(out_fname, test_path / out_fname.split("/")[-1])

        subprocess.run([
            "pipreqs", str(test_path), f"--savepath=instance_requirements_{thread_id}.txt", "--mode", "no-pin"
        ], capture_output=True)
        exec_res = subprocess.run(["conda", "run", "-n", conda_env, "pip", "install", "-r", f"instance_requirements_{thread_id}.txt"], capture_output=True, env=os.environ.copy())
        if exec_res.returncode != 0:
            return False, exec_res.stderr.decode("utf-8")
        return True, ""

    def execute_program(self, out_fname, conda_env):
        module_name = out_fname.replace("/", ".")[:-3]
        try:
            exec_res = subprocess.run(["conda", "run", "-n", conda_env, "python", "-m", module_name], capture_output=True, timeout=900)
            if exec_res.returncode == 0:
                return True, ""
            else:
                return False, exec_res.stderr.decode("utf-8")
        except subprocess.TimeoutExpired:
            return False, "Execution timeout."
        except FileExistsError:
            time.sleep(0.5)
        except Exception as e:
            return False, f"Unexpected error during install_dependencies: {str(e)}"
        return False, "Max retries exceeded due to persistent FileExistsError."

    def solve_task(self, task, out_fname, initial_prompt, conda_env):
        class CodeResponse(BaseModel):
            explanation: str
            code: str

        temperature = 0.7
        top_p = 0.95
        max_tokens = 20000

        self.sys_msg = initial_prompt
        self.history = [{'role': 'user', 'content': self.sys_msg}]
        code_response = ""
        if self.use_structured_output:
            try:
                assistant_output, _, _ = self.llm_engine.respond_structured(self.history, CodeResponse, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
                code_response = assistant_output.code
            except Exception as e:
                code_response = "STRUCTURED_OUTPUT_ERROR"
        else:
            assistant_output, _, _ = self.llm_engine.respond(self.history,  temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            match = re.search(r"```python(.*?)```", assistant_output, re.DOTALL)
            if match:
                code_response = match.group(1).strip()
            else:
                code_response = "REGULAR_EXPRESSION_ERROR"
        self.write_program(code_response, out_fname)
        self.history.append({'role': 'assistant', 'content': code_response})

        install_success, err_msg = self.install_dependencies(out_fname, conda_env)
        return_success_label = False
        if install_success:
            exec_success, err_msg = self.execute_program(out_fname, conda_env)
            if exec_success:
                return_success_label = True

        debug_attempt = 0
        while (not return_success_label) and self.use_self_debug and debug_attempt < self.max_debug_attempts:
            debug_attempt += 1
            try:
                debug_prompt = "## Self-debug Iteration " + str(debug_attempt) + ":\n"
                debug_code_response = assistant_output.code
            except Exception as e:
                debug_code_response = "STRUCTURED_OUTPUT_ERROR"
            debug_prompt += "## Code:\n" + str(debug_code_response)
            debug_prompt += "\nThe previous execution resulted in the following error message:\n" + err_msg + "\n"
            debug_prompt += SELF_DEBUG_PROMPT
            
            self.history.append({'role': 'user', 'content': debug_prompt})
            code_response_sd = ""
            if self.use_structured_output:
                try:
                    assistant_output, _, _ = self.llm_engine.respond_structured(self.history, CodeResponse, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
                    code_response = assistant_output.code
                except Exception as e:
                    code_response = "STRUCTURED_OUTPUT_ERROR"
            else:
                assistant_output, _, _ = self.llm_engine.respond(self.history,  temperature=temperature, top_p=top_p, max_tokens=max_tokens)
                match = re.search(r"```python(.*?)```", assistant_output, re.DOTALL)
                if match:
                    code_response_sd = match.group(1).strip()
                else:
                    code_response_sd = "REGULAR_EXPRESSION_ERROR"
            self.write_program(code_response_sd, out_fname)
            self.history.append({'role': 'assistant', 'content': code_response_sd})
            
            install_success_sd, err_msg = self.install_dependencies(out_fname, conda_env)
            
            if install_success_sd:
                exec_success_sd, err_msg = self.execute_program(out_fname, conda_env)
                if exec_success_sd:
                    return_success_label = True
                    break

        return {"history": self.history, "success": return_success_label, "error_message": err_msg}

lock = threading.Lock()
progress_lock = threading.Lock()
progress_bar = None

def worker(thread_id, task_queue, args, existing_file_urls):
    global progress_bar
    agent = CodingAgent(args.llm_engine_name, args.context_cutoff, args.api_version, args.use_self_debug, args.use_structured_output, args.max_debug_attempts)
    conda_env = f"sci-agent-eval-{thread_id}"

    while True:
        try:
            with lock:
                if task_queue.empty():
                    break
                x = task_queue.get_nowait()
        except:
            break

        file_url = x["file_url"]
        discipline = x["discipline"]

        with lock:
            if file_url in existing_file_urls:
                print(f"Skipping existing file URL: {file_url}")
                progress_bar.update(1)
                task_queue.task_done()
                continue

        file_name = file_url.split("/")[-1]
        github_part = file_url.split("github.com/")[-1]
        github_owner = github_part.split("/")[0]
        github_repo_name = github_part.split("/")[1]
        local_dir = f"{args.benchmark_dir}/{github_repo_name}"
        file_tree = print_dir(local_dir)

        initial_prompt = template.substitute(
            dataset_path=f"{local_dir}",
            dataset_structure=file_tree,
            code=x["code"],
            code_file_name=file_name,
            self_debug_prompt=SELF_DEBUG_PROMPT,
            format_prompt=FORMAT_PROMPT if args.use_structured_output else FORMAT_PROMPT_REGULAR,
        )

        if not os.path.exists(args.program_output_dir):
            os.makedirs(args.program_output_dir)
        out_fname = f"{args.program_output_dir}/{github_repo_name}_{file_name}"
        result = agent.solve_task({}, out_fname, initial_prompt, conda_env)

        result_dict = {
            "discipline": discipline,
            "file_url": file_url,
            "out_fname": out_fname,
            "success": 1 if result["success"] else 0,
            "error_message": result["error_message"],
            "assistant_reply": result["history"][-1]["content"],
        }

        with lock:
            existing_file_urls.add(file_url)
            with open(args.output_jsonl, "a", encoding="utf-8") as fw:
                fw.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
        
        with progress_lock:
            progress_bar.update(1)
            task_queue.task_done()

def main():
    global progress_bar
    parser = argparse.ArgumentParser(description="Run CodingAgent to process dataset-based scripts.")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--benchmark_dir", type=str, required=True, help="Directory for benchmark storage.")
    parser.add_argument("--context_cutoff", type=int, default=28000, help="Context cutoff length for LLM.")
    parser.add_argument("--use_self_debug", action="store_true", help="Enable self-debugging feature.")
    parser.add_argument("--use_structured_output", action="store_true", help="Enable structured output feature.")
    parser.add_argument("--llm_engine_name", type=str, required=True, help="LLM model name.")
    parser.add_argument("--api_version", type=str, default="2024-10-21", help="API version for Azure OpenAI.")
    parser.add_argument("--program_output_dir", type=str, default="../../generated_gold_programs", help="Directory to save generated programs.")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of worker threads.")
    parser.add_argument("--max_debug_attempts", type=int, default=3, help="Maximum number of self-debug attempts.")
    args = parser.parse_args()

    existing_file_urls = set()

    if os.path.exists(args.output_jsonl):
        with open(args.output_jsonl, "r", encoding="utf-8") as fr:
            for line in fr:
                try:
                    item = json.loads(line)
                    if "file_url" in item:
                        existing_file_urls.add(item["file_url"])
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line in {args.output_jsonl}, skipping.")
 
    print(f"Found {len(existing_file_urls)} already processed file URLs.")
    
    data = []
    try:
        with open(args.input_jsonl, "r", encoding="utf-8") as fr:
            for line in fr:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line in {args.input_jsonl}, skipping.")
    except Exception as e:
        print(f"Error loading input file: {e}")
        return
    
    task_queue = Queue()
    tasks_to_process = 0
    
    for item in data:
        file_url = item.get("file_url")
        if file_url and file_url not in existing_file_urls:
            task_queue.put(item)
            tasks_to_process += 1
        elif file_url in existing_file_urls:
            print(f"Skipping already processed URL: {file_url}")
    
    if tasks_to_process == 0:
        print("No new tasks to process. All URLs have been processed.")
        return
    print(f"Found {len(existing_file_urls)} already processed file URLs.")
    print(f"Added {tasks_to_process} tasks to the queue.")
    progress_bar = tqdm(total=tasks_to_process, desc="Progress")

    threads = []
    for i in range(min(args.num_threads, tasks_to_process)):
        t = threading.Thread(target=worker, args=(i, task_queue, args, existing_file_urls))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    progress_bar.close()
    print("All tasks completed.")

if __name__ == "__main__":
    main()
