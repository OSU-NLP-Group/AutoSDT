import json
import os
import argparse
from pathlib import Path
from string import Template
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from pydantic import BaseModel

import litellm
from litellm import model_cost, token_counter
from litellm.utils import trim_messages

from engine.base_engine import LLMEngine

def should_exclude(file_path, github_repo_name):
    """
    Check whether a file should be excluded based on the following conditions:
    1. The file is inside directories such as src, utils, test, docs, etc.
    2. The file name contains any of the excluded keywords (e.g., setup.py, __init__.py, etc.).
    3. The file path contains the repository name (likely a package file).
    """
    exclude_dirs = {"util", "utils", "test", "tests", "doc", "docs", "setting", "settings", "scripts", "build"}
    
    exclude_files = {"setup", "__init__", "utils", "util", "requirements", "config", "settings", "manage"}

    path_parts = file_path.lower().split("/")

    if any(excluded_dir in part for part in path_parts for excluded_dir in exclude_dirs):
        return True

    file_name = os.path.basename(file_path).lower()
    if any(excluded_file in file_name for excluded_file in exclude_files):
        return True

    #if github_repo_name.lower() in path_parts:
        #return True

    return False

def read_code_from_local(file_path: str, repo_base_dir: Path) -> str:
    """
    Read file content from a local path.
    """
    abs_file_path = repo_base_dir / file_path
    try:
        with abs_file_path.open("r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {abs_file_path}: {e}")
        return ""

def process_file(file_data: Dict[str, Any], llm_engine: LLMEngine, repo_base_dir: Path) -> Dict[str, Any]:
    
    class ResponseFormat (BaseModel):
        reasoning: str
        is_scientific_task: str
    """
    Process a single file in parallel, including:
    1. Read file content from local storage.
    2. Filter out unnecessary files.
    3. Call the LLM API.
    """
    file_path = file_data.get("file_path")
    file_url = file_data.get("file_url")
    repo_url = file_data.get("url")
    discipline = file_data.get("discipline")
    
    if not file_path or not repo_url:
        print(f"Warning: Missing file_path or repo_url in {file_data}, skipping.")
        return {}

    github_owner = repo_url.strip("/").split("/")[-2]
    github_repo_name = repo_url.strip("/").split("/")[-1]

    if should_exclude(file_path, github_repo_name):
        print(f"Skipping excluded file: {file_path}")
        return {}

    code = read_code_from_local(file_path, repo_base_dir)
    if not code:
        print(f"Warning: Failed to read code from local path: {file_path}, skipping.")
        return {}

    template = Template("""### Instruction
Given a code file, you need to verify if the current code is a scientific task. Several conditions:

1. Functionality: the functionality of the given program should be related to tasks in a scientific workflow. These tasks include but are not limited to feature engineering, machine learning, deep learning, computational analysis, data visualization, model training, numerical calculation/analysis, statistical methods, domain-specific analysis/simulation, etc.
2. Input: the program should receive at least one or multiple datasets as input. In other words, the program is dealing with a dataset and conducting analysis or experiments on top of the data. The data can either be loaded through built-in functions or be loaded from local files. If the current program does not receive and process any data, it cannot be considered as "a scientific task" here.
3. Output: the program should output numerical or visualization results that can be further evaluated.

A code file is considered a scientific task ONLY IF it completely satisfied the three dimensions above. For example, code files that purely contain modeling, training/testing, data pre-processing, or only consist of utility functions or class definitions, are not considered a scientific task.

### Program
# Program name: $code_name
$code

### Output format
You should first explain the reason behind your judgment in the reasoning field: {}, and finally output your judgment in the field is_scientific_task: {Yes/No}.
""")

    initial_prompt = template.substitute(code_name=file_url, code=code)
    # import pdb; pdb.set_trace()
    if code.count("\n") > 1000: #or "__main__" not in code:
        response = None
        judgment = "No"
    
    # updated 04/16/2025
    # keep condition: <1000, ipynb, py with __main__
    # if code.count("\n") > 1000:
    # # if code.count("\n") > 1000 or ("__main__" not in code and ".ipynb" not in file_url):
    #     response = None
    #     judgment = "No"
    else:
        user_input = [{"role": "user", "content": initial_prompt}]
        response, _, _ = llm_engine.respond_structured(
            user_input, temperature=0.1, struct_format = ResponseFormat, top_p=0.9, max_tokens=5000
        )
        #if not response:
            #response = "[REASON]: No response\n[IS-A-SCIENTIFIC-TASK]: No"
        reasoning = response.reasoning
        #judgment = response.split("[IS-A-SCIENTIFIC-TASK]: ")[-1].strip(".")
        judgment = response.is_scientific_task
    # import pdb; pdb.set_trace()
    return {
        "discipline": discipline,
        "file_url": file_url,
        "file_path": file_path,
        "code": code,
        "response": reasoning,
        "is_scientific_coding_task": judgment,
    }

def main():
    parser = argparse.ArgumentParser(description="Process Python files for scientific task classification.")
    parser.add_argument("--llm_engine_name", type=str, default="azure_gpt-4o", help="Name of the LLM model to use.")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to the input JSONL file.", default="../result/python_files.jsonl")
    parser.add_argument("--repo_base_dir", type=str, required=True, help="Base directory for repositories.", default="../../downloaded_repos")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Path to the output JSONL file.", default="../result/scientific_python_files.jsonl")
    parser.add_argument("--max_workers", type=int, default=20, help="Maximum number of threads for parallel processing.")
    parser.add_argument("--api_version", type=str, default="2024-10-21", help="API version for Azure OpenAI.")
    args = parser.parse_args()

    repo_base_dir = Path(args.repo_base_dir)
    data = [json.loads(x) for x in open(args.input_jsonl).readlines()]
    
    annotated_data = []
    api_key, azure_endpoint, api_version = None, None, None
    if "azure_" in args.llm_engine_name:
        api_key = os.environ.get("AZURE_API_KEY")
        azure_endpoint = os.environ.get("AZURE_ENDPOINT")
        api_version = args.api_version
    llm_engine = LLMEngine(args.llm_engine_name, api_key=api_key, \
                                azure_endpoint=azure_endpoint, api_version=api_version)
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_file = {
            executor.submit(process_file, file_data, llm_engine, repo_base_dir): file_data for file_data in data
        }
        with open(args.output_jsonl, "a", encoding="utf-8") as f:  # Open file in append mode
            for future in tqdm(as_completed(future_to_file), total=len(data), desc="Processing code files"):
                try:
                    result = future.result()
                    if result:
                        annotated_data.append(result)
                        # Write each result immediately to the file
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"Error processing repo: {e}")
    #     for future in tqdm(as_completed(future_to_file), total=len(data), desc="Processing files"):
    #         try:
    #             result = future.result()
    #             if result:
    #                 annotated_data.append(result)
    #         except Exception as e:
    #             print(f"Error processing file: {e}")
    
    # with open(args.output_jsonl, "w", encoding="utf-8") as f:
    #     for item in annotated_data:
    #         f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    main()