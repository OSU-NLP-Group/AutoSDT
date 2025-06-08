import json
import os
import argparse
from pathlib import Path
from string import Template
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from typing import List
from pydantic import BaseModel
import time
from engine.base_engine import LLMEngine
from litellm import model_cost, token_counter
from litellm.utils import trim_messages


import time

def retry_with_backoff(fn, retries=5, backoff=2):
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                print(f"[Retry] Rate limit hit. Sleeping {backoff} seconds (Attempt {attempt + 1})...")
                time.sleep(backoff)
                backoff *= 2
            else:
                raise e
    raise RuntimeError("Max retries exceeded.")


def get_repo_root(file_path: str, repo_base_dir: Path) -> Path:
    """
    Get the root directory of the repository from the file path.
    Assume the file_path format is: "repoName/path/to/file.py"
    """
    path_parts = file_path.split("/")
    if len(path_parts) < 2:
        raise ValueError(f"Invalid file_path format: {file_path}")

    repo_name = path_parts[0]

    repo_root = repo_base_dir / repo_name
    if not repo_root.exists():
        raise FileNotFoundError(f"Repo root not found: {repo_root}")
    
    # print(f"REPO ROOT: {repo_root}")
    return repo_root

def get_local_repo_tree(repo_root: Path) -> str:
    """
    Get the directory structure of the local repository and format the output.
    """
    if not repo_root.exists():
        return "Repo not found."
    
    tree_str = []
    for root, dirs, files in os.walk(repo_root):
        level = root.replace(str(repo_root), '').count(os.sep)
        indent = ' ' * 4 * level
        tree_str.append(f"{indent}|-- {os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            tree_str.append(f"{sub_indent}|-- {f}")
    final_tree = "\n".join(tree_str)
    #print(f"Final local repo tree:\n {final_tree}")
    return "\n".join(tree_str)

def process_repo_file(file_data: dict, llm_engine: LLMEngine, repo_base_dir: Path) -> dict:
    
    class DependenciesResponseFormat (BaseModel): 
        dataset_reasoning: str
        dataset_label: str
        dataset_paths: List[str]
        module_reasoning: str
        module_label: str
        module_paths: List[str]
    
    """
    Process a single file in parallel, including:
    1. Print project directory structure.
    2. Check if it includes dataset references.
    3. Invoke LLM for analysis.
    """
    file_path = file_data["file_path"]
    try:
        repo_root = get_repo_root(file_path, repo_base_dir)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error getting repo root: {e}")
        return {}

    if file_data["is_scientific_coding_task"].lower().replace("*", "").strip() == "no":
        file_data["dataset_analysis"] = None
        return file_data

    project_file_tree = get_local_repo_tree(repo_root)


    template = Template("""You are an expert software engineer who is very skilled at analyzing python code files and their repositories to extract dependencies. 
    In this task you will be given a python file and the GitHub file tree of the repository it belongs to, your job is to thoroughly understand the code and all the in-repository dependencies it needs. This is because we would like to run this code in a standalone environment and we have to make sure that all the dependencies that the code needs are copied in that environment. Hence, it is very important that you have a thorough understanding of the code and extract all in-repository dependencies needed. 
    Specifically, your job is to the following: 
    1. Recognize whether the code makes use of a dataset. The dataset can either be loaded via built-in library functions (e.g., data = MNIST ()) or loaded from a local file in the repository (csv, jsonl, xls, txt, parquet, or any other file type). If the dataset(s) used in the code are either loaded through built-in library functions or contained within the repository, you should output "Yes" in the dataset_label field.  Otherwise, you should output "No" in the dataset_label field. 
    2. In the case where the dataset used in the code is contained within the repository, you also have to find the relative path to the dataset file, based on the GitHub file tree that will be given to you. You will list the paths to all datasets used in the code as a list of paths in the field dataset_paths. 
    3. Make sure that your reasoning about the dataset recognition and the dataset file paths is correct and based on the code and the file tree, and put your reasoning in the dataset_reasoning field. 
    4. Besides the dataset, now you have to identify all other in-repository dependencies that the code uses, and extract their relative paths based on the file tree given to you. These can be modules, classes, models, or any other dependency that the code imports from a folder within the repository. If you identify that there are in-repository dependencies used, you should put a "Yes" in the module_label. Otherwise, output a "No". In the case of a "Yes", make sure to put the relative paths to all dependencies as a list of paths in the module_paths field, based on the GitHub file tree given to you. 
    5. Make sure that your reasoning about the module recognition and path extraction is correct and based on the code and file tree, and put your reasoning in the module_reasoning field. 
    6. Very important: in all path extractions, make sure to only return the path to the folder that contains the dependency, and not the full path to the file. This is because you might sometimes not be able to know which file the dependency is exactly located in based on only looking at the file tree. Thus, to stay on the safe side, just give the path to the folder that contains the dependency. 
                        
    Python code:
    $code

    Project directory:
    $directory
    
""")
    # template_deprecated = Template("""You will be given a specific python code file and the whole project directory tree of this python file. Your jobs are:
    # (1) Recognize the dataset used in this code script. The dataset can either be loaded via built-in library functions (e.g., data = MNIST()), or can be loaded from local files in csv, jsonl, tar.gz, or some domain-specific file formats. For the former case, you can output a “Yes” and the reason is “the data is loaded by built-in functions”. For the latter case, given the project structure, you need to figure out whether the used dataset file is contained somewhere in the project directory and output the relative path of that (or those) files. If the used dataset files can be found, you should output a “Yes” and give the relative path of the files based on the repo's directory (instead of based on the code). Otherwise, you should output a “No” and list the dataset file names used by the code. For all the cases when you explain the reason, you need to list the relevant code statements and explain why and which data file is used by these code statements.
    # (2) Recognize the relative module imports from local files. You need to figure out whether there are modules imported from local files instead of python libraries. If there are relative imports, you need to figure out whether the imported module files are contained somewhere in the project directory and output the relative path of that (or those) files. If there are no relative imports, you should output a "NaN" and leave the module file paths empty. If the imported module files can be found, you should output a "Yes" and output the relative path of the files based on the repo's directory. Otherwise, you should output a "No" and list the module file names that are not found in the project directory. For all the cases when you explain the reason, you need to list the relevant code statements and explain why and which module file is imported by these code statements.
    # Please strictly follow the output format, including generating the brackets:
    # [Dataset recognition]: \{Yes/No\}
    # [Reason]: \{the data is loaded by built-in functions / the data is loaded from local files\}
    # (If exist) [Dataset file path]: \{path1; path2; ...\} (Otherwise) [Dataset file path]: \{\}
    # [Module import recognition]: \{Yes/No\}
    # [Reason]: \{the module files are found in the project directory / the module files are not found in the project directory\}
    # (If exist) [Module file path]: \{path1; path2; ...\} (Otherwise) [Module file path]: \{\}

    # Python code:
    # $code

    # Project directory:
    # $directory
    # """)

    initial_prompt = template.substitute(code=file_data["code"], directory=project_file_tree)
    # print(f"Prompt: \n {initial_prompt}")
    user_input = [{"role": "user", "content": initial_prompt}]
    response, _, _ = retry_with_backoff(lambda: llm_engine.respond_structured(
        user_input,
        struct_format=DependenciesResponseFormat,
        temperature=0.1,
        top_p=0.9,
        max_tokens=2000,
    ))
    
    file_data["dataset_and_local_import_recognition_response"] = response.dataset_reasoning + "\n" + response.module_reasoning
    file_data["dataset_recognition_label"] = response.dataset_label
    file_data["dataset_paths"] = response.dataset_paths
    file_data["import_local_modules_label"] = response.module_label
    file_data["import_local_modules"] = response.module_paths
 
    # dr_match = re.search(r"\[Dataset recognition\]:\s*\{?(\w+)\}?", response)
    # file_data["dataset_recognition_label"] = dr_match.group(1).replace("*", "").strip() if dr_match else None
    # dataset_path_match = re.search(r"\[Dataset file path\]:\s*\{(.*?)\}\s*\*?", response, re.DOTALL)
    # if dataset_path_match and dataset_path_match.group(1).strip().lower() not in ["none", "{}"]:
    #     raw_paths_str = dataset_path_match.group(1).replace("*", "").strip()
    #     file_data["dataset_paths"] = [p.strip() for p in raw_paths_str.split(";")] if raw_paths_str else None
    # else:
    #     file_data["dataset_paths"] = None
    # module_import_match = re.search(r"\[Module import recognition\]:\s*\{?(\w+)\}?", response)
    # file_data["import_local_modules_label"] = module_import_match.group(1).replace("*", "").strip() if module_import_match else None
    # module_file_path_match = re.search(r"\[Module file path\]:\s*\{(.*?)\}\s*\*?", response, re.DOTALL)
    # if module_file_path_match and module_file_path_match.group(1).strip().lower() not in ["none", "{}"]:
    #     raw_module_paths_str = module_file_path_match.group(1).replace("*", "").strip()
    #     file_data["import_local_modules"] = [m.strip() for m in raw_module_paths_str.split(";")] if raw_module_paths_str else None
    # else:
    #     file_data["import_local_modules"] = None
    time.sleep(1)
    return file_data

def main():
    parser = argparse.ArgumentParser(description="Process repository files for dataset and module recognition.")
    parser.add_argument("--llm_engine_name", type=str, default="azure_gpt-4o", required=True, help="LLM model name.")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to the input JSONL file.", default="../result/scientific_python_files.jsonl")
    parser.add_argument("--repo_base_dir", type=str, required=True, help="Base directory for repositories.", default="../../downloaded_repos")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Path to the output JSONL file.", default="../result/scientific_python_files_w_dataset.jsonl")
    parser.add_argument("--max_workers", type=int, default=20, help="Number of threads for parallel processing.")
    parser.add_argument("--api_version", type=str, default="2024-10-21", help="API version for Azure OpenAI.")
    
    args = parser.parse_args()

    repo_base_dir = Path(args.repo_base_dir)
    data = [json.loads(x) for x in open(args.input_jsonl).readlines()]

    updated_data = []
    api_key, azure_endpoint, api_version = None, None, None
    #if args.llm_engine_name.contains("azure_"):
    if "azure_" in args.llm_engine_name:
        api_key = os.environ.get("AZURE_API_KEY")
        azure_endpoint = os.environ.get("AZURE_ENDPOINT")
        api_version = args.api_version
    llm_engine = LLMEngine(args.llm_engine_name, api_key=api_key, \
                                azure_endpoint=azure_endpoint, api_version=api_version)

    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_file = {
            executor.submit(process_repo_file, file_data, llm_engine, repo_base_dir): file_data for file_data in data
        }
        with open(args.output_jsonl, "a", encoding="utf-8") as f:  # Open file in append mode
            for future in tqdm(as_completed(future_to_file), total=len(data), desc="Processing code files"):
                try:
                    result = future.result()
                    if result:
                        updated_data.append(result)
                        # Write each result immediately to the file
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"Error processing repo: {e}")
    #     for future in tqdm(as_completed(future_to_file), total=len(data), desc="Processing code files"):
    #         try:
    #             result = future.result()
    #             if result:
    #                 updated_data.append(result)
    #         except Exception as e:
    #             print(f"Error processing repo: {e}")
    
    # with open(args.output_jsonl, "w", encoding="utf-8") as f:
    #     for item in updated_data:
    #         f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

