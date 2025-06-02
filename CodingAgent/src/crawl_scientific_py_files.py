import os
import json
import argparse
from pathlib import Path
from shutil import rmtree
import subprocess
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import nbformat
import requests


def get_default_branch(github_owner: str, github_repo_name: str) -> str:
    """
    Get the default branch of a GitHub repository.
    """
    try:
        # try to get the default branch using GitHub API
        api_url = f"https://api.github.com/repos/{github_owner}/{github_repo_name}"
        response = requests.get(api_url)
        if response.status_code == 200:
            repo_info = response.json()
            return repo_info.get("default_branch", "main")
    except Exception as e:
        print(f"Error getting default branch for {github_owner}/{github_repo_name}: {e}")
    
    # if the API call fails, try to get the default branch using git command
    try:
        repo_dir = Path(f"./temp_repo_{github_repo_name}")
        if not repo_dir.exists():
            repo_dir.mkdir(parents=True)
            subprocess.run(["git", "init"], cwd=str(repo_dir), check=True)
            subprocess.run(["git", "remote", "add", "origin", f"https://github.com/{github_owner}/{github_repo_name}.git"], cwd=str(repo_dir), check=True)
            subprocess.run(["git", "fetch", "--depth=1"], cwd=str(repo_dir), check=True)
        
        result = subprocess.run(["git", "remote", "show", "origin"], cwd=str(repo_dir), capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            if "HEAD branch" in line:
                default_branch = line.split(":")[-1].strip()
                # clean up temporary directory
                rmtree(repo_dir)
                return default_branch
    except Exception as e:
        print(f"Error getting default branch using git for {github_owner}/{github_repo_name}: {e}")
        # clean up temporary directory
        if repo_dir.exists():
            rmtree(repo_dir)


def download_repo(github_owner: str, github_repo_name: str, base_dir: Path) -> tuple[Path, str]:
    """
    Download or update the repository locally and return the default branch.
    """
    repo_dir = base_dir / f"{github_repo_name}"
    repo_url = f"https://github.com/{github_owner}/{github_repo_name}.git"
    
    # get the default branch
    default_branch = get_default_branch(github_owner, github_repo_name)
    
    if repo_dir.exists():
        print(f"Updating existing repository: {repo_dir}")
        subprocess.run(["git", "-C", str(repo_dir), "pull"], check=True)
    else:
        print(f"Cloning new repository: {repo_url} to {repo_dir}")
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)

    return repo_dir, default_branch


def list_local_python_files(repo_dir: Path) -> list[Path]:
    """
    Recursively find all .py and .ipynb files in the local repository.
    """
    python_files = []
    for root, dirs, files in os.walk(repo_dir):
        for file in files:
            if file.endswith(".py") or file.endswith(".ipynb"):
                python_files.append(Path(root) / file)
    return python_files


def extract_code_from_ipynb(file_content):
    """
    Extract code cells from an IPython Notebook (.ipynb).
    :param file_content: Content of the .ipynb file
    :return: Extracted code as a single string
    """
    notebook = nbformat.reads(file_content, as_version=4)
    code_cells = []
    
    for cell in notebook.cells:
        if cell.cell_type == "code":
            code_cells.append(cell.source)
    
    return "\n".join(code_cells)


def read_file_content(file_path: Path) -> str:
    """
    Load the content of a file.
    """
    try:
        with file_path.open("r", encoding="utf-8") as f:
            code_content = f.read()
            return code_content if ".ipynb" not in str(file_path) else extract_code_from_ipynb(code_content)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def main():
    parser = argparse.ArgumentParser(description="Download and process Python files from GitHub repositories.")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the CSV file containing GitHub repository URLs.", default="../../repo_list/expanded_repositories_with_papers_and_data (5).csv")
    parser.add_argument("--repo_base_dir", type=str, required=True, help="Base directory for storing downloaded repositories.", default="../../downloaded_repos")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.", default="../result/python_files.jsonl")
    args = parser.parse_args()

    repo_base_dir = Path(args.repo_base_dir)
    if not os.path.exists(repo_base_dir):
        os.makedirs(repo_base_dir)
    df = pd.read_csv(args.csv_file)
    url_to_discipline = dict(zip(df["url"], df["discipline"]))
    github_repo_urls = df["url"]

    python_files_all = defaultdict(dict)
    repo_default_branches = {}

    print("Total repositories:", len(github_repo_urls))
    for idx, url in enumerate(tqdm(github_repo_urls)):
        github_owner = url.strip("/").split("/")[-2]
        github_repo_name = url.strip("/").split("/")[-1]

        discipline = url_to_discipline.get(url, "")

        try:
            repo_dir, default_branch = download_repo(github_owner, github_repo_name, repo_base_dir)
            repo_default_branches[url] = default_branch
        except Exception as e:
            print(f"Error downloading repository from {url}, skipping. Error: {e}")
            continue

        python_files = list_local_python_files(repo_dir)
        python_files_all[url] = {}
        print(f"Current repository: {idx + 1}, {url}, default branch: {default_branch}")


        for python_file in tqdm(python_files):
            file_url_relative = str(python_file.relative_to(repo_dir))
            file_url = f"{url}/blob/{default_branch}/{file_url_relative}"
            file_path = str(python_file.relative_to(repo_base_dir))
            print("Processing file:", file_url)

            code_content = read_file_content(python_file)
            if not code_content:
                continue

            python_files_all[url][file_url] = {
                "code": code_content,
                "file_path": file_path,
                "discipline": discipline
            }

    num_files = [len(python_files_all[url]) for url in python_files_all]
    if len(num_files) > 0:
        print("Average number of files:", sum(num_files) / len(num_files))
        print("Mean number of files:", pd.Series(num_files).mean())
        print("Standard deviation:", pd.Series(num_files).std())
    else:
        print("No Python files found.")

    num_lines = []
    total_files_count = sum(num_files)
    for repo_url in python_files_all:
        num_lines.append([len(python_files_all[repo_url][file_url]["code"].split("\n"))
                          for file_url in python_files_all[repo_url]])
    if total_files_count > 0:
        print("Average number of lines:", sum([sum(n) for n in num_lines]) / total_files_count)
        print("Mean number of lines:", pd.Series([sum(n) for n in num_lines]).mean())
        print("Standard deviation:", pd.Series([sum(n) for n in num_lines]).std())

    with open(args.output_file, "w") as f:
        for repo_url in python_files_all:
            for file_url in python_files_all[repo_url]:
                file_data = python_files_all[repo_url][file_url]
                code_content = file_data["code"]
                file_path = file_data["file_path"]
                discipline = file_data["discipline"]

                # updated 04/07/2025
                # keep condition: <1000, ipynb, py with __main__
                if code_content.count("\n") > 1000:
                # if code_content.count("\n") > 1000 or ("__main__" not in code_content and ".ipynb" not in file_url):
                    continue

                row_dict = {
                    "discipline": discipline,
                    "url": repo_url,
                    "file_url": file_url,
                    "file_path": file_path,
                    "code": code_content,
                    "is_scientific_coding_task": ""
                }
                f.write(json.dumps(row_dict) + "\n")

if __name__ == "__main__":
    main()
