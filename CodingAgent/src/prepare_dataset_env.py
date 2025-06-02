import os
import json
import argparse
from pathlib import Path
from shutil import copyfile, rmtree, copytree
from tqdm import tqdm
from collections import defaultdict

MAX_PATH_LENGTH = 255  # Maximum path length limit for most operating systems

def truncate_path(path: str, max_length: int = MAX_PATH_LENGTH) -> str:
    """
    Truncate overly long file paths, keeping only the critical directories and file name.
    """
    if len(path) > max_length:
        truncated = "..." + path[-(max_length - 3):]
        print(f"[WARN] Path too long, truncated: {truncated}")
        return truncated
    return path

def copy_datasets_and_modules(
    repo_name: str, 
    base_dir: Path, 
    repo_base_dir: Path,
    dataset_paths=None, 
    import_local_modules=None
):
    """
    Copy dataset files to benchmark/datasets/{repo}/data/ and
    copy local module files to benchmark/dataset/{repo}/modules/.
    If dataset_paths or import_local_modules are None or empty,
    only the directories will be created without copying any files.
    """
    if dataset_paths is None:
        dataset_paths = []
    if import_local_modules is None:
        import_local_modules = []

    base_dir = Path(base_dir)
    #data_dir = base_dir / "data"
    #modules_dir = base_dir / "modules"
    data_dir = base_dir
    modules_dir = base_dir
    results_dir = Path("pred_results")
    data_dir.mkdir(parents=True, exist_ok=True)
    modules_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    data_flag, modules_flag = False, False

    # Copy dataset files
    for dpath in dataset_paths:
        source_path = repo_base_dir / dpath
        target_path = data_dir / Path(dpath).relative_to(repo_name)
        target_path_str = truncate_path(str(target_path))
        target_path = Path(target_path_str)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if source_path.exists():
                if target_path.exists(): 
                    print(f"[INFO] Dataset already exists: {target_path}, skipping.")
                else: 
                    if source_path.is_dir():
                        copytree(source_path, target_path)
                        print(f"[OK] Copied dataset directory -> {target_path}")
                    else: 
                        copyfile(source_path, target_path)
                        print(f"[OK] Copied dataset file -> {target_path}")
                data_flag = True
            else:
                print(f"[WARN] Dataset file or directory not found: {source_path}")
        except Exception as e:
            print(f"[ERR] Failed to copy dataset file or directory {dpath}: {e}")

    # Copy module files
    for mpath in import_local_modules:
        source_path = repo_base_dir / mpath
        target_path = modules_dir / Path(mpath).relative_to(repo_name)
        target_path_str = truncate_path(str(target_path))
        target_path = Path(target_path_str)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if source_path.exists():
                if target_path.exists(): 
                    print(f"[INFO] Module already exists: {target_path}, skipping.")
                else: 
                    if source_path.is_dir():
                        copytree(source_path, target_path)
                        print(f"[OK] Copied module directory -> {target_path}")
                    else: 
                        copyfile(source_path, target_path)
                        print(f"[OK] Copied module file -> {target_path}")
                modules_flag = True
            else:
                print(f"[WARN] Module file or directory not found: {source_path}")
        except Exception as e:
            print(f"[ERR] Failed to copy module file or directory {mpath}: {e}")

    return (len(dataset_paths) == 0 or data_flag) and (len(import_local_modules) == 0 or modules_flag)

def main():
    parser = argparse.ArgumentParser(description="Copy dataset and module files to benchmark directories.")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to the input JSONL file.", default="../result/scientific_python_files_w_dataset.jsonl")
    parser.add_argument("--repo_base_dir", type=str, required=True, help="Base directory for repositories.", default="../../downloaded_repos")
    parser.add_argument("--benchmark_base_dir", type=str, required=True, help="Base directory for benchmark storage.", default="../../benchmark/datasets")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Path to the output JSONL file.", default="../result/ready_python_files.jsonl")
    args = parser.parse_args()

    repo_base_dir = Path(args.repo_base_dir)
    benchmark_base_dir = Path(args.benchmark_base_dir)
    input_jsonl = args.input_jsonl
    output_jsonl = args.output_jsonl

    if not os.path.exists(input_jsonl):
        print(f"[ERR] JSONL file not found: {input_jsonl}")
        return

    data = [json.loads(x) for x in open(input_jsonl).readlines()]
    download_success_data = []

    for x in tqdm(data):
        if "is_scientific_coding_task" not in x or "dataset_recognition_label" not in x:
            continue
        if x.get("is_scientific_coding_task").lower() == "yes" and \
            (x["dataset_recognition_label"] is None or x["dataset_recognition_label"].lower() != "no"):

            file_path = x.get("file_path")
            if not file_path:
                print("[WARN] file_path not found in JSON data.")
                continue

            repo_name = file_path.split("/")[0]
            print(f"Repo name: {repo_name}")
            dataset_paths = x.get("dataset_paths") or []
            import_local_modules = x.get("import_local_modules") or []
            dataset_paths = [f"{repo_name}/{path}".replace("*", "").strip() for path in dataset_paths]
            import_local_modules = [f"{repo_name}/{path}".replace("*", "").strip() for path in import_local_modules]
            print(f"Dataset paths: {dataset_paths}")
            print(f"Import local modules: {import_local_modules}")
            output_dir = benchmark_base_dir / repo_name
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"Output directory: {output_dir}")
            copy_success = copy_datasets_and_modules(
                repo_name=repo_name, 
                base_dir=output_dir,
                repo_base_dir=repo_base_dir,
                dataset_paths=dataset_paths, 
                import_local_modules=import_local_modules
            )

            if copy_success:
                download_success_data.append(x)

    with open(output_jsonl, "w") as f:
        for x in download_success_data:
            f.write(json.dumps(x) + "\n")

if __name__ == "__main__":
    main()
