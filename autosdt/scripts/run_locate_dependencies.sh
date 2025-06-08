python ../src/autosdt_select_locate_dependencies.py \
--llm_engine_name struct_azure_gpt-4o \
--input_jsonl ../result/scientific_python_files.jsonl \
--repo_base_dir ../../downloaded_repos \
--output_jsonl ../result/scientific_python_files_w_dataset.jsonl \
--api_version 2024-12-01-preview \
--max_workers 1
