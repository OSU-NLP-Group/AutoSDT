python ../src/autosdt_select_verify_files.py \
--llm_engine_name struct_azure_gpt-4o \
--input_jsonl ../result/python_files.jsonl \
--repo_base_dir ../../downloaded_repos \
--output_jsonl ../result/scientific_python_files.jsonl \
--max_workers 4 \
--api_version 2024-12-01-preview
