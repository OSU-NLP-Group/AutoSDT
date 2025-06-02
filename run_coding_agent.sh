python CodingAgent/src/CodingAgent.py \
--llm_engine_name us.anthropic.claude-3-7-sonnet-20250219-v1:0 \
--input_jsonl CodingAgent/result/ready_python_files.jsonl \
--output_jsonl CodingAgent/result/coding_agent_result.jsonl \
--program_output_dir generated_gold_programs \
--benchmark_dir benchmark/datasets/ \
--api_version 2024-12-01-preview \
--num_threads 8 \
--use_self_debug