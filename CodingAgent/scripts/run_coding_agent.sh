python ../src/CodingAgent.py \
--llm_engine_name azure_gpt-4o \
--input_jsonl ../result/ready_python_files.jsonl \
--output_jsonl ../result/coding_agent_result.jsonl \
--program_output_dir ../../generated_gold_programs \
--benchmark_dir ../../benchmark_auto \
--api_version 2024-10-21
