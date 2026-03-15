#CUDA_VISIBLE_DEVICES=2 \
## SWIFT_DEBUG=1 \
#swift infer \
#  --model_type deepseek_janus_pro \
#  --model /data1/sshan24/Janus/model \
#	--adapters /data1/sshan24/Janus/medical_processor/output/v32-20260115-205207/checkpoint-532/ \
#	--stream true \
#	--max_length 8192 \
#  --infer_backend pt \
#  --temperature 0 \
#  --template llama3\
#  --model_kwargs '{"use_vision": false}'
#  --merge_lora true \

# Swift ��������
CUDA_VISIBLE_DEVICES=2 \
swift infer \
  --model /data1/sshan24/Janus/medical_processor/output_baseline_risk_v2/v1-20260313-181901/checkpoint-21763-merged \
  --val_dataset /data1/sshan24/Janus/medical_processor/data_risk_v2/eval.jsonl \
  --max_new_tokens 8192 \
  --temperature 0.0 \
  --logprobs true \
  # --result_path results_baseline_risk.jsonl