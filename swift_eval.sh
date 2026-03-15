CUDA_VISIBLE_DEVICES=2 \
swift eval \
  --model_type deepseek_janus_pro \
  --model /data1/sshan24/Janus/model \
  --ckpt_dir /data1/sshan24/Janus/medical_processor/output_baseline_risk_v2/v1-20260313-181901/checkpoint-21763 \
  --merge_lora true \
  --eval_backend Native \
  --infer_backend pt \
  --eval_dataset general_qa \
  --eval_dataset_args '{"general_qa": {"local_path": "/data1/sshan24/Janus/medical_processor/data", "subset_list": ["eval"]}}' \
  --max_length 8192 \
#  --ckpt_dir /data1/sshan24/Janus/medical_processor/output/v32-20260115-205207/checkpoint-532 \  # 780��
 
