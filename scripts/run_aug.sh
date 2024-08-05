python main.py  --pretrained_model_name_or_path=/home/ilee0022/cl-gen/models/miniSD-diffusers \
                --dataset_path=/home/ilee0022/cl-gen/datasets/imagenet100/huggingface \
                --prepared_dataset_path=/home/ilee0022/cl-gen/CPSD/output/20240715-035354_cp_cpsd_c_224_lucir_f_new/CI_dataset \
                --prepared_cpsd_path=/home/ilee0022/cl-gen/CPSD/output/20240730-032512_cp_cpsd2_224_anneal_mean_init_ad2/cp_embeddings \
                --cpsd_resolution=256 \
                --cpsd_batch_size=4 \
                --cpsd_gradient_accumulation_steps=1 \
                --mixed_precision=fp16 \
                --cpsd_num_train_epochs=3 \
                --cpsd_learning_rate=1e-04 \
                --cpsd_max_grad_norm=1 \
                --cpsd_lr_scheduler=constant \
                --cpsd_lr_warmup_steps=0 \
                --cpsd_ema \
                --output_dir=output \
                --num_class=100 \
                --cpsd_dist_match=0.003 \
                --cpsd_snr_gamma=5 \
                --base_task_class_num=10 \
                --total_task=10 \
                --n_replay=1300 \
                --cpsd_scale_lr \
                --num_inference_steps=50 \
                --use_blip \
                --c_epochs=40 \
                --c_lr=0.1 \
                --c_resolution=224 \
                --c_batch_size=128 \
                --c_wd=0.0001 \
                --method=cpsd \
                --run_name=cp_cpsd_224_aug \
                --trainer=aug \
                --max_gen_batch_size=4 \
                --v2_desc \
                --c_anneal_epochs=5 \
                --anti_discrim \
                --init_option=mean \

# --prepared_gen_dataset_path=/home/ilee0022/cl-gen/CPSD/output/20240721-232127_cp_cpsd_224_anneal_mix_init/gen_samples \