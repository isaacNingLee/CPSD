python main.py  --pretrained_model_name_or_path=/home/ilee0022/cl-gen/models/miniSD-diffusers \
                --dataset_path=/home/ilee0022/cl-gen/datasets/cifar100 \
                --prepared_dataset_path=/home/ilee0022/cl-gen/CPSD/prepared_ci_dat/CIFAR100_prep \
                --dataset_name=cifar100 \
                --cpsd_resolution=256 \
                --cpsd_batch_size=4 \
                --cpsd_gradient_accumulation_steps=1 \
                --mixed_precision=fp16 \
                --cpsd_num_train_epochs=1 \
                --cpsd_learning_rate=1e-04 \
                --cpsd_ema \
                --cpsd_max_grad_norm=1 \
                --cpsd_lr_scheduler=constant \
                --cpsd_lr_warmup_steps=0 \
                --output_dir=output \
                --num_class=100 \
                --cpsd_snr_gamma=5 \
                --base_task_class_num=10 \
                --total_task=10 \
                --n_replay=600 \
                --cpsd_scale_lr \
                --num_inference_steps=20 \
                --c_epochs=1 \
                --c_lr=0.1 \
                --c_resolution=32 \
                --c_batch_size=32 \
                --c_wd=0.0 \
                --method=sd \
                --run_name=sd_cifar100_gen \
                --project_name=cifar100 \
                --trainer=anneal \
                --shared_gen_replay \
                --max_gen_batch_size=8 \
                --c_anneal_epochs=1 \
                --anti_discrim \
                --init_option=mean \
                --cpsd_scheduler=DPMSolver \
                --cpsd_dist_match=0.003 \

#                 --cpsd_dist_match=0.003 \
# --prepared_gen_dataset_path=/home/ilee0022/cl-gen/CPSD/output/20240721-232127_cp_cpsd_224_anneal_mix_init/gen_samples \