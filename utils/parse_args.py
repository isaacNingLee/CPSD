import argparse

def parse_args():

    
    parser = argparse.ArgumentParser(description='Arguments')

    ## Manager args ##
    parser.add_argument('--shuffle_cl', type=bool, default=False)
    parser.add_argument('--dataset_name', type=str, default='imagenet100')
    parser.add_argument('--dataset_path', type=str, default='./data')
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--base_task_class_num', type=int, default=10)
    parser.add_argument('--total_task', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_blip', action='store_true', help='Use BLIP for training')
    parser.add_argument('--v2_desc', action='store_true', help='Use v2 descriptions')
    parser.add_argument('--c_resolution', type=int, default=32)
    parser.add_argument('--c_batch_size', type=int, default=32, help='Batch size for current task')
    parser.add_argument('--c_lr', type=float, default=2e-4, help='Learning rate for current task')
    parser.add_argument('--c_epochs', type=int, default=50, help='Number of training epochs for current task')
    parser.add_argument('--c_wd', type=float, default=0.9, help='Weight decay for current task')
    parser.add_argument('--c_ema', action='store_true', help='Use EMA for training')
    parser.add_argument('--c_sam', action='store_true', help='Use SAM for training')
    parser.add_argument('--c_zap_p', type=float, default=None, help='Probability of zapping weights')
    parser.add_argument('--c_dino', action='store_true', help='Use DINO for generation')
    parser.add_argument('--clip_grad_norm', type=float, default=None, help='Clip gradient norm')
    parser.add_argument('--trainer', type=str, default='normal', choices=['normal', 'dino', 'lucir', 'anneal', 'anneal+','aug'], help='Trainer to use')
    parser.add_argument('--joint_init', action='store_true', help='Initialize joint model')
    parser.add_argument('--syn_only', action='store_true', help='Only use synthetic data for training')
    parser.add_argument('--anti_discrim', action='store_true', help='Use anti-discriminator')
    parser.add_argument('--c_anneal_epochs', type=int, default=5, help='Number of annealing epochs')
    parser.add_argument('--init_option', type=str, default='random', choices=['random','old', 'new', 'mean'], help='Initialization option')
    parser.add_argument('--use_c_l1', action='store_true', help='Use L1 loss for training')


    parser.add_argument('--n_replay', type=int, default=100, help='Number of replay samples per class')
    parser.add_argument('--n_aug', type=int, default=10, help='Number of augmentation samples per class')
    parser.add_argument('--max_replay_set_size', type=int, default=50000, help='Maximum size of replay set')
    parser.add_argument('--max_gen_batch_size', type=int, default = 16, help='Maximum batch size for generation')
    parser.add_argument('--num_inference_steps', type=int, default=20, help='Number of inference steps for generation')
    parser.add_argument("--prepared_dataset_path", type=str, default=None, help="Path to prepared dataset (if available)")
    parser.add_argument("--shared_gen_replay", action='store_true', help="Use shared generator for replay samples")
    parser.add_argument("--prepared_gen_dataset_path", type=str, default=None, help="Path to prepared dataset for generator (if available)")
    parser.add_argument("--prepared_cpsd_path", type=str, default=None, help="Path to prepared CPSD model (if available)")

    ## Main args ##
    parser.add_argument('--project_name', type=str, default='CLIP-Projection')
    parser.add_argument('--run_name', type=str, default='debug')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='/home/ilee0022/cl-gen/models/miniSD-diffusers')
    parser.add_argument('--model_name', type=str, default='miniSD-diffusers')
    parser.add_argument('--method', type=str, default='cpsd', choices=['cpsd', 'cpsd+', 'cpsd_cont', 'replay', 'cpsd2', 'sd','ti'], help='Method to use for training')

    ## CPSD args ##
    # data
    parser.add_argument('--image_column', type=str, default='image', help='Column name for images in dataset')
    parser.add_argument('--caption_column', type=str, default='text', help='Column name for captions in dataset')
    parser.add_argument('--desc_column', type=str, default='desc', help='Column name for descriptions in dataset')

    # Training parameters
    parser.add_argument('--cpsd_gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--mixed_precision', type=str, default='no', choices=['no', 'fp16', 'bf16'], help='Mixed precision training')
    parser.add_argument('--cpsd_learning_rate', type=float, default=1e-4, help='Learning rate for CPSD')
    parser.add_argument('--cpsd_batch_size', type=int, default=4, help='Batch size for CPSD training')
    parser.add_argument('--cpsd_adam_beta1', type=float, default=0.9, help='Beta1 for Adam optimizer')
    parser.add_argument('--cpsd_adam_beta2', type=float, default=0.999, help='Beta2 for Adam optimizer')
    parser.add_argument('--cpsd_adam_weight_decay', type=float, default=0.01, help='Weight decay for Adam optimizer')
    parser.add_argument('--cpsd_adam_epsilon', type=float, default=1e-8, help='Epsilon for Adam optimizer')
    parser.add_argument('--cpsd_resolution', type=int, default=256, help='Resolution for input images')
    parser.add_argument('--cpsd_ema_alpha', type=float, default=0.5, help='EMA alpha for model')
    
    parser.add_argument('--cpsd_lr_scheduler', type=str, default='constant', help='Learning rate scheduler')
    parser.add_argument('--cpsd_lr_warmup_steps', type=int, default=0, help='Warmup steps for learning rate scheduler')
    parser.add_argument('--cpsd_num_train_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--cpsd_max_grad_norm', type=float, default=1.0, help='Max gradient norm for clipping')
    parser.add_argument('--cpsd_snr_gamma', type=float, help='SNR gamma for loss computation')
    parser.add_argument('--cpsd_dist_match', type=float, default=0.003, help='Distribution matching weight for loss')
    parser.add_argument('--cpsd_scale_lr', action='store_true', help='Scale learning rate by number of GPUs, gradient accumulation steps, and batch size')
    parser.add_argument('--cpsd_ema', action='store_true', help='Use EMA for training')
    parser.add_argument('--cpsd_contrastive_loss', type=float, default=None, help='Contrastive loss weight')
    parser.add_argument('--cpsd_scheduler', type=str, default='pndm', help='Scheduler to use')
    parser.add_argument('--cpsd_max_steps', type=int, default=None, help='Maximum number of steps')

    ## Backbone ##
    parser.add_argument('--dataloader_num_workers', type=int, default=4, help='Number of workers for dataloader')

    


    args = parser.parse_args()

    return args