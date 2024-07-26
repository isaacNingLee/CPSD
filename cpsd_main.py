from utils.parse_args import parse_args
import time
import os
import torch
import numpy as np
from manager.manager import Manager
from cpsd.cpsd import CPSDPipeline, CPSDPlusPipeline, CPSDContPipeline
from diffusers import DPMSolverMultistepScheduler
from cpsd.train_cpsd import train_cpsd
from cpsd.train_cpsd_plus import train_cpsd_plus
from cpsd.train_cpsd_cont import train_cpsd_cont

args = parse_args()

args.output_dir = args.output_dir + f'/{time.strftime("%Y%m%d-%H%M%S")}' + f'_{args.run_name}'
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# fix all seeds and random process
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
generator_seed = torch.Generator().manual_seed(args.seed)

dataset_manager = Manager(args, device)
task_ids, _ = dataset_manager.get_ids_info()

if args.prepared_dataset_path is None:
    cl_dataset_path = dataset_manager.get_cl_dataset_path()
else:
    cl_dataset_path = args.prepared_dataset_path
    dataset_manager.load_unique_desc(cl_dataset_path)

if args.method == 'cpsd':
    pipeline = CPSDPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.float16).to(device)

elif args.method == 'cpsd+':
    pipeline = CPSDPlusPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.float16).to(device)

elif args.method == 'cpsd_cont':
    pipeline = CPSDContPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.float16).to(device)

else:
    pipeline = CPSDPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.float16).to(device)


if pipeline is not None:
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.set_progress_bar_config(disable=True)


prev_current_ids = []
test_loader_list = []
class_range = 0
batch_size = args.c_batch_size

for task_id in range(args.total_task):
    print(f'\nTask {task_id} starts.... \n')

    current_task_class_ids = task_ids[task_id]
    
    if args.prepared_gen_dataset_path is None and args.prepared_cpsd_path is None:

        for class_id in current_task_class_ids:
            print(f'Training CLIP Projection for Class {class_id}')

            if args.method == 'cpsd':
                train_cpsd(args, cl_dataset_path+f'/class_{class_id}', class_id)

            elif args.method == 'cpsd+':
                train_cpsd_plus(args, cl_dataset_path+f'/class_{class_id}', class_id)
                

            elif args.method == 'cpsd_cont':
                train_cpsd_cont(args, cl_dataset_path+f'/class_{class_id}', class_id)

            elif args.method == 'replay':
                print("No training needed")
            torch.cuda.empty_cache() 

    if args.shared_gen_replay:
        replay_ids = current_task_class_ids
    else:
        replay_ids = prev_current_ids + current_task_class_ids

    _ , _ = dataset_manager.get_gen_dataloader(replay_ids, pipeline, task_id, batch_size = batch_size)
  