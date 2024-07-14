from utils.parse_args import parse_args
import time
import os
import wandb
import torch
import numpy as np
from manager.manager import Manager
from cpsd.cpsd import CPSDPipeline, CPSDPlusPipeline, CPSDContPipeline
from diffusers import DPMSolverMultistepScheduler
from utils.logger import Logger
from cpsd.train_cpsd import train_cpsd
from cpsd.train_cpsd_plus import train_cpsd_plus
from cpsd.train_cpsd_cont import train_cpsd_cont
from backbone.backbone import Backbone, DINOBackbone, LUCIRBackbone
from backbone.train import Trainer, DINOTrainer


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



wandb.init(project=args.project_name, name=args.run_name)
wandb.log(vars(args))

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


if args.trainer == 'dino':
    backbone = DINOBackbone(args.num_classes, pipeline, args.c_resolution).to(device)
    trainer = DINOTrainer(args, backbone, device)
else:
    if args.trainer == 'lucir':
        backbone = LUCIRBackbone(args.num_classes).to(device)
    else:
        backbone = Backbone(args.num_classes).to(device)
    trainer = Trainer(args, backbone, device)

test_dataloader_list = []
results, results_mask_classes = [], []
logger = Logger(setting_str = 'class-il', dataset_str = args.dataset_name, model_str = args.model_name)

prev_current_ids = []
test_loader_list = []
class_range = 0
batch_size = args.c_batch_size
for task_id in range(args.total_task):
    print(f'\nTask {task_id} starts.... \n')

    if args.trainer == 'lucir':
        backbone.add_task(len(task_ids[task_id]), device)

    class_range += len(task_ids[task_id])
    backbone.set_class_range(class_range)
    current_task_class_ids = task_ids[task_id]
    

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


    print(f'Training Backbone for Task {task_id}')

    if args.joint_init:
        batch_size = args.c_batch_size // 2
        if args.shared_gen_replay:
            replay_ids = current_task_class_ids
        else:
            replay_ids = prev_current_ids + current_task_class_ids

        gen_train_loader, gen_val_loader = dataset_manager.get_gen_dataloader(replay_ids, pipeline, task_id, batch_size = batch_size)
        train_loader, val_loader, test_loader = dataset_manager.get_current_task_dataloader(current_task_class_ids, batch_size)


    elif task_id > 0:
        batch_size = args.c_batch_size // 2

        if args.method == 'replay':
            gen_train_loader, gen_val_loader, _ = dataset_manager.get_current_task_dataloader(prev_current_ids, max_samples=args.n_replay, batch_size = batch_size)
        else:

            if args.shared_gen_replay:
                replay_ids = current_task_class_ids
            else:
                replay_ids = prev_current_ids
                
            gen_train_loader, gen_val_loader = dataset_manager.get_gen_dataloader(replay_ids, pipeline, task_id, batch_size = batch_size)

    
    else:
        gen_train_loader, gen_val_loader = None, None
        train_loader, val_loader, test_loader = dataset_manager.get_current_task_dataloader(current_task_class_ids, batch_size)


    test_loader_list.append(test_loader)

    trainer.train(task_id, task_ids, train_loader, val_loader, test_loader_list, gen_train_loader, gen_val_loader)

    prev_current_ids.extend(current_task_class_ids)
    

logger.add_bwt(results, results_mask_classes)
logger.add_forgetting(results, results_mask_classes)


logger.write(vars(args), args.output_dir)

d = logger.dump()
wandb.log(d)