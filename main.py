from utils.parse_args import parse_args
import time
import os
import wandb
import torch
import numpy as np
from manager.manager import Manager
from diffusers import StableDiffusionPipeline
from cpsd.cpsd import CPSDPipeline, CPSDPlusPipeline, CPSDContPipeline, CPSD2Pipeline
from diffusers import DPMSolverMultistepScheduler
from utils.logger import Logger
from cpsd.train_cpsd import train_cpsd
from cpsd.train_cpsd_plus import train_cpsd_plus
from cpsd.train_cpsd_cont import train_cpsd_cont
from cpsd.train_cpsd_2 import train_cpsd_2
from backbone.backbone import Backbone, LUCIRBackbone, AnnealingBackbone
from backbone.train import Trainer, AnnealTrainer, AnnealTrainerPlus, AugTrainer, AugAnnealTrainer
from torchvision import transforms
from PIL import Image

os.environ['wandb_api_key'] = '67265bb3f10a02ce2167c5006180fd57e2598daa'
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

elif args.method == 'cpsd2':
    print("Using CPSD2")
    pipeline = CPSD2Pipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.float16).to(device)

else:
    pipeline = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.float16).to(device)


if pipeline is not None:
    if args.cpsd_scheduler == 'DPMSolver':
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    #pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.set_progress_bar_config(disable=True)


# if args.trainer == 'dino':
#     backbone = DINOBackbone(args.num_classes, pipeline, args.c_resolution).to(device)
#     trainer = DINOTrainer(args, backbone, device)

if args.trainer == 'anneal' or args.trainer == 'aug':
    backbone = AnnealingBackbone(args, args.num_classes, args.anti_discrim, args.init_option).to(device)

    if args.trainer == 'aug':
        trainer = AugAnnealTrainer(args, backbone, device)

    else:
        trainer = AnnealTrainer(args, backbone, device)

elif args.trainer == 'anneal+':
    backbone = AnnealingBackbone(args, args.num_classes, args.anti_discrim, args.init_option, True).to(device)

    
    trainer = AnnealTrainerPlus(args, backbone, device)

else:
    if args.trainer == 'lucir':
        backbone = LUCIRBackbone(args, args.num_classes, zap_p = args.c_zap_p).to(device)

    else:
        backbone = Backbone(args, args.num_classes).to(device)

    # if args.trainer == 'aug':
    #     trainer = AugTrainer(args, backbone, device)


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
    
    if args.prepared_gen_dataset_path is None and args.prepared_cpsd_path is None:

        for class_id in current_task_class_ids:
            print(f'Training CLIP Projection for Class {class_id}')

            if args.method == 'cpsd':
                train_cpsd(args, cl_dataset_path+f'/class_{class_id}', class_id)

            elif args.method == 'cpsd+':
                train_cpsd_plus(args, cl_dataset_path+f'/class_{class_id}', class_id)
                

            elif args.method == 'cpsd_cont':
                train_cpsd_cont(args, cl_dataset_path+f'/class_{class_id}', class_id)

            elif args.method == 'cpsd2':
                train_cpsd_2(args, cl_dataset_path+f'/class_{class_id}', class_id)
                
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

    elif args.syn_only:
        
        train_loader, val_loader = dataset_manager.get_gen_dataloader(current_task_class_ids, pipeline, task_id, batch_size = batch_size)
        _, _, test_loader = dataset_manager.get_current_task_dataloader(current_task_class_ids, batch_size)
        gen_train_loader, gen_val_loader = None, None

    elif args.trainer == 'anneal' or args.trainer == 'aug':

        batch_size = args.c_batch_size
        if args.shared_gen_replay and not args.prepared_gen_dataset_path:
            replay_ids = current_task_class_ids

            if task_id == 0:

                _, _ = dataset_manager.get_gen_dataloader(replay_ids, pipeline, task_id, batch_size = batch_size) # just to let it generate the replay samples
                gen_train_loader, gen_val_loader = None, None
        else:
            replay_ids = prev_current_ids + current_task_class_ids

        if task_id > 0:
            gen_train_loader, gen_val_loader = dataset_manager.get_gen_dataloader(replay_ids, pipeline, task_id, batch_size = batch_size) #####
        else:
            gen_train_loader, gen_val_loader = None, None

        if args.trainer == 'aug':
            batch_size = args.c_batch_size // 2

        train_loader, val_loader, test_loader = dataset_manager.get_current_task_dataloader(current_task_class_ids, batch_size)
        if args.trainer == 'aug':
            
            resize_transform = transforms.Resize((args.cpsd_resolution, args.cpsd_resolution), antialias=True, interpolation=Image.BILINEAR)
            def custom_collate_fn(batch):
                transformed_batch = []
                for sample in batch:
                    pixel_values = sample['pixel_values']
                    cl_label = sample['cl_label']
                    
                    # Apply the resize transformation
                    transformed_pixel_values = resize_transform(pixel_values)
                    
                    transformed_sample = {
                        'pixel_values': transformed_pixel_values,
                        'cl_label': cl_label
                    }
                    transformed_batch.append(transformed_sample)
                batch = {key: torch.stack([d[key] for d in transformed_batch]) for key in transformed_batch[0]}
                return batch
                
            dataset = train_loader.dataset
            batch_size = train_loader.batch_size
            num_workers = train_loader.num_workers
            pin_memory = train_loader.pin_memory

            prep_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, collate_fn=custom_collate_fn)

            aug_train_loader = dataset_manager.get_aug_dataloader(prev_current_ids + current_task_class_ids, pipeline, task_id, prep_loader, batch_size = batch_size) 

    elif args.trainer == 'anneal+':
        batch_size = args.c_batch_size // 2
        if args.shared_gen_replay and not args.prepared_gen_dataset_path:
            replay_ids = current_task_class_ids

        else:
            replay_ids = prev_current_ids + current_task_class_ids

        gen_train_loader, gen_val_loader = dataset_manager.get_gen_dataloader(replay_ids, pipeline, task_id, batch_size = batch_size)
 
        train_loader, val_loader, test_loader = dataset_manager.get_current_task_dataloader(current_task_class_ids, batch_size)

    # elif args.trainer == 'aug':
    #     batch_size = args.c_batch_size // 2 
    #     aug_ids = prev_current_ids + current_task_class_ids
    #     train_loader, val_loader, test_loader = dataset_manager.get_current_task_dataloader(current_task_class_ids, batch_size)
    #     aug_train_loader = dataset_manager.get_aug_dataloader(aug_ids, pipeline, task_id, train_loader, batch_size = batch_size) 
    #     gen_train_loader, gen_val_loader = None, None

    elif task_id > 0:
        batch_size = args.c_batch_size // 2

        if args.method == 'replay':
            gen_train_loader, gen_val_loader, _ = dataset_manager.get_current_task_dataloader(prev_current_ids, max_samples=args.n_replay, batch_size = batch_size)
        else:


            replay_ids = prev_current_ids

            gen_train_loader, gen_val_loader = dataset_manager.get_gen_dataloader(replay_ids, pipeline, task_id, batch_size = batch_size)
            train_loader, val_loader, test_loader = dataset_manager.get_current_task_dataloader(current_task_class_ids, batch_size)

    
    else:
        
        gen_train_loader, gen_val_loader = None, None
        train_loader, val_loader, test_loader = dataset_manager.get_current_task_dataloader(current_task_class_ids, batch_size)

    


    test_loader_list.append(test_loader)

    if args.trainer == 'aug':
        accs = trainer.train(task_id, task_ids, train_loader, val_loader, test_loader_list, gen_train_loader, gen_val_loader, aug_train_loader)

    else:
        accs = trainer.train(task_id, task_ids, train_loader, val_loader, test_loader_list, gen_train_loader, gen_val_loader)

    results.append(accs[0])
    results_mask_classes.append(accs[1])

    trainer.end_task()

    prev_current_ids.extend(current_task_class_ids)

    torch.cuda.empty_cache()
    

logger.add_bwt(results, results_mask_classes)
logger.add_forgetting(results, results_mask_classes)


logger.write(vars(args), args.output_dir)

d = logger.dump()
wandb.log(d)
print(d)