import os
import json
import numpy as np
from datasets import load_from_disk, Dataset, load_dataset
import tqdm
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from copy import deepcopy
from torchvision import transforms
import pandas as pd
import random
from cpsd.cpsd import CPSDPipeline
import math
from PIL import Image

class Manager:

    def __init__(self, args, device):
        self.tasks_ids = None
        self.class2cl = None
        self.args = args
        self.shuffle_cl = args.shuffle_cl
        self.dataset_name = args.dataset_name
        self.dataset_path = args.dataset_path
        self.num_classes = args.num_classes
        self.output_dir = args.output_dir
        self.base_task_class_num = args.base_task_class_num
        self.total_task = args.total_task
        self.seed = args.seed

        self.load_dataset()

        self.tasks_ids, self.class2cl = self.class_task_partition(self.base_task_class_num, self.total_task)

        with open(args.dataset_path + '/label2text.json') as f:
            self.label2text = json.load(f)
        self.use_blip = args.use_blip

        if args.use_blip:
            self.processor = Blip2Processor.from_pretrained('Salesforce/blip2-opt-2.7b', cache_dir='.cache')
            self.blip = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b', cache_dir='.cache', torch_dtype=torch.float16).to(device)

        self.device = device

        self._train_transforms = transforms.Compose([transforms.Resize((args.c_resolution, args.c_resolution), antialias=True), 
                                                     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1), 
                                                     transforms.RandomHorizontalFlip(p=0.5), 
                                                     transforms.RandomResizedCrop(args.c_resolution, scale=(0.6, 1.0), interpolation=Image.BICUBIC), 
                                                     transforms.ToTensor(), 
                                                     transforms.Normalize(mean=[0.5], std=[0.5])])
        
        self._val_transforms = transforms.Compose([transforms.Resize((args.c_resolution, args.c_resolution), antialias=True), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])

        def preprocess(examples, label_map, transform):
            if len(examples.keys()) == 1:
                if 'pixel_value' in examples:
                    processed_images = []
                    for image in examples['image']:
                        processed_images.append(transform(image.convert('RGB')))
                    examples['pixel_values'] = torch.stack(processed_images)
                    examples.pop('image', None)
                    return examples
                else:
                    if 'cl_label' in examples:
                        examples['cl_label'] = torch.tensor([label_map[label] for label in examples['label']])
                        return examples
                    if 'label' in examples:
                        return examples
            processed_images = []
            for image in examples['image']:
                processed_images.append(transform(image.convert('RGB')))
            examples['pixel_values'] = torch.stack(processed_images)
            examples.pop('image', None)
            examples['cl_label'] = torch.tensor([label_map[label] for label in examples['label']])
            return examples
        
        self.preprocess = preprocess

        self.prepared_dataset_path = args.prepared_dataset_path

    def get_ids_info(self):
        return (self.tasks_ids, self.class2cl)

    def get_cl_dataset_path(self):
        if self.prepared_dataset_path is None:
            self.prepare_clip_proj_dataset()
            CL_dataset_dir = self.output_dir + '/CI_dataset'
            return CL_dataset_dir
        CL_dataset_dir = self.prepared_dataset_path
        return CL_dataset_dir

    def get_current_task_dataset(self, current_task_ids, max_samples=None):
        task_dataset = {}
        for phase in ['train', 'validation', 'test']:
            indices = torch.nonzero(torch.isin(torch.tensor(self.dataset[phase]['label']), torch.tensor(current_task_ids))).squeeze()
            if max_samples is not None:
                indices = indices[torch.randperm(len(indices))[:max_samples]]
            task_dataset[phase] = deepcopy(self.dataset[phase].select(indices))
        task_dataset = self.dataset_preparation(task_dataset)
        return task_dataset

    def get_current_task_dataloader(self, current_task_ids, batch_size, max_samples=None):
        print('Preparing dataloader for current task')
        task_dataset = self.get_current_task_dataset(current_task_ids, max_samples=max_samples)
        task_dataloader = {}
        for phase in ['train', 'validation', 'test']:
            if phase not in task_dataset:
                continue
            task_dataloader[phase] = torch.utils.data.DataLoader(task_dataset[phase], batch_size=batch_size, shuffle=True, num_workers=self.args.dataloader_num_workers)
        return (task_dataloader['train'], task_dataloader['validation'], task_dataloader['test'])

    def dataset_preparation(self, dataset):
        for phase in ['train', 'validation', 'test']:
            if phase not in dataset:
                continue
            if phase == 'train':
                dataset[phase].set_transform(lambda examples: self.preprocess(examples, self.class2cl, self._train_transforms))
            else:
                dataset[phase].set_transform(lambda examples: self.preprocess(examples, self.class2cl, self._val_transforms))
        return dataset

    def class_task_partition(self, base_task_class_num, total_task):
        set_sizes = [0] * total_task
        set_sizes[0] = base_task_class_num
        for i in range(1, total_task):
            set_sizes[i] = (self.num_classes - base_task_class_num) // (total_task - 1)
        class_ids = list(range(self.num_classes))
        if self.shuffle_cl:
            np.random.shuffle(class_ids)
        task_ids = []
        for i in range(len(set_sizes)):
            task_ids.append(class_ids[sum(set_sizes[:i]):sum(set_sizes[:i + 1])])
        task_ids = {i: task_ids[i] for i in range(len(task_ids))}
        class_to_cl_idx = {}
        count = 0
        for i, task_id in task_ids.items():
            for class_id in task_id:
                class_to_cl_idx[class_id] = count
                count += 1
        if not os.path.exists(self.output_dir + '/CI_dataset'):
            os.makedirs(self.output_dir + '/CI_dataset')
        with open(self.output_dir + '/CI_dataset/class_to_cl_idx.json', 'w') as f:
            json.dump(class_to_cl_idx, f)

        with open(self.output_dir + '/CI_dataset/task_ids.json', 'w') as f:
            json.dump(task_ids, f)

        return task_ids, class_to_cl_idx

    def load_dataset(self):
        self.dataset = load_from_disk(self.dataset_path)
        if 'validation' not in self.dataset:
            temp = self.dataset['train'].train_test_split(test_size=0.1, seed=self.seed, shuffle=True)
            self.dataset['validation'] = temp['test']
            self.dataset['train'] = temp['train']
            if 'image' not in self.dataset['train'].column_names:
                self.dataset['train'] = self.dataset['train'].rename_column('img', 'image')
                self.dataset['validation'] = self.dataset['validation'].rename_column('img', 'image')
                self.dataset['test'] = self.dataset['test'].rename_column('img', 'image')
            if 'text' not in self.dataset['train'].column_names:
                self.dataset['train'] = self.dataset['train'].rename_column('fine_label', 'text')
                self.dataset['validation'] = self.dataset['validation'].rename_column('fine_label', 'text')
                self.dataset['test'] = self.dataset['test'].rename_column('fine_label', 'text')
        self.val_ratio = len(self.dataset['validation']) / len(self.dataset['train'])

    def sample_clip_proj(self, prev_current_task_class_ids, pipeline: CPSDPipeline, task_id):
        n_rep_per_class = self.args.n_replay // len(prev_current_task_class_ids)

        if self.args.shared_gen_replay:
            replay_dir = self.args.output_dir + f'/gen_samples'
        else:
            replay_dir = self.args.output_dir + f'/gen_samples/task_{task_id}'
        if not os.path.isdir(replay_dir):
            os.makedirs(replay_dir)

        filename = []
        labels = []
        text = []
        guidance_scale = []
        for class_id in prev_current_task_class_ids:
            pipeline.load_embed_trans(self.args.output_dir + f'/cp_embeddings/class_{class_id}.pt')
            prompts = [f"{self.label2text[str(class_id)].split(',')[0]}, {random.choice(self.unique_desc[class_id])}" for _ in range(n_rep_per_class)]
            replay_path = replay_dir + f'/class_{class_id}'
            generated = 0
            print(f'Generating {len(prompts)} samples for class {class_id}.....')
            while generated < len(prompts):
                batch_size = min(self.args.max_gen_batch_size, len(prompts) - generated)
                rnd_guidance_scale = np.random.uniform(3, 9)
                guidance_scale.extend([rnd_guidance_scale] * batch_size)
                images = pipeline(prompts[generated:generated + batch_size], num_inference_steps=self.args.num_inference_steps, width=self.args.cpsd_resolution, height=self.args.cpsd_resolution, guidance_scale=rnd_guidance_scale).images
                for i, img in enumerate(images):
                    file = f'replay_{generated + i}.jpeg'
                    filename.append(f'class_{class_id}_{file}')
                    labels.append(class_id)
                    text.append(self.label2text[str(class_id)].split(',')[0])
                    img.save(replay_path + f'_{file}')
                generated += batch_size
                print(f'Generated {generated} samples')
        metadata = {'file_name': filename, 'label': labels, 'text': text, 'ucg': guidance_scale}
        metadata = pd.DataFrame(metadata)

        # check if metadata file already exists
        if os.path.exists(replay_dir + '/metadata.csv'):
            metadata = pd.concat([pd.read_csv(replay_dir + '/metadata.csv'), metadata], ignore_index=True)
        else:
            metadata.to_csv(replay_dir + '/metadata.csv', index=False)
        return replay_dir

    def prepare_gen_dataset(self, prev_current_task_class_ids, pipeline: CPSDPipeline, task_id):
        print(f'Preparing dataset for CLIP projection for task {task_id}....')
        replay_dir = self.sample_clip_proj(prev_current_task_class_ids, pipeline, task_id)
        dataset = load_dataset('imagefolder', data_dir=replay_dir)
        dataset = dataset['train'].train_test_split(test_size=self.val_ratio, seed=self.args.seed, shuffle=True)
        dataset['validation'] = dataset['test']
        dataset.pop('test')
        dataset = self.dataset_preparation(dataset)
        return dataset

    def get_gen_dataloader(self, prev_current_task_class_ids, pipeline: CPSDPipeline, task_id, batch_size):
        gen_dataset = self.prepare_gen_dataset(prev_current_task_class_ids, pipeline, task_id)
        gen_dataloader = {}
        for phase in ['train', 'validation']:
            if phase not in gen_dataset:
                continue
            gen_dataloader[phase] = torch.utils.data.DataLoader(gen_dataset[phase], batch_size=batch_size, shuffle=True, num_workers=self.args.dataloader_num_workers)
        return (gen_dataloader['train'], gen_dataloader['validation'])

    def prepare_clip_proj_dataset(self):
        print('Preparing dataset for CLIP projection...')
        if self.use_blip:
            self.unique_desc = {}
        if not os.path.exists(self.output_dir + '/CI_dataset'):
            os.makedirs(self.output_dir + '/CI_dataset')
        for i in tqdm.tqdm(range(self.num_classes)):
            indices = torch.nonzero(torch.tensor(self.dataset['train']['label']) == i).squeeze()
            images = self.dataset['train'][indices]['image']
            text = self.label2text[str(i)].split(',')[0]
            if self.use_blip:
                desc = []
                nbatch = math.ceil(len(images) / 32)
                for j in range(nbatch):
                    batch_images = images[j * 32:(j + 1) * 32]
                    batch_text = [f'{text},'] * len(batch_images)
                    inputs = self.processor(batch_images, batch_text, return_tensors='pt').to(self.device, torch.float16)
                    outputs = self.blip.generate(**inputs)
                    decoded_outputs = self.processor.batch_decode(outputs, skip_special_tokens=True)
                    desc.extend((out.strip() for out in decoded_outputs))
                data = {'image': images, 'text': [text] * len(images), 'label': [self.class2cl[i]] * len(images), 'desc': desc}
                task_dataset = Dataset.from_dict(data)
                task_dataset.save_to_disk(self.output_dir + f'/CI_dataset/class_{i}')
                self.unique_desc[i] = list(set(desc))
                with open(self.output_dir + f'/CI_dataset/class_{i}/unique_desc.json', 'w') as f:
                    json.dump(self.unique_desc[i], f)
            else:
                data = {'image': images, 'text': [text] * len(images), 'label': [self.class2cl[i]] * len(images)}
                task_dataset = self.dataset.from_dict(data)
                task_dataset.save_to_disk(self.output_dir + f'/CI_dataset/class_{i}')

    def load_unique_desc(self, prepared_dataset_path):
        print('Loading unique descriptions for each class...')
        self.unique_desc = {}
        for i in range(self.num_classes):
            with open(prepared_dataset_path + f'/class_{i}/unique_desc.json') as f:
                self.unique_desc[i] = json.load(f)