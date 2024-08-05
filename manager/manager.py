import os
import json
import numpy as np
from datasets import load_from_disk, Dataset, load_dataset, concatenate_datasets
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
from diffusers import PNDMScheduler

from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput



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


        
        self.use_blip = args.use_blip

        if args.use_blip and not args.prepared_dataset_path:
            self.processor = Blip2Processor.from_pretrained('Salesforce/blip2-opt-2.7b', cache_dir='.cache')
            self.blip = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b', cache_dir='.cache', torch_dtype=torch.float16).to(device)
        else:
            self.processor = None
            self.blip = None

        self.device = device


        if not args.c_dino:
            self._train_transforms = transforms.Compose([transforms.Resize((args.c_resolution, args.c_resolution), antialias=True, interpolation=Image.LANCZOS), 
                                                     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1), 
                                                     transforms.RandomHorizontalFlip(p=0.5), 
                                                     transforms.RandomResizedCrop(args.c_resolution, scale=(0.6, 1.0), interpolation=Image.LANCZOS, antialias=True), 
                                                     transforms.ToTensor(), 
                                                     transforms.Normalize(mean=[0.5], std=[0.5])])
        else:
            self._train_transforms = transforms.Compose([transforms.Resize((args.c_resolution, args.c_resolution), antialias=True), 
                                                     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1), 
                                                     transforms.RandomSolarize(0.2),
                                                     transforms.RandomHorizontalFlip(p=0.5), 
                                                     transforms.RandomResizedCrop(args.c_resolution, scale=(0.6, 1.0), interpolation=Image.BICUBIC), 
                                                     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05), 
                                                     transforms.RandomSolarize(0.1),
                                                     transforms.RandomResizedCrop(args.c_resolution, scale=(0.8, 1.0), interpolation=Image.BICUBIC), 
                                                     transforms.ToTensor(), 
                                                     transforms.Normalize(mean=[0.5], std=[0.5]),
                                                     transforms.RandomResizedCrop(args.c_resolution, scale=(0.2, 0.6), interpolation=Image.BICUBIC)])
        
        self._val_transforms = transforms.Compose([transforms.Resize((args.c_resolution, args.c_resolution), antialias=True, interpolation=Image.LANCZOS), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])



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
        for phase in task_dataset.keys():
            if phase not in task_dataset:
                continue
            task_dataloader[phase] = torch.utils.data.DataLoader(task_dataset[phase], batch_size=batch_size, shuffle=True, num_workers=self.args.dataloader_num_workers)


        return (task_dataloader['train'], task_dataloader['validation'], task_dataloader['test'])

    def dataset_preparation(self, dataset):


        for phase in dataset.keys():

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

        with open(self.args.dataset_path + '/label2text.json') as f:
            print('Loading label2text...')
            self.label2text = json.load(f)

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

                if 'fine_label' in self.dataset['train'].column_names:
                    self.dataset['train'] = self.dataset['train'].rename_column('fine_label', 'text')
                    self.dataset['validation'] = self.dataset['validation'].rename_column('fine_label', 'text')
                    self.dataset['test'] = self.dataset['test'].rename_column('fine_label', 'text')

                elif 'label' in self.dataset['train'].column_names:
                    # map label to text using label2text.json by adding new col
                    self.dataset['train'] = self.dataset['train'].map(lambda x: {'text': self.label2text[str(x['label'])]})
                    self.dataset['validation'] = self.dataset['validation'].map(lambda x: {'text': self.label2text[str(x['label'])]})
                    self.dataset['test'] = self.dataset['test'].map(lambda x: {'text': self.label2text[str(x['label'])]})
                    

        self.val_ratio = len(self.dataset['validation']) / len(self.dataset['train'])

    def sample_boomerang(self, prev_task_class_ids, pipeline: CPSDPipeline, task_id, train_dataloader):
        n_rep_per_class = self.args.n_replay

        # if self.args.shared_gen_replay:
        #     replay_dir = self.args.output_dir + f'/aug_samples'
        # else:
        replay_dir = self.args.output_dir + f'/aug_samples/task_{task_id}'
            
        if not os.path.isdir(replay_dir):
            os.makedirs(replay_dir, exist_ok=True)

        filename = []
        labels = []
        text = []
        guidance_scale = []
        for class_id in prev_task_class_ids:

            if self.args.method != 'sd':
                if self.args.prepared_cpsd_path:
                    pipeline.load_embed_trans(self.args.prepared_cpsd_path + f'/class_{class_id}.pt')
                else:
                    pipeline.load_embed_trans(self.args.output_dir + f'/cp_embeddings/class_{class_id}.pt')

            prompts = [f"{self.label2text[str(class_id)].split(',')[0]}, {random.choice(self.unique_desc[class_id])}" for _ in range(n_rep_per_class)]
            replay_path = replay_dir + f'/class_{class_id}'
            generated = 0
            print(f'Generating {len(prompts)} samples for class {class_id}.....')
            
            for batch in train_dataloader:

                images = batch['pixel_values'].to(self.device)
                rnd_guidance_scale = np.random.uniform(0.3, 0.99)
                
                bs = len(images) if len(images) < n_rep_per_class - generated else max(n_rep_per_class - generated, 0)

                if bs == 0:
                    break

                guidance_scale.extend([rnd_guidance_scale] * bs)

                aug_imgs = pipeline.boomerang_aug(img = images[:bs], prompt = prompts[generated:generated + bs], percent_noise = rnd_guidance_scale, num_inference_steps=50)

                for i, img in enumerate(aug_imgs):
                    file = f'replay_{generated + i}.jpeg'
                    filename.append(f'class_{class_id}_{file}')
                    labels.append(class_id)
                    text.append(self.label2text[str(class_id)].split(',')[0])
                    img.save(replay_path + f'_{file}')
                generated += len(images)
                print(f'Generated {generated} samples')


        metadata = {'file_name': filename, 'label': labels, 'text': text, 'ucg': guidance_scale}
        metadata = pd.DataFrame(metadata)

        # check if metadata file already exists
        if os.path.exists(replay_dir + '/metadata.csv'):
            metadata = pd.concat([pd.read_csv(replay_dir + '/metadata.csv'), metadata], ignore_index=True)

        metadata.to_csv(replay_dir + '/metadata.csv', index=False)

        return replay_dir
    
    def prepare_aug_dataset(self, prev_task_class_ids, pipeline: CPSDPipeline, task_id, train_dataloader):
        print(f'Preparing dataset for Boomerang augmentation for task {task_id}....')
        original_call = type(pipeline).__call__ 
        original_scheduler = pipeline.scheduler

        pipeline.scheduler = PNDMScheduler.from_config(pipeline.scheduler.config)
        
        @torch.no_grad()
        def __call__(
            self,
            prompt: Union[str, List[str]],
            percent_noise: int,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        ):
            
            # 0. Default height and width to unet
            height = height or self.unet.config.sample_size * self.vae_scale_factor
            width = width or self.unet.config.sample_size * self.vae_scale_factor

            # 1. Check inputs. Raise error if not correct
            self.check_inputs(
                prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
            )

            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            device = self._execution_device
            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = guidance_scale > 1.0

            # 3. Encode input prompt
            prompt_embeds = self._encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
            )

            # 4. Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

            # 5. Prepare latent variables
            num_channels_latents = self.unet.config.in_channels
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )

            # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # 7. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):

                    #######################################################################################
                    # BOOMERANG CODE:
                    # Skip any steps in [0, 1000] that are before (i.e., greater than) 1000 * percent noise
                    if t - 1 > 1000 * percent_noise:
                        continue
                    #print(t)
                    #######################################################################################

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)

            if output_type == "latent":
                image = latents
                has_nsfw_concept = None
            elif output_type == "pil":
                # 8. Post-processing
                image = self.decode_latents(latents)

                # 9. Run safety checker
                image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

                # 10. Convert to PIL
                image = self.numpy_to_pil(image)
            else:
                # 8. Post-processing
                image = self.decode_latents(latents)

                # 9. Run safety checker
                image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

            # Offload last model to CPU
            if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
                self.final_offload_hook.offload()

            if not return_dict:
                return (image, has_nsfw_concept)

            return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)



        type(pipeline).__call__ = __call__
        
        replay_dir = self.sample_boomerang(prev_task_class_ids, pipeline, task_id, train_dataloader)
        dataset = load_dataset('imagefolder', data_dir=replay_dir)

        dataset = self.dataset_preparation(dataset)

        for phase in dataset.keys():
            if len(dataset[phase]) > self.args.max_replay_set_size:
                indices = torch.randperm(len(dataset[phase]))[:self.args.max_replay_set_size]
                dataset[phase] = deepcopy(dataset[phase].select(indices.tolist()))

        type(pipeline).__call__ = original_call
        pipeline.scheduler = original_scheduler

        return dataset
    
    def get_aug_dataloader(self, prev_task_class_ids, pipeline: CPSDPipeline, task_id, train_dataloader, batch_size):
        

        
        aug_dataset = self.prepare_aug_dataset(prev_task_class_ids, pipeline, task_id, train_dataloader)
        
        aug_train_loader = torch.utils.data.DataLoader(aug_dataset['train'], batch_size=batch_size, shuffle=True, num_workers=self.args.dataloader_num_workers)

        


        return aug_train_loader



    def sample_clip_proj(self, prev_current_task_class_ids, pipeline: CPSDPipeline, task_id):
        n_rep_per_class = self.args.n_replay

        if self.args.shared_gen_replay:
            replay_dir = self.args.output_dir + f'/gen_samples'
        else:
            replay_dir = self.args.output_dir + f'/gen_samples/task_{task_id}'
            
        if not os.path.isdir(replay_dir):
            os.makedirs(replay_dir, exist_ok=True)

        filename = []
        labels = []
        text = []
        guidance_scale = []
        for class_id in prev_current_task_class_ids:
            
            if self.args.method != 'sd':
                if self.args.prepared_cpsd_path:
                    pipeline.load_embed_trans(self.args.prepared_cpsd_path + f'/class_{class_id}.pt')
                else:
                    pipeline.load_embed_trans(self.args.output_dir + f'/cp_embeddings/class_{class_id}.pt')

            prompts = [f"{self.label2text[str(class_id)].split(',')[0]}, {random.choice(self.unique_desc[class_id])}" for _ in range(n_rep_per_class)]
            replay_path = replay_dir + f'/class_{class_id}'
            generated = 0
            print(f'Generating {len(prompts)} samples for class {class_id}.....')
            while generated < len(prompts):
                batch_size = min(self.args.max_gen_batch_size, len(prompts) - generated)
                rnd_guidance_scale = np.random.uniform(5, 9)
                
                gen_output = pipeline(prompts[generated:generated + batch_size], num_inference_steps=self.args.num_inference_steps, width=self.args.cpsd_resolution, height=self.args.cpsd_resolution, guidance_scale=rnd_guidance_scale)
                
                images = gen_output.images
                has_error = gen_output.nsfw_content_detected
                

                for i, img in enumerate(images):

                    if not has_error[i]:
                        file = f'replay_{generated + i}.jpeg'
                        filename.append(f'class_{class_id}_{file}')
                        labels.append(class_id)
                        text.append(self.label2text[str(class_id)].split(',')[0])
                        guidance_scale.append(rnd_guidance_scale)
                        img.save(replay_path + f'_{file}')

                        generated += 1

                print(f'Generated {generated} samples')
        metadata = {'file_name': filename, 'label': labels, 'text': text, 'ucg': guidance_scale}
        metadata = pd.DataFrame(metadata)

        # check if metadata file already exists
        if os.path.exists(replay_dir + '/metadata.csv'):
            metadata = pd.concat([pd.read_csv(replay_dir + '/metadata.csv'), metadata], ignore_index=True)
        metadata.to_csv(replay_dir + '/metadata.csv', index=False)

        return replay_dir

    def prepare_gen_dataset(self, prev_current_task_class_ids, pipeline: CPSDPipeline, task_id):
        print(f'Preparing dataset for CLIP projection for task {task_id}....')

        if self.args.prepared_gen_dataset_path:
            dataset = load_dataset('imagefolder', data_dir=self.args.prepared_gen_dataset_path)

            for phase in dataset.keys():
                indices = torch.nonzero(torch.isin(torch.tensor(dataset[phase]['label']), torch.tensor(prev_current_task_class_ids))).squeeze()
                

                dataset[phase] = deepcopy(dataset[phase].select(indices))
        else:
            replay_dir = self.sample_clip_proj(prev_current_task_class_ids, pipeline, task_id)
            dataset = load_dataset('imagefolder', data_dir=replay_dir)

        dataset = dataset['train'].train_test_split(test_size=self.val_ratio, seed=self.args.seed, shuffle=True)
        dataset['validation'] = dataset['test']
        dataset.pop('test')
        dataset = self.dataset_preparation(dataset)

        for phase in dataset.keys():
            if len(dataset[phase]) > self.args.max_replay_set_size:
                indices = torch.randperm(len(dataset[phase]))[:self.args.max_replay_set_size]
                dataset[phase] = deepcopy(dataset[phase].select(indices.tolist()))

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

                    if self.args.v2_desc:
                        if random.random() > 0.67:
                            template = '{},background of'
                        elif random.random() > 0.33:
                            template = '{},with'
                        else:
                            template = '{},'
                    else:
                        template = '{},'

                    
                    batch_text = [template.format(text)] * len(batch_images)
                    add_template_text = template.split(',')[1]

                    inputs = self.processor(batch_images, batch_text, return_tensors='pt').to(self.device, torch.float16)
                    outputs = self.blip.generate(**inputs)
                    decoded_outputs = self.processor.batch_decode(outputs, skip_special_tokens=True)
                    desc.extend((f'{add_template_text} {out.strip()}' for out in decoded_outputs))


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