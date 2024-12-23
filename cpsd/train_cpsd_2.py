import logging
import math
import os
import datasets


import PIL
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
import accelerate
from accelerate.state import AcceleratorState

# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    UNet2DConditionModel,
)
from diffusers.training_utils import compute_snr
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from datasets import load_from_disk

# from textual_inversion.cpsd import ClipProj

if is_wandb_available():
    import wandb

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.27.0.dev0")

logger = get_logger(__name__)

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, hidden_states):
        # check if hidden states is in (bs, -1)
        if hidden_states.dim() != 2:
            hidden_states = hidden_states.view(hidden_states.size(0), -1)

        batch_size = hidden_states.size(0)
        pairwise_distances = torch.cdist(hidden_states, hidden_states, p=2)  # Compute pairwise distances

        # Create a mask to ignore the diagonal (distance with itself)
        mask = torch.eye(batch_size, device=hidden_states.device).bool()
        pairwise_distances = pairwise_distances[~mask].view(batch_size, -1)

        # Compute the contrastive loss
        contrastive_loss = F.relu(self.margin - pairwise_distances).mean()

        return contrastive_loss

class CosineContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=0.5):
        super(CosineContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, hidden_states):
        # Check if hidden states is in (bs, -1)
        if hidden_states.dim() != 2:
            hidden_states = hidden_states.view(hidden_states.size(0), -1)

        batch_size = hidden_states.size(0)

        # Normalize hidden states
        hidden_states = F.normalize(hidden_states, p=2, dim=1)

        # Compute pairwise cosine similarity
        cosine_similarity = torch.mm(hidden_states, hidden_states.t())

        # Convert to cosine distance
        cosine_distance = 1 - cosine_similarity

        # Create a mask to ignore the diagonal (distance with itself)
        mask = torch.eye(batch_size, device=hidden_states.device).bool()
        cosine_distance = cosine_distance[~mask].view(batch_size, -1)

        # Compute the contrastive loss
        contrastive_loss = F.relu(self.margin - cosine_distance).mean()

        return contrastive_loss

def train_cpsd_2(args, train_data_dir, class_id):

    logging_dir = os.path.join(args.output_dir, 'cpsd_logs')

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.cpsd_gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with='tensorboard',
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer")

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae")

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )

    # Freeze vae and text_encoder and unet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)


    if args.cpsd_scale_lr:
        learning_rate = (
            args.cpsd_learning_rate * args.cpsd_gradient_accumulation_steps * args.cpsd_batch_size * accelerator.num_processes
        )

    optimizer_cls = torch.optim.AdamW

    embed_trans = torch.nn.Linear(768,768, bias = True)
    with torch.no_grad():  
        embed_trans.weight.copy_(torch.eye(768))  # Set weights to identity matrix
        embed_trans.bias.zero_()  # Set bias to zero


    embed_trans.to(accelerator.device)
    optimizer = optimizer_cls(
        embed_trans.parameters(),
        lr=learning_rate,
        betas=(args.cpsd_adam_beta1, args.cpsd_adam_beta2),
        weight_decay=args.cpsd_adam_weight_decay,
        eps=args.cpsd_adam_epsilon,
    )

    dataset = load_from_disk(train_data_dir)
    column_names = dataset.column_names


    image_column = args.image_column
    caption_column = args.caption_column
    desc_column = args.desc_column

    if image_column not in column_names:
        raise ValueError(f"Image column {image_column} not found in the dataset.")
    if caption_column not in column_names:
        raise ValueError(f"Caption column {caption_column} not found in the dataset.")

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples):
        captions = []

        if args.use_blip:
            for caption, desc in zip(examples[caption_column], examples[desc_column]):
                captions.append(f"{caption}, {desc}")
        else:
            for caption in examples[caption_column]:

                captions.append(caption)

        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.cpsd_resolution, antialias=True),
            transforms.RandomResizedCrop(args.c_resolution, scale=(0.6, 1.0), interpolation=PIL_INTERPOLATION["lanczos"]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    with accelerator.main_process_first():
        # Set the training transforms
        train_dataset = dataset.with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples], dtype=torch.long)
        return {"pixel_values": pixel_values, "input_ids": input_ids, "labels": labels}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.cpsd_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.cpsd_gradient_accumulation_steps)


    lr_scheduler = get_scheduler(
        args.cpsd_lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.cpsd_lr_warmup_steps * accelerator.num_processes,
        num_training_steps=num_update_steps_per_epoch * args.cpsd_num_train_epochs,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler, embed_trans = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler, embed_trans
    )


    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
        # if args.classifier_guidance:
        #     classifier.net = classifier.net.half()
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)



    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers('cpsd', tracker_config)


    # Train!
    total_batch_size = args.cpsd_batch_size * accelerator.num_processes * args.cpsd_gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.cpsd_num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.cpsd_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.cpsd_gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.cpsd_num_train_epochs * num_update_steps_per_epoch}")
    global_step = 0
    first_epoch = 0


    initial_global_step = 0


    progress_bar = tqdm(
        range(0, args.cpsd_num_train_epochs * num_update_steps_per_epoch),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    contrastive_loss_fn = CosineContrastiveLoss(margin=0.2)
    for epoch in range(first_epoch, args.cpsd_num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()


                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]
                encoder_hidden_states = embed_trans(encoder_hidden_states.reshape(-1, 768)).reshape(encoder_hidden_states.shape) 



                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")


                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
                


                if args.cpsd_snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.cpsd_snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                
                dist_loss = 0
                if args.cpsd_dist_match:
                    mse_loss_weights = mse_loss_weights.view(bsz, 1, 1, 1)
                    model_pred_ws = (model_pred.float() * mse_loss_weights).sum(dim=0)
                    target_ws = (target.float() * mse_loss_weights).sum(dim=0)
                    dist_loss = F.mse_loss(model_pred_ws, target_ws, reduction="mean") 
                    loss = loss + dist_loss * args.cpsd_dist_match

                contrastive_loss = 0
                if args.cpsd_contrastive_loss:
                    contrastive_loss = contrastive_loss_fn(encoder_hidden_states)
                    loss += contrastive_loss * args.cpsd_contrastive_loss

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.cpsd_batch_size)).mean()
                train_loss += avg_loss.item() / args.cpsd_gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.cpsd_max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0


            logs = {"step_loss": loss.detach().item(), "dist_loss": dist_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)


    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.output_dir + f'/cp_embeddings' is not None:
            os.makedirs(args.output_dir + f'/cp_embeddings', exist_ok=True)

        torch.save(embed_trans, args.output_dir + f'/cp_embeddings/class_{class_id}.pt')

    accelerator.end_training()