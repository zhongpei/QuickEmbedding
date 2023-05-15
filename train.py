import logging
import math
import os
import random


import transformers
import numpy as np
import PIL
import torch
import torch.utils.checkpoint
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)

from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available


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
#check_min_version("0.13.0.dev0")

logger = get_logger(__name__)

def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)  # 4 cuts -> shape = (4,1,768)
    y = F.normalize(y, dim=-1)  # text embed always (1,768)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


####################################################################################################


def save_progress(text_encoder, placeholder_token_id, accelerator, args, save_path, subtokens):
    if isinstance(placeholder_token_id, list):
        logger.info("Saving embeddings")
        learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id]
        sd_version = "v1" if "runway" in args.pretrained_model_name_or_path else "v2"
        learned_embeds_dict = {"placeholder_token": args.placeholder_token,
                               "subtokens" : {},
                               "SD_version":sd_version}
        for i, tok in enumerate(subtokens):
            learned_embeds_dict["subtokens"][tok] = learned_embeds[i].detach().cpu()
        torch.save(learned_embeds_dict, save_path)
    else:
        logger.info("Saving embeddings")
        learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id]
        #TODO i imagine we're going to be using custom models with this so will need a better solution
        sd_version = "v1" if "runway" in args.pretrained_model_name_or_path else "v2"
        learned_embeds_dict = {"placeholder_token": args.placeholder_token,
                               "embedding": learned_embeds.detach().cpu(),
                               "SD_version":sd_version}
        torch.save(learned_embeds_dict, save_path)




imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]

blank_template = ["{}"]

preprocess_clip = T.Compose([
    T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],std=[0.26862954, 0.26130258, 0.27577711])
])
resize224 = T.Resize((224), interpolation=InterpolationMode.BICUBIC)

def sliding_cutouts(image, num_cuts=4, cut_size=224):
    image_tensor = preprocess_clip(image)
    image_tensor = resize224(image_tensor)

    cutouts = []
    sideY, sideX = image_tensor.shape[2:4]
    largersize = max(sideX, sideY)
    if sideX < sideY:
        smaller_side = "sideX"
    else:
        smaller_side = "sideY"

    cut_spots = torch.linspace(0, largersize - 224, num_cuts)
    for cut in cut_spots:
        if smaller_side == "sideX":
            cutout = image_tensor[:, :, int(cut):int(cut) + 224, :]
        else:
            cutout = image_tensor[:, :, :, int(cut):int(cut) + 224]
        cutouts.append(F.interpolate(cutout, size=(cut_size, cut_size), mode='bicubic', align_corners=False, antialias=True))
        del cutout

    cutouts = torch.cat(cutouts, dim=0)
    return cutouts


class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
        vision_model_path=None,
        initializer_token="*",
        custom_prompts=False,
        pad_tokens=True,
        subtokens=None,
        vae_path = None,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.initializer_token = initializer_token
        self.vision_model_path = vision_model_path
        self.pad_tokens = pad_tokens

        self.subtokens = subtokens
        if subtokens is not None:
            self.num_vectors = len(subtokens)
        else:
            self.num_vectors = 1

        #TODO need better way to recognize picture extension
        self.image_paths = [os.path.join("./traindata", file_path) for file_path in os.listdir("./traindata") if file_path.endswith(".jpeg") or file_path.endswith(".png") or file_path.endswith(".jpg")]
        if custom_prompts:
            # check if there are text files to pull from
            txt_file_exists = False
            all_files = os.listdir(self.data_root)
            for file_path in all_files:
                if file_path.endswith(".txt"):
                    txt_file_exists = True
                    break
            if txt_file_exists:
                captions = []
                for file_path in os.listdir(self.data_root):
                        if file_path.endswith(".txt"):
                            with open(os.path.join(self.data_root, file_path), 'r') as f:
                                captions.append(f.readline())
                self.captions = captions
            else:
                self.captions = [file_path.split(".")[0] for file_path in os.listdir(self.data_root) if file_path.endswith(".jpeg") or file_path.endswith(".png") or file_path.endswith(".jpg")]

        else:
            self.caption = None

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small if learnable_property == "object" else blank_template
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

        # get embeddings for first wave
        if vision_model_path is not None:
            image_model = CLIPVisionModelWithProjection.from_pretrained(vision_model_path).to("cuda").to(torch.float16).eval()

            with torch.no_grad():
                self.clip_embeddings = []
                for path in self.image_paths:
                    image = Image.open(path).convert("RGB")
                    image = self.flip_transform(image)
                    image = np.array(image).astype(np.float32) / 255.0
                    image = image[None].transpose(0, 3, 1, 2)
                    image = torch.from_numpy(image).to("cuda").to(torch.float16)
                    cuts = sliding_cutouts(image, num_cuts=7, cut_size=224)
                    embeds = image_model(cuts).image_embeds.mean(dim=0)
                    self.clip_embeddings.append(embeds)

                self.initializer_token_id = self.tokenizer(initializer_token).input_ids[1] # ignore start and end token

        if vae_path is not None:
            with torch.no_grad():
                #TODO set revision
                vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae", revision="fp16").to("cuda").to(torch.float16).eval()
                self.latents = []
                for path in self.image_paths:
                    image = Image.open(path).convert("RGB")
                    img = np.array(image).astype(np.uint8)

                    if self.center_crop:
                        crop = min(img.shape[0], img.shape[1])
                        (
                            h,
                            w,
                        ) = (
                            img.shape[0],
                            img.shape[1],
                        )
                        img = img[(h - crop) // 2: (h + crop) // 2, (w - crop) // 2: (w + crop) // 2]

                    image = Image.fromarray(img)
                    image = image.resize((self.size, self.size), resample=self.interpolation)

                    image = self.flip_transform(image)
                    image = np.array(image).astype(np.uint8)
                    image = (image / 127.5 - 1.0).astype(np.float32)
                    image = torch.from_numpy(image).permute(2, 0, 1).to(torch.float16).to("cuda")
                    latents = vae.encode(image).latent_dist.sample()
                    latents = latents * 0.18125
                    self.latents.append(latents)
        else:
            self.latents = None


    def __len__(self):
        return self._length

    def __getitem__(self, i):
        if self.vision_model_path is not None:
            example = {}

            if self.captions is None:
                if self.num_vectors > 1:
                    assert isinstance(self.subtokens, list)
                    text = random.choice(self.templates)
                    begin, end = text.replace("{", "").split("}")
                    newstr = ""
                    for tok in self.subtokens:
                        newstr = newstr + tok + " "
                    text = newstr + end
                else:
                    text = random.choice(self.templates).format(self.placeholder_token)
            else:
                text = self.captions[i % self.num_images]

            example["input_ids"] = self.tokenizer(
                text,
                padding="max_length" if self.pad_tokens else "do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]

            if self.what_to_train != "layer":
                example["token_position"] = torch.where(example["input_ids"] == self.initializer_token_id)[0][0].item()

            example["clip_embeds"] = self.clip_embeddings[i % self.num_images]
            return example

        else:
            example = {}

            if self.latents == None:
                text = random.choice(self.templates).format(self.placeholder_token)

                example["input_ids"] = self.tokenizer(
                    text,
                    padding="max_length" if self.pad_tokens else "do_not_pad",
                    truncation=True,
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids[0]

                image = Image.open(self.image_paths[i % self.num_images])

                if not image.mode == "RGB":
                    image = image.convert("RGB")

                # default to score-sde preprocessing
                img = np.array(image).astype(np.uint8)

                if self.center_crop:
                    crop = min(img.shape[0], img.shape[1])
                    (
                        h,
                        w,
                    ) = (
                        img.shape[0],
                        img.shape[1],
                    )
                    img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

                image = Image.fromarray(img)
                image = image.resize((self.size, self.size), resample=self.interpolation)

                image = self.flip_transform(image)
                image = np.array(image).astype(np.uint8)
                image = (image / 127.5 - 1.0).astype(np.float32)
                example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
            else:
                example["latents"] = self.latents[i % self.num_images]

            return example


def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
    )

    if args.report_to == "wandb":
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.clip_model_path)

    # Load scheduler and models
    text_encoder = CLIPTextModelWithProjection.from_pretrained(args.clip_model_path).to(accelerator.device)

    with torch.no_grad():
        # Add the placeholder token in tokenizer
        if args.num_vectors == 1:
            num_added_tokens = tokenizer.add_tokens(args.placeholder_token)
            subtokens = None
        else:
            num_added_tokens = 0
            subtokens = [f'xyzzyx{i}' for i in range(args.num_vectors)]
            for tok in subtokens:
                num_added_tokens = tokenizer.add_tokens(tok)

        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )

        if args.initializer_token is not None:
            # Convert the initializer_token, placeholder_token to ids
            token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids) > 1:
                raise ValueError("The initializer token must be a single token.")

            initializer_token_id = token_ids.input_ids

        if args.num_vectors > 1:
            initializer_token_id = [initializer_token_id] * args.num_vectors

            placeholder_token_id = tokenizer.convert_tokens_to_ids(subtokens)
        else:
            placeholder_token_id = tokenizer.convert_tokens_to_ids(args.placeholder_token)

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder.resize_token_embeddings(len(tokenizer))

        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = text_encoder.get_input_embeddings().weight.data
        if args.initializer_token is not None:
            if args.num_vectors > 1:
                for i in range(args.num_vectors):
                    token_embeds[placeholder_token_id[i]] = token_embeds[initializer_token_id[i]]
            else:
                token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]
        else:
            if args.num_vectors > 1:
                for i in range(args.num_vectors):
                    token_embeds[placeholder_token_id[i]] = torch.randn_like(token_embeds[0]) * 0.01
            else:
                token_embeds[placeholder_token_id] = torch.randn_like(token_embeds[0]) * 0.01

        # Freeze all parameters except for the token embeddings in text encoder
        text_encoder.text_model.encoder.requires_grad_(False)
        text_encoder.text_model.final_layer_norm.requires_grad_(False)
        text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
        text_encoder.get_input_embeddings().weight.requires_grad_(True)

        if args.gradient_checkpointing:
            # Keep unet in train mode if we are using gradient checkpointing to save memory.
            # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
            text_encoder.gradient_checkpointing_enable()


    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = TextualInversionDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        placeholder_token=args.placeholder_token,
        repeats=args.repeats,
        learnable_property=args.learnable_property,
        center_crop=args.center_crop,
        vision_model_path=args.vision_model_pretrained,
        initializer_token=args.initializer_token,
        pad_tokens=args.pad_tokens,
        subtokens=subtokens,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.clip_train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )


    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("textual_inversion", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # keep original embeddings as reference
    orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()
    # assume 49407 max
    index_no_updates = torch.arange(len(tokenizer)) <= 49407

    pbar = tqdm(range(args.clip_train_steps))
    for epoch in pbar:
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step

            # Get the text embedding for conditioning
            cond_tok_ids = batch["input_ids"].to(accelerator.device)
            im_embed = batch["clip_embeds"].to(accelerator.device).to(dtype=torch.float32).detach()
            text_embeds = text_encoder(cond_tok_ids).text_embeds.to(dtype=torch.float32)

            if args.spherical_clip_loss == True:
                loss = spherical_dist_loss(text_embeds, im_embed)
            else:
                text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
                im_embed = im_embed / im_embed.norm(p=2, dim=-1, keepdim=True)
                sim = -torch.matmul(text_embeds, im_embed.t())
                #only similarity with correct pairings needed
                mask = torch.diag(torch.ones(sim.shape[0])).float()
                sim = sim * mask
                loss = sim.mean()

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                if args.clip_max_grad_norm is not None:
                    accelerator.clip_grad_norm_(text_encoder.get_input_embeddings().parameters())

            optimizer.step()
            #lr_scheduler.step()
            optimizer.zero_grad()

            pbar.set_description(f"Epoch {epoch}, loss: {loss.mean().detach().item():.4f}")

            # Let's make sure we don't update any embedding weights besides the newly added token
            with torch.no_grad():
                accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[index_no_updates] = orig_embeds_params[index_no_updates]



    # Dataset and DataLoaders creation:
    train_dataset = TextualInversionDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        placeholder_token=args.placeholder_token,
        repeats=args.repeats,
        learnable_property=args.learnable_property,
        center_crop=args.center_crop,
        vision_model_path=None,
        initializer_token=args.initializer_token,
        pad_tokens=args.pad_tokens,
        vae_path=args.pretrained_model_name_or_path if args.cache_latents else None,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)


    # For mixed precision training we cast the unet and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16


    # remake the optimizer
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    if args.cache_latents == False:
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
        # Freeze vae and unet
        vae.requires_grad_(False)
        vae.to(accelerator.device, dtype=weight_dtype)

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    if args.gradient_checkpointing:
        # Keep unet in train mode if we are using gradient checkpointing to save memory.
        # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
        unet.train()
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()


    # Move vae and unet to device and cast to weight_dtype
    unet.requires_grad_(False)
    unet.to(accelerator.device, dtype=weight_dtype)


    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # second wave
    for epoch in range(first_epoch, args.num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(text_encoder):
                if args.cache_latents == False:
                    # Convert images to latent space
                    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                    latents = latents * 0.18125
                else:
                    latents = batch["latent"].to(dtype=weight_dtype).detach()

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"]).last_hidden_state.to(dtype=weight_dtype)

                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    if args.max_grad_norm is not None:
                        accelerator.clip_grad_norm_(
                            text_encoder.get_input_embeddings().parameters(),
                            args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Let's make sure we don't update any embedding weights besides the newly added token
                with torch.no_grad():
                    accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                        index_no_updates
                    ] = orig_embeds_params[index_no_updates]


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"learned_embeds-steps-{global_step}.bin")
                    save_progress(text_encoder, placeholder_token_id, accelerator, args, save_path, subtokens)

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.mean().detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
            logger.info(
                f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                f" {args.validation_prompt}."
            )
            # create pipeline (note: unet and vae are loaded again in float32)
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=accelerator.unwrap_model(text_encoder),
                tokenizer=tokenizer,
                revision=args.revision,
                torch_dtype=weight_dtype,
            )
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
            pipeline = pipeline.to(accelerator.device)
            pipeline.set_progress_bar_config(disable=True)

            # run inference
            generator = (
                None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
            )
            images = []
            for _ in range(args.num_validation_images):
                with torch.autocast("cuda"):
                    image = pipeline(args.validation_prompt, num_inference_steps=25, generator=generator).images[0]
                images.append(image)

            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                if tracker.name == "wandb":
                    tracker.log(
                        {
                            "validation": [
                                wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                for i, image in enumerate(images)
                            ]
                        }
                    )

            del pipeline
            torch.cuda.empty_cache()

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "learned_embeds.bin")
        save_progress(text_encoder, placeholder_token_id, accelerator, args, save_path, subtokens)

    accelerator.end_training()