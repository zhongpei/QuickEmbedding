from types import SimpleNamespace
train_args = dict({
"save_steps": 500,
"clip_model_path": "openai/clip-vit-large-patch14",
"pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
"revision": None,
"tokenizer_name": None,
"train_data_dir": "./traindata",
"placeholder_token": "sksts",
"initializer_token": "cat",
"learnable_property": "object",
"repeats": 100,
"output_dir": "text-inversion-model",
"seed": None,
"resolution": 512,
"center_crop": True,
"clip_train_batch_size": 64,
"train_batch_size": 4,
"num_train_epochs": 100,
"clip_train_steps": 20,
"clip_train_lr": 8.0e-04,
"max_train_steps": 140,
"gradient_accumulation_steps": 1,
"gradient_checkpointing": True,
"learning_rate": 7.0e-04,
"scale_lr": False,
"lr_scheduler": "constant",
"lr_warmup_steps": 0,
"dataloader_num_workers": 4, # set to 0 if this causes issues
"adam_beta1": 0.9,
"adam_beta2": 0.999,
"adam_weight_decay": 1e-2,
"adam_epsilon": 1e-08,
"logging_dir": "logs",
"mixed_precision": "bf16",
"allow_tf32": True,
"report_to": "tensorboard",
"validation_prompt": None,
"num_validation_images": 4,
"validation_epochs": 50,
"local_rank": -1,
"checkpointing_steps": 5000,
"resume_from_checkpoint": None,
"enable_xformers_memory_efficient_attention": True,
"spherical_clip_loss": False,
"vision_model_pretrained": "openai/clip-vit-large-patch14",
"zero_out_mismatches":True,
"what_to_train": "layer",
"max_grad_norm": 1,
"clip_max_grad_norm": 1,
})

train_args = SimpleNamespace(**train_args)






