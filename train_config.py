from types import SimpleNamespace
train_args = dict({
"save_steps": 500,
"clip_model_path": "openai/clip-vit-large-patch14",

"pretrained_model_name_or_path": "E:\\sdwebui\\stable-diffusion-webui\\QuickEmbedding\\models\\runway\\",
"revision": None,
"tokenizer_name": None,
"train_data_dir": "E:\\sdwebui\\stable-diffusion-webui\\QuickEmbedding\\traindata",
"placeholder_token": "zhubao",
"initializer_token": "jewelry",
"learnable_property": "object", # style object

"repeats": 100,
"output_dir": "text-inversion-model",
"seed": None,
"resolution": 768,
"center_crop": True,

"clip_train_batch_size": 32,
"clip_max_train_steps": 400,
"clip_train_lr": 1.0e-02,
"clip_phase_gradient_checkpointing": False,
"clip_lr_scheduler": "cosine_with_restarts", # linear cosine_with_restarts constant_with_warmup constant
"spherical_clip_loss": False,
"clip_max_grad_norm": 1.5,
"max_train_steps": 100,
"gradient_accumulation_steps": 1,
"gradient_checkpointing": True,
"train_batch_size": 4,
"learning_rate": 7.0e-04, # "learning_rate": 1.5e-04(constant) 7.0e-04(cosine_with_restarts)
"scale_lr": False,
"lr_scheduler": "cosine_with_restarts", # constant cosine_with_restarts constant_with_warmup
"lr_num_cycles": 1,
"optimizer": "adamw", # adamw dadaptadam
"lr_warmup_steps": 0,
"dataloader_num_workers": 0, # set to 0 if this causes issues
"adam_beta1": 0.9,
"adam_beta2": 0.999,
"adam_weight_decay": 1e-2,
"adam_epsilon": 1e-08,
"logging_dir": "logs",
"mixed_precision": "fp16",# fp16 bf16
"allow_tf32": False,
"report_to": "wandb", # tensorboard wandb
"validation_prompt": None,
"num_validation_images": 4,
"validation_epochs": 50,
"local_rank": -1,
"checkpointing_steps": 5000,
"resume_from_checkpoint": None,
"enable_xformers_memory_efficient_attention": True,
"vision_model_pretrained": "openai/clip-vit-large-patch14",
"zero_out_mismatches":True,
"what_to_train": "layer",
"max_grad_norm": 1,
"num_vectors": 1,
"pad_tokens": True,
"cache_latents": True,
"best_loss":0.02
})

train_args = SimpleNamespace(**train_args)






