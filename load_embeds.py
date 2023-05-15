
import os
import torch
import torch.utils.checkpoint



def load_embeds(pipe, embeddings_dir, SD_version="v1"):
    paths = os.listdir(embeddings_dir)
    paths = [p for p in paths if p.endswith(".pt")] #TODO depends on what we use

    #TODO for now store the placeholder token in filename, issue with illegal characters
    num_added_tokens = 0
    loaded_tokens = []
    for path in paths:
        embed_dict = torch.load(os.path.join(embeddings_dir,path))

        if 'SD_version' in embed_dict.keys():
            embed_version = embed_dict["SD_version"]
        else:
            # lazy fix
            embed_version = SD_version

        if SD_version == embed_version:
            num_added_tokens += pipe.tokenizer.add_tokens(embed_dict["placeholder_token"])
            placeholder_token_id = pipe.tokenizer.convert_tokens_to_ids(embed_dict["placeholder_token"])

            # Resize the token embeddings as we are adding new special tokens to the tokenizer
            pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))

            # load the embed
            token_embeds = pipe.text_encoder.get_input_embeddings().weight.data
            token_embeds[placeholder_token_id] = embed_dict["embedding"]

            # running total of tokens to know what got loaded
            loaded_tokens.append(embed_dict["placeholder_token"])
        else:
            print("token:" + embed_dict["placeholder_token"] + "not compatible with current SD version, skipping...")

    print("trained tokens added:", loaded_tokens)

