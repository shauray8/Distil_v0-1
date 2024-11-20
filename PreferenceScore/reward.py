## https://github.com/THUDM/ImageReward/blob/main/ImageReward/ImageReward.py
import os
import torch
import torch.nn as nn
from PIL import Image
from .bert import BertConfig, BertModel
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

import warnings
warnings.filterwarnings("ignore")

import torch
import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file
from transformers import BertTokenizer

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


def create_vit(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):

    assert vit in ['base', 'large'], "vit parameter must be base or large"
    if vit=='base':
        vision_width = 768
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12,
                                           num_heads=12, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0 or drop_path_rate
                                          )
    elif vit=='large':
        vision_width = 1024
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=24,
                                           num_heads=16, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0.1 or drop_path_rate
                                          )
    return visual_encoder, vision_width


def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")

def load_checkpoint(model,url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu')
    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        raise RuntimeError('checkpoint url or path is invalid')

    state_dict = checkpoint['model']

    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                                         model.visual_encoder_m)
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape!=model.state_dict()[key].shape:
                print(key, ": ", state_dict[key].shape, ', ', model.state_dict()[key].shape)
                del state_dict[key]

    msg = model.load_state_dict(state_dict,strict=False)
    print('load checkpoint from %s'%url_or_filename)
    return model,msg

class BLIP_Pretrain(nn.Module):
    def __init__(self,
                 med_config = "med_config.json",
                 image_size = 224,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 embed_dim = 256,
                 queue_size = 57600,
                 momentum = 0.995,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer, 0)

        self.tokenizer = init_tokenizer()
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False)

        text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            #nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # initial MLP param
        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0/(self.input_size+1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)
        
    def forward(self, input):
        return self.layers(input)


class ImageReward(nn.Module):
    def __init__(self, med_config, device='cpu'):
        super().__init__()
        self.device = device
        
        self.blip = BLIP_Pretrain(image_size=224, vit='large', med_config=med_config)
        self.preprocess = _transform(224)
        self.mlp = MLP(768)
        
        self.mean = 0.16717362830052426
        self.std = 1.0333394966054072


    def score_gard(self, prompt_ids, prompt_attention_mask, image):

        image_embeds = self.blip.visual_encoder(image)
        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(self.device)
        text_output = self.blip.text_encoder(prompt_ids,
                                                    attention_mask = prompt_attention_mask,
                                                    encoder_hidden_states = image_embeds,
                                                    encoder_attention_mask = image_atts,
                                                    return_dict = True,
                                                )
        
        txt_features = text_output.last_hidden_state[:,0,:] # (feature_dim)
        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std
        
        return rewards


    def score(self, prompt, image):
        
        if (type(image).__name__=='list'):
            _, rewards = self.inference_rank(prompt, image)
            return rewards
            
        # text encode
        text_input = self.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(self.device)
        
        # image encode
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, str) and os.path.isfile(image):
            pil_image = Image.open(image)
        else:
            raise TypeError(r'This image parameter type has not been supportted yet. Please pass PIL.Image or file path str.')
            
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        image_embeds = self.blip.visual_encoder(image)
        
        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(self.device)
        text_output = self.blip.text_encoder(text_input.input_ids,
                                                attention_mask = text_input.attention_mask,
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,
                                                return_dict = True,
                                            )
        
        txt_features = text_output.last_hidden_state[:,0,:].float() # (feature_dim)
        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std
        
        return rewards.detach().cpu().numpy().item()


    def inference_rank(self, prompt, generations_list):
        
        text_input = self.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(self.device)
        
        txt_set = []
        for generation in generations_list:
            # image encode
            if isinstance(generation, Image.Image):
                pil_image = generation
            elif isinstance(generation, str):
                if os.path.isfile(generation):
                    pil_image = Image.open(generation)
            else:
                raise TypeError(r'This image parameter type has not been supportted yet. Please pass PIL.Image or file path str.')
            image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            image_embeds = self.blip.visual_encoder(image)
            
            # text encode cross attention with image
            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(self.device)
            text_output = self.blip.text_encoder(text_input.input_ids,
                                                    attention_mask = text_input.attention_mask,
                                                    encoder_hidden_states = image_embeds,
                                                    encoder_attention_mask = image_atts,
                                                    return_dict = True,
                                                )
            txt_set.append(text_output.last_hidden_state[:,0,:])
            
        txt_features = torch.cat(txt_set, 0).float() # [image_num, feature_dim]
        rewards = self.mlp(txt_features) # [image_num, 1]
        rewards = (rewards - self.mean) / self.std
        rewards = torch.squeeze(rewards)
        _, rank = torch.sort(rewards, dim=0, descending=True)
        _, indices = torch.sort(rank, dim=0)
        indices = indices + 1
        
        return indices.detach().cpu().numpy().tolist(), rewards.detach().cpu().numpy().tolist()
