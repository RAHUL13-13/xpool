import torch
import torch.nn as nn
from config.base_config import Config
from modules.transformer import Transformer
# import jax.numpy as jnp
import numpy as np
    
class CLIPTransformer(nn.Module):
    def __init__(self, config: Config):
        super(CLIPTransformer, self).__init__()
        self.config = config
        
        if self.config.huggingface:
            from transformers import CLIPModel, FlaxCLIPTextModel, BertModel
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.model = BertModel.from_pretrained('bert-base-uncased')
            # self.model = FlaxCLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        else:
            from model.clip_model import load_clip
            self.clip = load_clip(config.clip_arch)
            
        config.pooling_type = 'transformer'
        self.pools = Transformer(config)
        self.linear_layer = torch.nn.Linear(768, 512)
        self.linear_layer2 = torch.nn.Linear(768, 512)
        
    def forward(self, data, return_all_frames=False):
        batch_size = data['video'].shape[0]
        text_data = data['text_sing']
        text_data_sequential = data['text_seq']
        video_data = data['video']
        # print("IN CLIPtransforer, before encoding t v", text_data, video_data)
        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)
        
        if self.config.huggingface:
            # clip
            text_features = self.clip.get_text_features(**text_data)
            
            # Bert
            # text_features = self.model(**text_data).last_hidden_state[:, 0, :]
            video_features = self.clip.get_image_features(video_data)
        else:
            text_features = self.clip.encode_text(text_data)
            video_features = self.clip.encode_image(video_data)
        
        
        # text_features_sequential = self.clip(**text_data_sequential)["text"]
        
        # BERT
        text_features_sequential = self.model(**text_data_sequential).last_hidden_state
        text_features_sequential = self.linear_layer(text_features_sequential)
        
        # FLAX
        # text_data_sequential['input_ids'] = (text_data_sequential['input_ids'].cpu())
        # text_data_sequential['input_ids'] = (text_data_sequential['input_ids']).numpy()
        
        # text_data_sequential['attention_mask'] = (text_data_sequential['attention_mask'].cpu())
        # text_data_sequential['attention_mask'] = (text_data_sequential['attention_mask']).numpy()
        
        # text_features_sequential = self.model(input_ids = jnp.array(text_data_sequential['input_ids']), attention_mask=jnp.array(text_data_sequential['attention_mask'])).last_hidden_state
        # text_features_sequential = np.array(text_features_sequential)
        # text_features_sequential = torch.from_numpy(text_features_sequential).cuda()
        
        
        
        # print(text_features.shape, video_features.shape, text_features_sequential.shape)
        # torch.Size([BS, 512]) torch.Size([348, 512]) torch.Size([BS, 14, 768]
        
        video_features = video_features.reshape(batch_size, self.config.num_frames, -1)
        video_features_averaged = video_features.mean(dim=1)
        # print("IN CLIPtransforer, before POOL t vs v ts", text_features.shape, video_features.shape, video_features_averaged.shape, text_features_sequential.shape)
        
        # print(self.pools.layer_norm1.weight.device, text_features.device, video_features.device, video_features_averaged.device, text_features_sequential.device)
        video_features_pooled, text_features_pooled = self.pools(text_features, video_features, video_features_averaged, text_features_sequential)
        
        # print("IN CLIPtransforer, after POOL t v", video_features_pooled.shape, text_features_pooled.shape)
        
        if return_all_frames:
            return text_features_pooled, video_features_pooled, text_features, video_features, text_features_sequential, video_features_averaged
        # print(text_features_pooled.shape, video_features_pooled.shape)
        return text_features_pooled, video_features_pooled, text_features, video_features, text_features_sequential, video_features_averaged
