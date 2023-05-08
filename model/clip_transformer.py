import torch
import torch.nn as nn
from config.base_config import Config
from modules.transformer import Transformer, TransformerEncoderWithCLS
# import jax.numpy as jnp
import numpy as np
    
class CLIPTransformer(nn.Module):
    def __init__(self, config: Config):
        super(CLIPTransformer, self).__init__()
        self.config = config
        
        if self.config.huggingface:
            from transformers import CLIPModel, FlaxCLIPTextModel, BertModel
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.bert_model = BertModel.from_pretrained('bert-base-uncased')
            # self.model = FlaxCLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        else:
            from model.clip_model import load_clip
            self.clip = load_clip(config.clip_arch)
            
        config.pooling_type = 'transformer'
        self.pools = Transformer(config)
        self.linear_layer = torch.nn.Linear(768, config.embed_dim)
        self.linear_layer2 = torch.nn.Linear(768, config.embed_dim)
        # self.linear_layer3 = torch.nn.Linear(config.num_frames*config.embed_dim, config.embed_dim)
        
        self.TransformerEncoderWithCLS = TransformerEncoderWithCLS(config)
        
    def forward(self, data, return_all_frames=False):
        batch_size = data['video'].shape[0]
        text_data = data['text_sing']
        text_data_sequential = data['text_seq']
        video_data = data['video']
        
        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)
        
        if self.config.huggingface:
            # CLIP non-seq
            text_features = self.clip.get_text_features(**text_data)
            
            # sequential video output but its flattened 
            # (batch_size*num frames) x embed size
            video_features = self.clip.get_image_features(video_data)
        else:
            text_features = self.clip.encode_text(text_data)
            video_features = self.clip.encode_image(video_data)     
        
        # BERT for Sequential Text
        # batch_size x num_tokens x 768_embed_size
        text_features_sequential = self.bert_model(**text_data_sequential).last_hidden_state
        # batch_size x num_tokens x 512_embed_size
        text_features_sequential = self.linear_layer(text_features_sequential)     
        
        # When using CLIP, for sequential video embedding, unflatten video_features
        # batch_size x num frames x embed size
        video_features = video_features.reshape(batch_size, self.config.num_frames, self.config.embed_dim) 
        
        # Get Non-Sequential Video by averaging in token dimension
        # batch_size x embed size
        # video_features_non_seq = video_features.mean(dim=1)
        
        # Using transformer encoder for Non-Sequential Video
        video_features_non_seq = self.TransformerEncoderWithCLS(video_features)
        
        # print("IN CLIPtransformer, before POOL t vs v ts", text_features.shape, video_features.shape, video_features_non_seq.shape, text_features_sequential.shape)
        
        # (num_video x num_text x embed_size) & (num_text x num_video x embed_size)
        video_features_pooled, text_features_pooled = self.pools(text_features, video_features, video_features_non_seq, text_features_sequential)
        # print("IN CLIPtransformer, after poolT  poolV", video_features_pooled.shape, text_features_pooled.shape)
        
        return text_features_pooled, video_features_pooled, text_features, video_features, text_features_sequential, video_features_non_seq


# When replacing CLIP's text encoder with BERT for Non-sequential Text embedding
# text_features = self.model(**text_data).last_hidden_state[:, 0, :]
# text_features = self.linear_layer(text_features)

# FlaxCLIP seq
# text_data_sequential['input_ids'] = (text_data_sequential['input_ids'].cpu())
# text_data_sequential['input_ids'] = (text_data_sequential['input_ids']).numpy()
# text_data_sequential['attention_mask'] = (text_data_sequential['attention_mask'].cpu())
# text_data_sequential['attention_mask'] = (text_data_sequential['attention_mask']).numpy()      
# text_features_sequential = self.model(input_ids = jnp.array(text_data_sequential['input_ids']), attention_mask=jnp.array(text_data_sequential['attention_mask'])).last_hidden_state
# text_features_sequential = np.array(text_features_sequential)
# text_features_sequential = torch.from_numpy(text_features_sequential).cuda()   