import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.base_config import Config

class MultiHeadedAttention(nn.Module):
    def __init__(self, config: Config):
        super(MultiHeadedAttention, self).__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_mha_heads
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    
    def forward(self, text_embeds, video_embeds):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            o: num_vids x num_texts x embed_dim
        """
        # print("pooling text, vid", text_embeds.shape, video_embeds.shape)
        num_texts, _ = text_embeds.shape
        # num_texts x embed_dim
        q = self.q_proj(text_embeds)
        q = q.reshape(num_texts, self.num_heads, self.head_dim)
        # num_heads x head_dim x num_texts
        q = q.permute(1,2,0)

        num_vids, num_frames, _ = video_embeds.shape
        # num_vids x num_frames x embed_dim
        k = self.k_proj(video_embeds)
        k = k.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        # num_vids x num_heads x num_frames x head_dim
        k = k.permute(0,2,1,3)

        # num_vids x num_frames x embed_dim
        v = self.v_proj(video_embeds)
        v = v.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        # num_vids x num_heads x head_dim x num_frames
        v = v.permute(0,2,3,1)
        
        # print(k.shape, q.shape)
        # num_vids x num_heads x num_frames x num_texts
        attention_logits = k @ q
        attention_logits = attention_logits / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_logits, dim=2)

        # num_vids x num_heads x head_dim x num_texts
        attention = v @ attention_weights
        # num_vids x num_texts x num_heads x head_dim
        attention = attention.permute(0,3,1,2)
        attention = attention.reshape(num_vids, num_texts, self.embed_dim)

        # num_vids x num_texts x embed_dim
        o = self.out_proj(attention)
        return o

class MultiHeadedAttention_(nn.Module):
    def __init__(self, config: Config):
        super(MultiHeadedAttention_, self).__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_mha_heads
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    
    def forward(self, video_features_averaged, text_features_sequential):
        """
        Input
            video_features_averaged: num_vids x embed_dim
            text_features_sequential: num_texts x num_tokens x embed_dim
        Output
            o: num_vids x num_texts x embed_dim
        """
        
        num_texts, _ = video_features_averaged.shape
        q = self.q_proj(video_features_averaged)
        q = q.reshape(num_texts, self.num_heads, self.head_dim)
        
        q = q.permute(1,2,0)

        num_vids, num_frames, _ = text_features_sequential.shape
        k = self.k_proj(text_features_sequential)
        k = k.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        
        k = k.permute(0,2,1,3)


        v = self.v_proj(text_features_sequential)
        v = v.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        
        v = v.permute(0,2,3,1)
        
        
        attention_logits = k @ q
        attention_logits = attention_logits / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_logits, dim=2)


        attention = v @ attention_weights
        
        attention = attention.permute(0,3,1,2)
        attention = attention.reshape(num_vids, num_texts, self.embed_dim)

        # num_texts x num_vids embed_dim
        o = self.out_proj(attention)
        return o




class Transformer(nn.Module):
    def __init__(self, config: Config):
        super(Transformer, self).__init__()
        self.embed_dim = config.embed_dim
        dropout = config.transformer_dropout

        self.cross_attn = MultiHeadedAttention(config)
        self.cross_attn_ = MultiHeadedAttention_(config)

        self.linear_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear_proj_ = nn.Linear(self.embed_dim, self.embed_dim)
            
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.layer_norm3 = nn.LayerNorm(self.embed_dim)
        self.layer_norm4 = nn.LayerNorm(self.embed_dim)
        self.layer_norm5 = nn.LayerNorm(self.embed_dim)
        self.layer_norm6 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_parameters()

    
    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)


    def forward(self, text_embeds, video_embeds, video_features_averaged, text_features_sequential):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            out: num_vids x num_texts x embed_dim
        """
        text_embeds = self.layer_norm1(text_embeds)
        video_embeds = self.layer_norm1(video_embeds)

        # num_vids x num_texts x embed_dim
        attn_out = self.cross_attn(text_embeds, video_embeds)
        attn_out = self.layer_norm2(attn_out)

        linear_out = self.linear_proj(attn_out)
        out = attn_out + self.dropout(linear_out)
        out = self.layer_norm3(out)



        video_features_averaged = self.layer_norm4(video_features_averaged)
        text_features_sequential = self.layer_norm4(text_features_sequential)

        # num_vids x num_texts x embed_dim
        attn_out_ = self.cross_attn_(video_features_averaged, text_features_sequential)
        attn_out_ = self.layer_norm5(attn_out_)

        linear_out_ = self.linear_proj_(attn_out_)
        out_ = attn_out_ + self.dropout(linear_out_)
        out_ = self.layer_norm6(out_)

        return out, out_
