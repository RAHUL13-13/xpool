from config.base_config import Config
import numpy as np
import torch
from collections import defaultdict, deque
from trainer.base_trainer import BaseTrainer
from modules.metrics import sim_matrix_training, sim_matrix_inference, generate_embeds_per_video_id, \
    sim_matrix_training_modified, sim_matrix_inference_modified, generate_embeds_per_video_id_modified
from tqdm import tqdm


class Trainer(BaseTrainer):
    """
    Trainer class
    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, metrics, optimizer, config: Config, train_data_loader, 
                 valid_data_loader, tokenizer, lr_scheduler=None, writer=None):

        super().__init__(model, loss, metrics, optimizer, config, writer)
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.tokenizer1 = tokenizer[0]
        self.tokenizer2 = tokenizer[1]

        self.pooling_type = config.pooling_type
        self.window_metric = defaultdict(lambda: deque(maxlen=config.eval_window_size))
        self.best_window = -1.0
        self.best = -1.0


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        """
        self.model.train()
        total_loss = 0.0
        num_steps = len(self.train_data_loader)
        eval_steps = np.linspace(0, num_steps-1, self.evals_per_epoch+1, dtype=int)[1:]
        
        for batch_idx, data in enumerate(self.train_data_loader):
            # then assume we must tokenize the input, e.g. its a string
            if self.tokenizer1 is not None:
                # clip
                data['text_sing'] = self.tokenizer1(data['text'], return_tensors='pt', padding=True, truncation=True)
                
                # Bert
                # data['text_sing'] = self.tokenizer2(data['text'], pad_to_max_length=True, add_special_tokens=False, truncation=True, max_length=12, return_tensors="pt")
                data['text_seq'] =  self.tokenizer2(data['text'], pad_to_max_length=True, add_special_tokens=False, truncation=True, max_length=12, return_tensors="pt")
            if isinstance(data['text_sing'], torch.Tensor):
                data['text_sing'] = data['text_sing'].to(self.device)
                data['text_seq'] = data['text_seq'].to(self.device)
            else:
                data['text_sing'] = {key: val.to(self.device) for key, val in data['text_sing'].items()}
                data['text_seq'] = {key: val.to(self.device) for key, val in data['text_seq'].items()}
                
            data['video'] = data['video'].to(self.device)
            
            text_embeds_pooled, video_embeds_pooled, text_embeds, video_embeds, text_features_sequential, video_features_non_seq  = self.model(data)
            # print("IN TRAIN before loss calc, coming from model t tseq tP vP", text_embeds.shape, text_features_sequential.shape, text_embeds_pooled.shape, video_embeds_pooled.shape)
            
            # loss no.1 (between diag of conditioned t and conditioned v)
            # text_embeds_pooled_diag = torch.diagonal(text_embeds_pooled)
            # text_embeds_pooled_diag = text_embeds_pooled_diag.permute(1,0)
            # output1 = sim_matrix_training(text_embeds_pooled_diag, video_embeds_pooled, self.pooling_type)
            # L1 = self.loss(output1, self.model.clip.logit_scale)
            
            # loss no. 2 (between t and conditioned v)
            # output_ori = sim_matrix_training(text_embeds, video_embeds_pooled, self.pooling_type)
            # L2 = self.loss(output_ori, self.model.clip.logit_scale)
            
            # loss no. 3 (between conditioned t and diag of conditioned v)
            # video_embeds_pooled_diag = torch.diagonal(video_embeds_pooled)
            # video_embeds_pooled_diag = video_embeds_pooled_diag.permute(1,0)
            # output3 = sim_matrix_training(video_embeds_pooled_diag, text_embeds_pooled, self.pooling_type)
            # L3 = self.loss(output3, self.model.clip.logit_scale)
            
            # loss no. 4 (between conditioned t and conditioned v)
            output4 = sim_matrix_training_modified(video_embeds_pooled, text_embeds_pooled, self.pooling_type)
            L4 = self.loss(output4, self.model.clip.logit_scale)
            
            loss = L4#+L1+L3
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
            torch.clamp_(self.model.clip.logit_scale.data, max=np.log(100))
            
            self.global_step += 1
            if self.writer is not None:
                self.writer.add_scalar('train/loss_train', loss.detach().item(), self.global_step)

            total_loss += loss.detach().item()

            if batch_idx % self.log_step == 0:
                print('Train Epoch: {} dl: {}/{} Loss: {:.6f}'.format(
                    epoch,
                    batch_idx,
                    num_steps-1,
                    loss.detach().item()), end= "   ")

            if batch_idx in eval_steps:
                val_res = self._valid_epoch_step(epoch, batch_idx, num_steps-1)
                self.model.train()

                if val_res['R1-window'] > self.best_window:
                    self.best_window = val_res['R1-window']
                    self._save_checkpoint(epoch, save_best=True)

                if val_res['R1'] > self.best:
                    self.best = val_res['R1']

                print(" Current Best Window Average R@1 is {}".format(self.best_window))
                print(" Current Best R@1 is {}\n\n".format(self.best))

        res = {
            'loss_train':  total_loss / num_steps
        }

        return res

    
    def _valid_epoch_step(self, epoch, step, num_steps):
        """
        Validate at a step when training an epoch at a certain step
        :return: A log that contains information about validation
        """
        self.linear_layer = torch.nn.Linear(768, 512)
        self.model.eval()
        total_val_loss = 0.0
        text_embed_arr = []
        vid_embed_arr = []
        text_embed_seq_arr = [] # correct only
        vid_embed_non_seq_arr = []
        all_vid_ids = []
        
        with torch.no_grad():
            for _, data in tqdm(enumerate(self.valid_data_loader)):
                if self.tokenizer1 is not None:
                    # clip
                    data['text_sing'] = self.tokenizer1(data['text'], return_tensors='pt', padding=True, truncation=True)
                    
                    # Bert
                    # data['text_sing'] = self.tokenizer2(data['text'], pad_to_max_length=True, add_special_tokens=False, truncation=True, max_length=12, return_tensors="pt")
                    data['text_seq'] =  self.tokenizer2(data['text'], pad_to_max_length=True, add_special_tokens=False, truncation=True, max_length=12, return_tensors="pt")
                if isinstance(data['text_sing'], torch.Tensor):
                    data['text_sing'] = data['text_sing'].to(self.device)
                    data['text_seq'] = data['text_seq'].to(self.device)
                else:
                    data['text_sing'] = {key: val.to(self.device) for key, val in data['text_sing'].items()}
                    data['text_seq'] = {key: val.to(self.device) for key, val in data['text_seq'].items()}
                    
                data['video'] = data['video'].to(self.device)
                
                text_embed_pooled, vid_embed_pooled, text_embed, vid_embed, text_feature_sequential, video_features_non_seq = self.model(data, return_all_frames=True)
                # print("IN val before loss calc, coming from model t tseq tP vP", text_embed.shape, text_feature_sequential.shape, text_embed_pooled.shape, vid_embed_pooled.shape)
                
                text_embed_arr.append(text_embed.cpu())
                vid_embed_arr.append(vid_embed.cpu())
                
                text_embed_seq_arr.append(text_feature_sequential.cpu())
                vid_embed_non_seq_arr.append(video_features_non_seq.cpu())
                
                # loss 1
                # text_embed_pooled_diag = torch.diagonal(text_embed_pooled)
                # text_embed_pooled_diag = text_embed_pooled_diag.permute(1,0)
                # output1 = sim_matrix_training(text_embed_pooled_diag, vid_embed_pooled, self.pooling_type)
                # L1 = self.loss(output1, self.model.clip.logit_scale)
                
                # loss 2
                # output_ori = sim_matrix_training(text_embed, vid_embed_pooled, self.pooling_type)
                # L2 = self.loss(output_ori, self.model.clip.logit_scale)   
                
                # loss 3
                # video_embeds_pooled_diag = torch.diagonal(vid_embed_pooled)
                # video_embeds_pooled_diag = video_embeds_pooled_diag.permute(1,0)
                # output3 = sim_matrix_training(video_embeds_pooled_diag, text_embed_pooled, self.pooling_type)
                # L3 = self.loss(output3, self.model.clip.logit_scale)
                
                # loss 4
                output4 = sim_matrix_training_modified(vid_embed_pooled, text_embed_pooled, self.pooling_type)
                L4 = self.loss(output4, self.model.clip.logit_scale)
                
                total_val_loss += L4.item() #+ L1.item() + L3.item()
                
                for v_id in data['video_id']:
                    all_vid_ids.append(v_id)
                
            text_embeds = torch.cat(text_embed_arr)
            vid_embeds = torch.cat(vid_embed_arr)
            text_embeds_seq = torch.cat(text_embed_seq_arr)
            video_features_non_seq = torch.cat(vid_embed_non_seq_arr)
            
            # text_embed_pooled, vid_embeds_pooled = self.model.pools(text_embeds, vid_embeds)
            # print(text_embeds.shape, vid_embeds.shape, video_features_non_seq.shape, text_embeds_seq.shape)
            
            # self.model.pools.cpu()
            # correct_pooled_text = torch.empty((0, 512), dtype=torch.float32)
            # for ith in range(text_embeds.shape[0]):
                # _, text_embed_pooled = self.model.pools(torch.unsqueeze(text_embeds[ith], 0), torch.unsqueeze(vid_embeds[ith], 0), torch.unsqueeze(video_features_non_seq[ith], 0), torch.unsqueeze(text_embeds_seq[ith], 0))
                # one_correct_pooled_text = torch.diagonal(text_embed_pooled)
                # one_correct_pooled_text = one_correct_pooled_text.permute(1,0)
                # correct_pooled_text = torch.cat((correct_pooled_text, one_correct_pooled_text), dim=0)
            # print(correct_pooled_text.shape)
            # self.model.pools.cuda()
            
            # Since we have all pairs, remove duplicate videos when there's multiple captions per video
            vid_embeds_per_video_id = {}
            vid_embed_non_seq_per_video_id = {}
            for idx, v_id in enumerate(all_vid_ids):
                if v_id not in vid_embeds_per_video_id:
                    vid_embeds_per_video_id[v_id] = vid_embeds[idx]
                    vid_embed_non_seq_per_video_id[v_id] = video_features_non_seq[idx]
            
            vid_embeds = torch.stack([vid_embeds_per_video_id[v_id] for v_id in vid_embeds_per_video_id])
            video_features_non_seq = torch.stack([vid_embed_non_seq_per_video_id[v_id] for v_id in vid_embed_non_seq_per_video_id])
            
            # print("IN VAL, before pool t v", text_embeds.shape, vid_embeds.shape) 
            # torch.Size([65, 512]) torch.Size([2, 12, 512])
            
            # Pool frames for inference once we have all texts and videos
            self.model.pools.cpu()
            # text_embed_pooled, vid_embeds_pooled = self.model.pools(text_embeds, vid_embeds)
            vid_embeds_pooled, text_embed_pooled = self.model.pools(text_embeds, vid_embeds, video_features_non_seq, text_embeds_seq)
            self.model.pools.to(torch.device("cuda:7"))
            
            # print(vid_embeds_pooled.shape, text_embed_pooled.shape, text_embeds.shape, vid_embeds.shape)
            # print(all_vid_ids)
            text_embeds_pooled_per_video_id, vid_embeds_pooled_per_video_id = generate_embeds_per_video_id_modified(text_embed_pooled, 
                    vid_embeds_pooled, all_vid_ids, self.pooling_type)
            
            sims = sim_matrix_inference_modified(text_embeds_pooled_per_video_id, vid_embeds_pooled_per_video_id, self.pooling_type)
            # print(sims.shape)
            
            total_val_loss = total_val_loss / len(self.valid_data_loader)
            
            metrics = self.metrics
            res = metrics(sims)
            
            # Compute window metrics
            for m in res:
                self.window_metric[m].append(res[m])
                
            # Compute average of window metrics
            for m in self.window_metric:
                res[m + "-window"] = np.mean(self.window_metric[m])
                
            print(f"-----Val Epoch: {epoch}, dl: {step}/{num_steps}-----\n",
                  f"R@1: {res['R1']} (window: {res['R1-window']})\n", 
                  f"R@5: {res['R5']} (window: {res['R5-window']})\n", 
                  f"R@10: {res['R10']} (window: {res['R10-window']})\n",
                  f"MedR: {res['MedR']} (window: {res['MedR-window']})\n",
                  f"MeanR: {res['MeanR']} (window: {res['MeanR-window']})\n",
                  f"Loss: {total_val_loss}")
            
            res['loss_val'] =  total_val_loss
            
            if self.writer is not None:
                for m in res:
                    self.writer.add_scalar(f'val/{m}', res[m], self.global_step)
                    
            return res
