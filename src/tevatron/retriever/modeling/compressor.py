
import torch

from transformers import AutoModel, AutoTokenizer
import math 

class Compressor(torch.nn.Module):
    def __init__(self, compr_model_name, compr_rate, compr_linear_type, decoder_hidden_size):
        super().__init__()
        # init model
        self.model_name = compr_model_name
        self.model = AutoModel.from_pretrained(compr_model_name, torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(compr_model_name, use_fast=True)
        self.compr_rate = compr_rate
        self.compressing_mode = compr_linear_type

        if self.compressing_mode == 'concat':
            self.linear = torch.nn.Linear(self.model.config.hidden_size*self.compr_rate, decoder_hidden_size)
        elif self.compressing_mode in ['cls', 'mean', 'sep']:
            self.linear = torch.nn.Linear(self.model.config.hidden_size, decoder_hidden_size)
        self.linear = self.linear.bfloat16()

    def forward(self, input_ids, attention_mask):
        segment_compress_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        num_embs = math.ceil(input_ids.size(1) / self.compr_rate)

        all_hidden_states_emb = list()
        if self.compressing_mode == 'concat':
            for segment_idx in range(num_embs):
                start_idx = segment_idx * self.compr_rate
                end_idx = (segment_idx + 1) * self.compr_rate
                hidden_state = segment_compress_outputs.hidden_states[-1][:, start_idx:end_idx, :]
                hidden_state_concat = torch.flatten(hidden_state, start_dim=1) #batch_size, hidden_state_dim * compression_rate
                all_hidden_states_emb.append(hidden_state_concat)


        elif self.compressing_mode == "mean":
            for segment_idx in range(num_embs):
                start_idx = segment_idx * self.compr_rate
                end_idx = (segment_idx + 1) * self.compr_rate
                hidden_state = segment_compress_outputs.hidden_states[-1][:, start_idx:end_idx, :]
                # Apply mean pooling to get the final embedding for the segment
                all_hidden_states_emb.append(hidden_state)


        all_hidden_states_emb_cat = torch.stack(all_hidden_states_emb, dim=1)
        transformed_embeds = self.linear(all_hidden_states_emb_cat)
        

        if self.compressing_mode == "mean":
            transformed_embeds = torch.mean(transformed_embeds, dim=2)


        return  transformed_embeds