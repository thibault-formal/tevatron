import torch
import copy 
import os
import logging
from .encoder import EncoderModel

logger = logging.getLogger(__name__)

class DenseModelFrozen(EncoderModel):
    def __init__(self,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.encoder_q = copy.deepcopy(self.encoder)
        self.freeze(self.encoder)

    def freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False
            
    def encode_query(self, qry):
        query_hidden_states = self.encoder_q(**qry, return_dict=True)
        query_hidden_states = query_hidden_states.last_hidden_state
        return self._pooling(query_hidden_states, qry['attention_mask'])
    
    def encode_passage(self, psg):
        doc_hidden_states = self.encoder(**psg, return_dict=True)
        doc_hidden_states = doc_hidden_states.last_hidden_state
        return self._pooling(doc_hidden_states, psg['attention_mask'])
        

    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling in ['cls', 'first']:
            reps = last_hidden_state[:, 0]
        elif self.pooling in ['mean', 'avg', 'average']:
            masked_hiddens = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            reps = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling in ['last', 'eos']:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        else:
            raise ValueError(f'unknown pooling method: {self.pooling}')
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps
    
    @classmethod
    def load(cls,
             model_name_or_path: str,
             pooling: str = 'cls',
             normalize: bool = False,
             lora_name_or_path: str = None,
             **hf_kwargs):
        base_model = cls.TRANSFORMER_CLS.from_pretrained(os.path.join(model_name_or_path, "doc"), device_map='auto', **hf_kwargs)
        model = cls(
                encoder=base_model,
                pooling=pooling,
                normalize=normalize
            )
        # then load the query model  
        model.encoder_q = cls.TRANSFORMER_CLS.from_pretrained(os.path.join(model_name_or_path, "query"))
        return model

    def save(self, output_dir: str):
        self.encoder.save_pretrained(os.path.join(output_dir, "doc"))  # not really needed but easier 
        self.encoder_q.save_pretrained(os.path.join(output_dir, "query"))