from dataclasses import dataclass, fields
from typing import Optional

@dataclass
class ModelParams:
    vocab_size:int = 0

    train_batch_size:int = 32

    valid_batch_size:int = 32

    sequence_len:int = 260

    embed_dim:int = 456

    num_heads:int = 8 # aka. num heads of Query

    num_kv_heads:Optional[int] = None # num heads of Key & Value

    num_decoders:int = 6

    ff_dim:int = 400

    dropout_prob:float = 0.2
    activation_name:str = 'gelu'
    norm_placing:str = 'post'
    learning_rate:float = 1e-2


def get_model_config() -> dict:
    return {
        "epochs": 25,
        "train_batch_size": 32,
        "valid_batch_size": 32,
        "sequence_len": 260,
        "embed_dim": 456,
        "num_heads": 8,
        "num_kv_heads": None,
        "num_decoders": 6,
        "ff_dim": 400,
        "dropout_prob": 0.2,
        "activation_name": 'gelu',
        "norm_placing": 'post',
        "learning_rate": 1e-2
    }

def get_general_config() -> dict:
    return {
        "data_src": {
            "dirname": "data",
            "corpus_file_name": "sp_fin_all_small_word_tokenized_sentences.txt",
            'sp_model_file': "sp_tokenizer_model_word.model"
        },
        "model": {
            "final": {
                "dirname": "export", 
                "state_file_basename": "fin_model_state_dict_final",
            },
            "checkpoint":{
                "dirname": "checkpoint", 
                "state_file_basename_epoch": "fin_model_state_dict_epoch"
            },
            "tuning": {
                "checkpoint_basename": "raytune_checkpoint.pt"
            }
        }
    }

def get_model_params(conf:dict=None) -> ModelParams:
    if conf == None:
        return ModelParams()
    
    filtered_data = {f.name: conf[f.name] for f in fields(ModelParams)}
    return ModelParams(**filtered_data)