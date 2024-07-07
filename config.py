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


def get_model_config(num_key:int=2) -> dict:
    config = {
        '1': {
            "epochs": 25,
            "train_batch_size": 32,
            "valid_batch_size": 32,
            "sequence_len": 250,
            "embed_dim": 384, # 456, 480, 512, 516, 576
            "num_heads": 8,
            "num_kv_heads": None,
            "num_decoders": 6,
            "ff_dim": 541, # 400, 433, 451, 512
            "dropout_prob": 0.2,
            "activation_name": 'gelu',
            "norm_placing": 'post',
            "learning_rate": 1e-2
        },
        '2': {
            "epochs": 25,
            "train_batch_size": 32,
            "valid_batch_size": 32,
            "sequence_len": 300,
            "embed_dim": 480, # 456, 480, 512, 516, 576
            "num_heads": 12,
            "num_kv_heads": None,
            "num_decoders": 8,
            "ff_dim": 512, # 400, 433, 451, 512
            "dropout_prob": 0.2,
            "activation_name": 'gelu',
            "norm_placing": 'post',
            "learning_rate": 0.01,
            "scale_lr": 0.1,
            "lr_scheduler": True
        },
        '3': {
            "epochs": 100,
            "train_batch_size": 32,
            "valid_batch_size": 32,
            "sequence_len": 300,
            "embed_dim": 576, # 456, 480, 512, 516, 576
            "num_heads": 12,
            "num_kv_heads": None,
            "num_decoders": 12,
            "ff_dim": 512, # 400, 433, 451, 512
            "dropout_prob": 0.2,
            "activation_name": 'gelu',
            "norm_placing": 'post',
            "learning_rate": 0.1,
            "scale_lr": 0.1,
            "lr_scheduler": True
        }
    }    
    return config[str(num_key)]

def get_general_config(num_sent_corpus:str='101000') -> dict:
    
    return {
        "data_src": {
            "dirname": "data",
            "corpus_file_name": f"sp_fin_all_small_space_word_tokenized_{num_sent_corpus}_sentences.txt",
            'sp_model_file': f"sp_tokenizer_{num_sent_corpus}_sentences_model_space_word.model"
        },
        "model": {
            "final": {
                "dirname": "export", 
                "state_file_basename": f"fin_model_state_dict_final_{num_sent_corpus}",
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