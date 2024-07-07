import model
import config as cfg
import os, datetime
from tokenizer import VnTokenizer
import torch

ROOT_PATH = os.getcwd()

def train_model(num_sent_corpus:str=None):
    if num_sent_corpus == None:
        num_sent_corpus = '101000'

    m_config = cfg.get_model_config()
    g_config = cfg.get_general_config(num_sent_corpus)

    # vocab_file = f"{data_dir}/{g_config['data_src']['vocab_file_name']}"
    # word_tokenized_file = f"{data_dir}/viwiki_word_tokenized_sentences.txt"
    # vocab_file = f"{data_dir}/viwiki_vocab.json"

    data_dirname = g_config['data_src']['dirname']
    checkpoint_dirname = g_config['model']['checkpoint']['dirname']
    
    corpus_file = f"{ROOT_PATH}/{data_dirname}/{g_config['data_src']['corpus_file_name']}"
    sp_model_file = f"{ROOT_PATH}/{data_dirname}/{g_config['data_src']['sp_model_file']}"
    tmp_dir = f"{ROOT_PATH}/tmp/{num_sent_corpus}"
    checkpoint_dir = f"{ROOT_PATH}/{checkpoint_dirname}"


    gpt_model = model.train(m_config,
                            g_config,
                            corpus_file=corpus_file,                            
                            sp_model_file=sp_model_file,
                            tmp_dir=tmp_dir,
                            checkpoint_dir=checkpoint_dir,
                            epochs=m_config['epochs'],
                            keep_checkpoint_file_num=2)
    
    print("Training phase has finished.")
      
    cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    export_dirname = g_config['model']['final']['dirname']
    file_name = f"{ROOT_PATH}/{export_dirname}/{g_config['model']['final']['state_file_basename']}_{cur_time}.pt"
    model.save_model_state_dict(gpt_model, file_name)
    print(f"Model has been saved as: {file_name}")


if __name__ == "__main__":    
    train_model('151000')