import model
import config as cfg
import os, datetime
from tokenizer import VnTokenizer
import torch

ROOT_PATH = os.getcwd()

def train_model():
    m_config = cfg.get_model_config()
    g_config = cfg.get_general_config()

    # vocab_file = f"{data_dir}/{g_config['data_src']['vocab_file_name']}"
    # word_tokenized_file = f"{data_dir}/viwiki_word_tokenized_sentences.txt"
    # vocab_file = f"{data_dir}/viwiki_vocab.json"

    data_dirname = g_config['data_src']['dirname']
    checkpoint_dirname = g_config['model']['checkpoint']['dirname']
    
    corpus_file = f"{ROOT_PATH}/{data_dirname}/{g_config['data_src']['corpus_file_name']}"
    sp_model_file = f"{ROOT_PATH}/{data_dirname}/{g_config['data_src']['sp_model_file']}"
    tmp_dir = f"{ROOT_PATH}/tmp"
    checkpoint_dir = f"{ROOT_PATH}/{checkpoint_dirname}"


    gpt_model = model.train(m_config,
                            g_config,
                            corpus_file=corpus_file,                            
                            sp_model_file=sp_model_file,
                            tmp_dir=tmp_dir,
                            checkpoint_dir=checkpoint_dir,
                            epochs=m_config['epochs'])
    print("Training phase has finished.")
      
    cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    export_dirname = g_config['model']['final']['dirname']
    file_name = f"{ROOT_PATH}/{export_dirname}/{g_config['model']['final']['state_file_basename']}_{cur_time}.pt"
    model.save_model_state_dict(gpt_model, file_name)
    print(f"Model has been saved as: {file_name}")


def text_generation(prompts:list[str]):
    m_config = cfg.get_model_config()
    g_config = cfg.get_general_config()
    
    data_dir = g_config['data_src']['dirname']
    sp_model_file = f"{data_dir}/{g_config['data_src']['sp_model_file']}"

    sp_tokenizer = VnTokenizer(sp_model_file)
    m_config['vocab_size'] = sp_tokenizer.vocab_size

    model_file = _latest_file()
    gpt_model = model.load_model_state_dict(m_config, model_file)
    model.generate(gpt_model, sp_tokenizer, m_config, prompts=prompts, top_type='k', top_k=20, temperature=0.9)


def _latest_file():
    files = []
    config = cfg.get_general_config()
    dirname = config['model']['final']['dirname']
    for file_name in os.listdir(dirname):
        if config['model']['final']['state_file_basename'] in file_name:
            files.append(f"{dirname}/{file_name}")    
    
    if len(files) == 0:
        return None
    
    return max(files, key=os.path.getctime)
         


if __name__ == "__main__":
    
    train_model() 
    
    # prompts = [
    #     'Ngân hàng SCB và các khoản vay của bà Trương Mỹ Lan',
    #     'Chỉ số chứng khoán VNINDEX năm 2023',
    #     # 'Dự án sân bay biên hòa',
    #     # 'Dự án căn hộ chung cư tại TPHCM năm 2023'
    # ]
    # text_generation(prompts)