import model
import config as cfg
import os, datetime
from tokenizer import VnTokenizer


ROOT_PATH = os.getcwd()

def text_generation(prompts:list[str]):
    m_config = cfg.get_model_config()
    g_config = cfg.get_general_config()
    
    data_dir = g_config['data_src']['dirname']
    sp_model_file = f"{ROOT_PATH}/{data_dir}/{g_config['data_src']['sp_model_file']}"

    sp_tokenizer = VnTokenizer(sp_model_file)
    m_config['vocab_size'] = sp_tokenizer.vocab_size

    model_file = _latest_file()
    gpt_model = model.load_model_state_dict(m_config, model_file)
    model.generate(gpt_model, sp_tokenizer, m_config, prompts=prompts, top_type='p', top_p=0.7, temperature=1.4)


def _latest_file():
    files = []
    config = cfg.get_general_config()
    dirname = config['model']['final']['dirname']
    for file_name in os.listdir(os.path.join(ROOT_PATH, dirname)):
        if config['model']['final']['state_file_basename'] in file_name:
            files.append(f"{ROOT_PATH}/{dirname}/{file_name}")    
    
    if len(files) == 0:
        return None
    
    files.sort()

    # return max(files, key=os.path.getctime)
    return files[-1]


if __name__ == "__main__":

    prompts = [
        'Ngân hàng SCB và các khoản vay của bà Trương Mỹ Lan',
        # 'Chỉ số chứng khoán VNINDEX năm 2023',
        # 'Dự án sân bay biên hòa',
        #'Dự án căn hộ chung cư tại TPHCM năm 2023'
        
    ]
    text_generation(prompts)