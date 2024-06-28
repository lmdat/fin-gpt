import torch
from torch.utils.data import Dataset, DataLoader
import os, json
from tqdm import tqdm
from typing import Iterable, Any
import random
from tokenizer import VnTokenizer

class FinCorpusDataset(Dataset):
    def __init__(self, 
                 raw_data:list, 
                 sp_tokenizer:VnTokenizer, 
                 sequence_len:int) -> None:
        """
        """
        super().__init__()

        self.dataset = raw_data
        # self.vocab = vocab
        self.sp_tokenizer = sp_tokenizer
        self.seq_len = sequence_len

        self.sos_token = torch.tensor([sp_tokenizer.sos_token_id], dtype=torch.int64)
        self.eos_token = torch.tensor([sp_tokenizer.eos_token_id], dtype=torch.int64)
        self.pad_token = torch.tensor([sp_tokenizer.pad_token_id], dtype=torch.int64)        

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        sentence = self.dataset[index]       
       
        token_ids = self.sp_tokenizer.encode(sentence, False, False)
        
        decoder_num_padding_tokens = self.seq_len - (len(token_ids) + 1)
        if decoder_num_padding_tokens < 0:
            raise ValueError(f"Sentence is too long. Max sequence length is: {self.seq_len}")
        
        padding_token_tensor = torch.tensor([self.pad_token] * decoder_num_padding_tokens, dtype=torch.int64)
        
        # Add only <SOS> token
        decoder_input = torch.cat(
            [
                torch.tensor(self.sp_tokenizer.encode(sentence, True, False), dtype=torch.int64),
                padding_token_tensor
            ],
            dim=0,
        )
        
        # Add only <EOS> token
        label = torch.cat(
            [
                torch.tensor(self.sp_tokenizer.encode(sentence, False, True), dtype=torch.int64),
                padding_token_tensor
            ],
            dim=0,
        )

        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # (1, seq_len)
        decoder_mask_zero_padding = (decoder_input != self.pad_token).unsqueeze(0).int() # is pad_token: 0; otherwise: 1 --> [1,1,1,1,0,0,0]
        # (1, seq_len, seq_len)
        causal_mask = generate_causal_mask(self.seq_len) 
        
        return {
            'decoder_input': decoder_input, # (seq_len)
            'decoder_mask': decoder_mask_zero_padding & causal_mask, # (1, seq_len) & (1, seq_len, seq_len) = (1, seq_len, seq_len)
            'label': label # (seq_len)
            # 'sentence_input': sentence
        }
    

def generate_causal_mask(seq_len:int) -> torch.Tensor:
    """
    Mask is the lower triangular part of matrix (includes the diagonal) filled with True
    """    
    mask = torch.tril(torch.ones(1, seq_len, seq_len)) # (1, seq_len, seq_len)
    return mask.bool() # 1: True; 0: False


def _load_word_tokenized_sentences(input_file:str, max_sequence_len:int, max_buffer_size:int=32) -> tuple[list, int]:
    """
    Load word tokenized from file and produce the unique word list.

    Parameters:
        input_file: word tokenized sentences file.
        max_sequence_len: max sequence len
        max_buffer_size: the buffer size can read per batch.

    Return:
        tokens: the unique list of word tokens
        max_word_count: the max sequence len
    """
    if os.path.exists(input_file) == False:
        print(f"Cannot find this file: {input_file}")
        return None, 0
    
    size_hint = -1
    if max_buffer_size != None:
        size_hint = max_buffer_size * 1024 * 1024

    k = 1
    sentences = []
    max_word_count = 0
    print(f"{'='*10}Load sentences dataset{'='*10}")
    with open(input_file, mode='r', encoding='utf-8') as f:
        while True:
            lines = f.readlines(size_hint)
            if not lines:
                break

            for line in tqdm(lines, desc=f"Batch {k}"):
                sent = line.strip()
                word_count = sent.count(" ") + 1
                if sent == '' or word_count > max_sequence_len:
                    continue
                
                # sent_dict = json.loads(line.strip())                                
                max_word_count = max(max_word_count, word_count) 
                sentences.append(sent)
            k += 1
        f.close()
    return sentences, max_word_count

def _create_vocab_with_index(word_tokens:Iterable=None, 
                             special_tokens:list=None, 
                             vocab_file:str=None, 
                             forced:bool=False) -> dict:
    """
    Create the vocab json file.

    Parameters:
        word_tokens: the list of unique word tokens
        special_tokens: if None: the default are: <SOS>, <EOS>, <PAD>, <UNK>
        vocab_file: vacab file output
        forced: if True: Load and create new vocab file

    Return:
        vocab: list of vocab
    """
    if forced == False or word_tokens == None:
        with open(vocab_file, mode='r', encoding='utf-8') as f:
            vocab = json.loads(f.read().strip())
            f.close()
            print(f"\nLoad vocab from file: {vocab_file}")
        return vocab

    tokens = {}
    start_idx = 0
    if special_tokens == None:
        tokens = {
            "<SOS>": 0,
            "<EOS>": 1,
            "<PAD>": 2,
            "<UNK>": 3
        }
        start_idx = 4
    else:
        for i, token in enumerate(special_tokens):
            tokens[token] = i
        start_idx = len(special_tokens)
        
    for i, token in enumerate(tqdm(word_tokens, desc="Vocab"), start_idx):
        tokens[token] = i
    
    vocab = {
        'size': len(tokens),
        'tokens': tokens
    }
        
    with open(vocab_file, mode='w', encoding='utf-8') as f:
        str_json = json.dumps(vocab, indent=2, ensure_ascii=False).encode('utf-8')
        f.write(str_json.decode())
        f.close()
    print(f"Create and save vocab to file: {vocab_file}")
    return vocab


def _split_dataset(data_src:list, 
                   test_size:float=None, 
                   shuffle:bool=True, 
                   random_state:int=None) -> tuple[list, list]:
    """
    Split the whole dataset into train and test(or validation) data aparts
    """
    data = data_src.copy()

    if shuffle == True:
        if random_state != None:
            random.seed(random_state)
        random.shuffle(data)

    if test_size == None:
        test_size = 0.2
    
    n = len(data)
    train_size = 1 - test_size
    
    train_data = data[ : int(n * train_size)]
    test_data = data[int(n * train_size) : ]
    
    return train_data, test_data   


def prepair_corpus(config:dict,
                   corpus_file:str,
                   tmp_dir:str="tmp",
                   test_size:int=0.2,
                   forced_create_file:bool=False) -> tuple[str, str]:
    """
    """
    train_file = f"{tmp_dir}/train_raw_data.txt"
    valid_file = f"{tmp_dir}/valid_raw_data.txt"

    if forced_create_file == True or not (os.path.exists(train_file) and os.path.exists(valid_file)):
        sentences_data, max_seq_len = _load_word_tokenized_sentences(corpus_file, max_sequence_len=config['sequence_len'])
        
        train_raw_data, valid_raw_data = _split_dataset(sentences_data, test_size=test_size)

        if os.path.dirname(train_file) != "":
            os.makedirs(os.path.dirname(train_file), exist_ok=True)
        
        with open(train_file, mode='w', encoding='utf-8') as f_train:
            str_lines = []
            for item in train_raw_data:
                # json_line = json.dumps(item, ensure_ascii=False).encode('utf-8')
                # str_lines.append(f"{json_line.decode()}\n")
                str_lines.append(f"{item}\n")

            f_train.writelines(str_lines)
            f_train.close()
            print(f"Train file: {train_file} is created.")

    
        with open(valid_file, mode='w', encoding='utf-8') as f_valid:
            str_lines = []
            for item in valid_raw_data:
                # json_line = json.dumps(item, ensure_ascii=False).encode('utf-8')
                # str_lines.append(f"{json_line.decode()}\n")
                str_lines.append(f"{item}\n")
            
            f_valid.writelines(str_lines)
            f_valid.close()
            print(f"Valid file: {valid_file} is created.")   
  
    return train_file, valid_file



def load_dataloader_chunks(config:dict,
                           sp_tokenizer:VnTokenizer,
                           data_file:str,
                           batch_size:int,
                           is_shuffle:bool=True,
                           max_buffer_size:float=16):
    """
    """
    if not os.path.exists(data_file):
        raise ValueError(f"File: {data_file} does not exist.")
    
    chunk_size = -1
    if max_buffer_size != None and max_buffer_size > 0:
        chunk_size = int(max_buffer_size * 1024 * 1024)

    
    with open(data_file, mode='r', encoding='utf-8') as f:
        while True:
            lines = f.readlines(chunk_size)
            if not lines:
                break
            
            chunk_raw_data = []
            for item in lines:
                if item.strip() == '':
                    continue
                # sent = json.loads(item.strip())
                chunk_raw_data.append(item)

            fin_data = FinCorpusDataset(
                raw_data=chunk_raw_data,
                sp_tokenizer=sp_tokenizer,
                sequence_len=config['sequence_len']
            )
            train_dataloader = DataLoader(fin_data, batch_size=batch_size, shuffle=is_shuffle)
            yield train_dataloader
        f.close()


def load_dataset(config:dict,
                 corpus_file:str,
                 vocab_file:str,
                 forced_create_file:bool=False) -> tuple[DataLoader, DataLoader, dict, int]:
    """
    """
       
    
    sentences_data, tokens, max_seq_len = _load_word_tokenized_sentences(corpus_file, max_sequence_len=config['sequence_len'])

    if forced_create_file == True or vocab_file == None or os.path.exists(vocab_file) == False:
        vocab = _create_vocab_with_index(word_tokens=tokens, vocab_file=vocab_file, forced=True)
    else:
        vocab = _create_vocab_with_index(vocab_file=vocab_file)

    train_raw_data, valid_raw_data = _split_dataset(sentences_data, test_size=0.2)

    train_data = FinCorpusDataset(
        raw_data=train_raw_data,
        vocab=vocab,
        sequence_len=config['sequence_len']
    )

    valid_data = FinCorpusDataset(
        raw_data=valid_raw_data,
        vocab=vocab,
        sequence_len=config['sequence_len']
    )

    train_dataloader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=True)

    print(f"{'='*10}Dataset{'='*10}")
    print(f"\t.Raw data size: {len(sentences_data)} sentences.")
    print(f"\t.Raw train data size: {len(train_raw_data)} sentences.")
    print(f"\t.Raw valid data size: {len(valid_raw_data)} sentences.")
    print(f"\t.Max length of sentence: {max_seq_len}")
    print(f"\t.Vocab size: {vocab['size']} words")

    return train_dataloader, valid_dataloader, vocab, max_seq_len