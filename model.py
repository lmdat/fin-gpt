import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import cores
import dataset
from tqdm import tqdm
import numpy as np
import datetime, time, os, json, sys
from pathlib import Path
from tokenizer import VnTokenizer

from ray import train as ray_train
import tempfile
from typing import Union

from config import ModelParams, get_model_params


def build_model(conf:dict) -> cores.TransformerDecoder:
    params = get_model_params(conf)
    return cores.TransformerDecoder(params)

def save_checkpoint(payload:dict, output_file:str) -> str:
    if os.path.dirname(output_file) != '':
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    torch.save(payload, output_file)
    return output_file

def save_model_state_dict(model:cores.TransformerDecoder, output_file:str) -> str:
    if os.path.dirname(output_file) != '':
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    torch.save(model.state_dict(), output_file)
    return output_file


def load_model_state_dict(config:dict, model_file:str) -> cores.TransformerDecoder:
    model = build_model(config)
    payload = torch.load(model_file)
    if 'model_state_dict' in payload:
        model.load_state_dict(payload['model_state_dict'])
    else:
        model.load_state_dict(payload)
    return model

def stop_or_continue(timeout:int=10):
    try:
        len_fill = len(str(timeout))        
        while timeout >= 0:
            print(f"\rThe training will continue. Press Ctrl+C to stop within {str(timeout).rjust(len_fill, '0')} seconds.", end="\r")
            time.sleep(1)
            timeout -= 1
        print("")
    except KeyboardInterrupt:
        sys.exit(1)

def get_epoch_loss_from_checkpoint(file_list:list):
    loss_list = []
    for file_name in file_list:
        payload = torch.load(file_name)
        epoch = payload['epoch'] + 1
        train_loss = payload['train_loss']
        valid_loss = payload['valid_loss']
        loss_list.append((epoch, train_loss, valid_loss))
    return loss_list

def show_epoch_loss(loss_list:list):
    print("\nEpoch | Train loss | Valid loss")
    for item in loss_list:
        print(f"{item[0]:02d} | {item[1]:10.5f} | {item[2]:10.5f}")
    print('')
    
def train(m_config:dict,
          g_config:dict,
          corpus_file:str,
          sp_model_file:str,
          tmp_dir:str="tmp",
          checkpoint_dir:str='checkpoint',
          epochs:int=20,
          tuning_mode:bool=False,
          keep_checkpoint_file_num:int=None) -> Union[object, None]:

    # train_dataloader, valid_dataloader, vocab, max_seq_len = dataset.load_dataset(config, 
    #                                                                               word_tokenized_file=corpus_file, 
    #                                                                               vocab_file=vocab_file)
    
    # print('Loader Size: ', len(train_dataloader))
    # batch = next(iter(train_dataloader))
    # print(len(batch['sentence']))


    sp_tokenizer = VnTokenizer(sp_model_file)
    m_config['vocab_size'] = sp_tokenizer.vocab_size
    
    print(f"\n{'='*10}Config{'='*10}")
    print(json.dumps(m_config, indent=2))

    train_file, valid_file = dataset.prepair_corpus(m_config, corpus_file, tmp_dir, test_size=0.2)

    model = build_model(m_config)
    loss_func = nn.CrossEntropyLoss(ignore_index=sp_tokenizer.pad_token_id)
    optimizer = optim.Adam(params=model.parameters(), lr=m_config['learning_rate'])
    
    epoch_loss_list = []
    
    init_epoch = 0
    if tuning_mode == True:
        last_checkpoint = ray_train.get_checkpoint()
        if last_checkpoint:
            with last_checkpoint.as_directory() as chkp_dir:
                payload = torch.load(os.path.join(chkp_dir, g_config['model']['tuning']['checkpoint_basename']))
                init_epoch = payload["epoch"] + 1
                model.load_state_dict(payload["model_state_dict"])
                optimizer.load_state_dict(payload["optimizer_state_dict"])
    else:
        checkpoint_files = []
        for file_name in os.listdir(checkpoint_dir):
            if g_config['model']['checkpoint']['state_file_basename_epoch'] in file_name:
                checkpoint_files.append(f"{checkpoint_dir}/{file_name}")  

        total_files = len(checkpoint_files)
        if total_files > 0:
            checkpoint_files.sort()
            epoch_loss_list = get_epoch_loss_from_checkpoint(checkpoint_files)
            payload = torch.load(checkpoint_files[-1])
            init_epoch = payload["epoch"] + 1
            model.load_state_dict(payload["model_state_dict"])
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            print(f"\n{'='*10}Checkpoint{'='*10}")
            print(f".::.Load last checkpoint from file: {checkpoint_files[-1]}")
            print(f".::.Epoch {init_epoch}: Train cost: {payload['train_loss']} | Valid cost: {payload['valid_loss']}")

            if keep_checkpoint_file_num != None and total_files > keep_checkpoint_file_num:
                delele_file_num = total_files - keep_checkpoint_file_num
                for file in checkpoint_files[:delele_file_num]:
                    if os.path.exists(file):
                        os.remove(file)


    print(f"\n{'='*10}Start Training{'='*10}")
    print(f"Epoch start at: {init_epoch + 1}")
    for epoch in range(init_epoch, epochs):
        
        if len(epoch_loss_list) > 0:
            show_epoch_loss(epoch_loss_list)

        k = epoch + 1
        # Switch to training mode
        model.train()
        train_cost = 0
        train_cost_count = 0
        chunk = 1
        for train_dataloader in dataset.load_dataloader_chunks(m_config,
                                                               sp_tokenizer,
                                                               train_file,
                                                               m_config['train_batch_size'],
                                                               max_buffer_size=0.5):
            
            print(f".::.Train chunk {chunk} size: {train_dataloader.dataset.__len__()} sentences | Batch size: {train_dataloader.batch_size} | Num Batches: {train_dataloader.__len__()}")
            
            tqdm_iter = tqdm(train_dataloader, desc=f"Epoch {k:02d}/{epochs}|Step {chunk}")
            for batch in tqdm_iter:
                decoder_input = batch['decoder_input']
                decoder_mask = batch['decoder_mask']
                targets = batch['label'] # (batch_size, seq_len)
                                                            
                # Forward pass
                pred_logits = model(decoder_input, decoder_mask) # (batch_size, seq_len, vocab_size)
            
                # pred_logits.view(-1, vocab['size']) same as pred_logits.flatten(0, 1): (batch_size, seq_len, vocab_size) --> (batch_size * seq_len, vocab_size)
                # target.view(-1) same as target.flatten(): (batch_size, seq_len) --> (batch_size * seq_len)
                train_loss = loss_func(pred_logits.view(-1, m_config['vocab_size']), targets.view(-1))
                tqdm_iter.set_postfix({"Train loss": f"{train_loss.item():10.4f}"})
                train_cost += train_loss.item()
                train_cost_count += 1

                # Backprop
                optimizer.zero_grad()
                train_loss.backward()

                # Update parameters
                optimizer.step()
            chunk += 1
            
               
        # Switch to validation mode
        model.eval()
        valid_cost = 0
        valid_cost_count = 0
        chunk = 1
        with torch.inference_mode():
            for valid_dataloader in dataset.load_dataloader_chunks(m_config,
                                                                   sp_tokenizer,
                                                                   valid_file,
                                                                   m_config['valid_batch_size'],
                                                                   max_buffer_size=0.5):
                
                print(f".::.Valid chunk {chunk} size: {valid_dataloader.dataset.__len__()} sentences | Batch size: {valid_dataloader.batch_size}  | Num Batches: {valid_dataloader.__len__()}")
                tqdm_iter = tqdm(valid_dataloader, desc=f"Epoch {k:02d}/{epochs}|Step {chunk}")
                for batch in tqdm_iter:
                    decoder_input = batch['decoder_input']
                    decoder_mask = batch['decoder_mask']
                    targets = batch['label'] # (batch_size, seq_len)
                    pred_logits = model(decoder_input, decoder_mask) # (batch_size, seq_len, vocab_size)
                    valid_loss = loss_func(pred_logits.view(-1, m_config['vocab_size']), targets.view(-1))
                    tqdm_iter.set_postfix({"Valid loss": f"{valid_loss.item():10.4f}"})
                    valid_cost += valid_loss.item()
                    valid_cost_count += 1
                chunk += 1
        
        train_cost = train_cost / train_cost_count
        valid_cost = valid_cost / valid_cost_count
        
        payload = {
            'epoch': epoch,
            'train_loss': train_cost,
            'valid_loss': valid_cost,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }

        if tuning_mode == False:
            num_ordered = str(k).rjust(len(str(epochs)), '0')
            epoch_model_file = f"{checkpoint_dir}/{g_config['model']['checkpoint']['state_file_basename_epoch']}-{num_ordered}.pt"
            save_checkpoint(payload, epoch_model_file)
        else:
            with tempfile.TemporaryDirectory() as chkp_dir:
                torch.save(
                    payload,
                    os.path.join(chkp_dir, g_config['model']['tuning']['checkpoint_basename'])
                )
                ray_train.report(
                    {
                        'train_cost': train_cost,
                        'valid_cost': valid_cost
                    },
                    checkpoint=ray_train.Checkpoint.from_directory(chkp_dir)
                )
        
        print(f"{' '*4}.:.Train Cost: {train_cost} | Valid Cost: {valid_cost}\n")

        if epoch < epochs:
            stop_or_continue()
        
        epoch_loss_list.append((k, train_cost, valid_cost))

    if tuning_mode == False:
        return model
    
    return None


def generate(model:cores.TransformerDecoder, sp_tokenizer:VnTokenizer, m_config:dict, prompts:list[str], temperature:float, top_type:str=None, top_p:float=None, top_k:int=None, max_generation_len:int=None):
    
    prompts = sp_tokenizer.word_tokenize_text_all(prompts, lower_case=True)
    
    if max_generation_len == None:
        max_generation_len = m_config['sequence_len']
    
    prompt_tokens = []
    prompt_lens = []
    for p in prompts:
        encode_tokens = sp_tokenizer.encode(p, sos=True)
        prompt_tokens.append(encode_tokens)
        prompt_lens.append(len(encode_tokens))

    batch_size = len(prompt_tokens)
    assert batch_size <= m_config['train_batch_size']

    max_prompt_len = max(prompt_lens)
    min_prompt_len = min(prompt_lens)
    print(prompt_tokens)
    print(max_prompt_len, min_prompt_len)

    assert max_prompt_len <= m_config['sequence_len']

    total_prompt_len = min(max_prompt_len + max_generation_len - 1, m_config['sequence_len'])
    print(total_prompt_len)

    prompt_padding_tokens = torch.full((batch_size, total_prompt_len), fill_value=sp_tokenizer.pad_token_id, dtype=torch.int64)
    
    for b, prompt in enumerate(prompt_tokens):
        prompt_padding_tokens[b, :len(prompt)] = torch.tensor(prompt, dtype=torch.int64)
    # print(prompt_padding_tokens)

    eos_reached = torch.tensor([False] * batch_size)
    # print(eos_reached)

    for cur_pos in range(min_prompt_len, total_prompt_len):
        model.eval()
        with torch.inference_mode():
            prompt_padding_mask_tokens = (prompt_padding_tokens != sp_tokenizer.pad_token_id)
            prompt_masks = torch.cat(
                [(p & dataset.generate_causal_mask(total_prompt_len)).unsqueeze(1) for p in prompt_padding_mask_tokens.int()],
                dim=0
            )
            assert prompt_padding_tokens.size(0) == prompt_masks.size(0)
            logits = model(prompt_padding_tokens, prompt_masks)
            
            # print(logits[:,-1,:])
            # -1: In each batch, take the last token represents the word to be the next
            logits = logits[:, -1, :] # (batch_size, seq_len, vocab_size) -> (batch_size, vocab_size)
            
            next_token = None
            if top_type != None:
                if top_type.lower() == 'k':
                    logits = topK(logits, top_k)
                    if temperature > 0:
                        logits = logits / temperature
                        probs = torch.softmax(logits, dim=-1)  # (batch_size, seq_len)
                        next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

                elif top_type.lower() == 'p':
                    if temperature > 0:
                        logits = logits / temperature
                        probs = torch.softmax(logits, dim=-1)  # (batch_size, seq_len)
                        next_token = topP(probs, top_p)
                else:
                    pass

            if next_token == None:
                next_token = torch.argmax(torch.softmax(logits, dim=-1), dim=-1, keepdim=True)

            # print(next_token)
            # print(next_token.reshape(-1))
            
            next_token = next_token.reshape(-1) # next_token.flatten()
            
            # print(next_token)
            # print(prompt_padding_mask_tokens[:, cur_pos])
            # print(prompt_padding_tokens[:, cur_pos])
            
            # Only replace token if prompt has already been generated
            next_token = torch.where(condition=prompt_padding_mask_tokens[:, cur_pos], # do if condition == True
                                     input=prompt_padding_tokens[:, cur_pos],
                                     other=next_token)            
            
            # Set the next token to the current pos of each prompt
            prompt_padding_tokens[:, cur_pos] = next_token

            # print(~prompt_padding_mask_tokens[:, cur_pos])
            # print((next_token == sp_tokenizer.eos_token_id))
            # print((~prompt_padding_mask_tokens[:, cur_pos]) & (next_token == sp_tokenizer.eos_token_id))
            # print(eos_reached)

            eos_reached |= (~prompt_padding_mask_tokens[:, cur_pos]) & (next_token == sp_tokenizer.eos_token_id)            
            if all(eos_reached):
                break
    
    gen_list = sp_tokenizer.decode_all(prompt_padding_tokens.tolist())
    for item in gen_list:
        print(item + "\n\n")
    
    return True


def topK(logits:torch.Tensor, top_k:int) -> torch.Tensor:
    """
    """
    top_logits, top_pos = torch.topk(logits, top_k)
    
    # Get the min value of top_k (aka the last item in top_k)
    # min_logit = top_logits[:, -1].unsqueeze(-1) # (min_value) --> (batch_size, min_value)
    min_values = top_logits[:, -1].view(logits.size(0), -1) # (batch_size, min_value)
    
    # Replace the values in the logits by the -inf masked if its values < min value
    masked_logits = torch.where(condition=logits < min_values,
                         input=torch.tensor(float('-inf')),
                         other=logits)
    # print(masked_logits)
    return masked_logits
    

def topP(probs:torch.Tensor, top_p:float) -> torch.Tensor:
    """
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > top_p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token