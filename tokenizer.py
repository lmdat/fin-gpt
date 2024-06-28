import os
from sentencepiece import SentencePieceProcessor
from underthesea import word_tokenize

class VnTokenizer:
    def __init__(self, sp_model_file:str) -> None:
        """
        """
        assert os.path.exists(sp_model_file), f"{sp_model_file} does not exist."
        
        self.tokenizer = SentencePieceProcessor(model_file=sp_model_file)
        self.vocab_size = self.tokenizer.vocab_size()
        self.sos_token_id = self.tokenizer.bos_id()
        self.eos_token_id = self.tokenizer.eos_id()
        self.pad_token_id = self.tokenizer.pad_id()

        print(f"Tokenizer model: {sp_model_file}")
        print(f"Vocab size: {self.vocab_size}")
        print(f".:.{self.tokenizer.id_to_piece(self.sos_token_id)}: {self.sos_token_id}")
        print(f".:.{self.tokenizer.id_to_piece(self.eos_token_id)}: {self.eos_token_id}")
        print(f".:.{self.tokenizer.id_to_piece(self.pad_token_id)}: {self.pad_token_id}")

    def encode(self, sentence:str, sos:bool=False, eos:bool=False) ->list[int]:
        token_ids = self.tokenizer.encode_as_ids(sentence)

        if sos == True:
            token_ids = [self.sos_token_id] + token_ids
        
        if eos == True:
            token_ids = token_ids + [self.eos_token_id]

        return token_ids
    
    def decode(self, token_ids:list[int]) -> str:
        return self.origin_sentence(self.tokenizer.decode(token_ids))
    
    def decode_all(self, prompt_tokens:list[list[int]]) -> list[str]:
        return [self.decode(tokens) for tokens in prompt_tokens]
            
    
    def word_tokenize_text(self, prompt:str, lower_case:bool=True) -> str:
        if lower_case == True:
            prompt = prompt.lower()
        return word_tokenize(prompt, format='text')
    
    
    def word_tokenize_text_all(self, prompts:list[str], lower_case:bool=True) -> list[str]:
        return [self.word_tokenize_text(p, lower_case) for p in prompts]
        
    
    def origin_sentence(self, text:str) -> str:
        patterns = {
            '"': '',
            '_': ' ',
            ' ,': ',',
            ' .': '.',
            ' ?': '?',
            ' :': ':'
        }
        for k, v in patterns.items():
            text = text.replace(k, v)

        # Capitalize the first char of the word after dot sign
        i = text.find('. ')
        if i != -1:
            while True:
                i += 2
                if i < len(text):
                    text = text[:i] + text[i].capitalize() + text[(i + 1):]
                i = text.find('. ', i)
                if i == -1:
                    break
        
        text = text[0].capitalize() + text[1:]
        return text