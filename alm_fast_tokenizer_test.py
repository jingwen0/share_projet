from tokenizers import processors
from transformers.convert_slow_tokenizer import SpmConverter
from transformers import PreTrainedTokenizerFast, AutoTokenizer

class GLMArabicConverter(SpmConverter):
    def vocab(self, proto):
        vocab = [(piece.piece, piece.score) for piece in proto.pieces]
        vocab = vocab + [("<|endoftext|>", 0), ("[SEP]", 0), ("[CLS]", 0), \
                         ("[MASK]", 0), ("[UNUSED1]", 0), ("[UNUSED2]", 0), 
                         ("<|startofpiece|>", 0), ("<|endofpiece|>", 0), ("[sMASK]", 0), \
                         ("[gMASK]", 0), ("[UNK]", -100)]
        return vocab
    
    def post_processor(self):
        return processors.TemplateProcessing(
            single="[CLS] $0 <|endoftext|>",
            special_tokens=[
                ("[CLS]", self.original_tokenizer.convert_tokens_to_ids("[CLS]")),
                ("<|endoftext|>", self.original_tokenizer.convert_tokens_to_ids("<|endoftext|>"))])
    
def get_alm_tokenizer_fast(alm_slow_tokenizer):
    fast_alm_tokenizer = PreTrainedTokenizerFast(tokenizer_object=GLMArabicConverter(alm_slow_tokenizer).converted())
    fast_alm_tokenizer.add_special_tokens(alm_slow_tokenizer.special_tokens_map)
    return fast_alm_tokenizer
        

if __name__ == "__main__":
    
    alm_slow_tokenizer_path = "HF-ALM"
    alm_slow_tokenizer = AutoTokenizer.from_pretrained(alm_slow_tokenizer_path, trust_remote_code=True)
    alm_fast_tokenizer = get_alm_tokenizer_fast(alm_slow_tokenizer)

    slow_alm_inputs = alm_slow_tokenizer(["شرم الشيخ وجهة سياحية شهيرة [gMASK]"])
    print(slow_alm_inputs.input_ids)

    fast_alm_inputs = alm_fast_tokenizer(["شرم الشيخ وجهة سياحية شهيرة [gMASK]"])
    print(fast_alm_inputs.input_ids)

    print(fast_alm_inputs.word_ids())
