# -*- coding: utf-8 -*-
import torch
class BilingualDataset():
    def __init__(self,ds,tokenizer_src,tokenizer_tgt,src_lang,tgt_lang,seq_len)->None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")],dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")],dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")],dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    # 把原始数据转换为张量
    def __getitem__(self,index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]
        # 文本转token，再转id。tokenizer把句子拆分为词，再把词转为词表中的id
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        # 填充句子，让长度达到seq_len。减二是eos和sos
        enc_num_padding_tokens = self.seq_len-len(enc_input_tokens)-2
        # decoder输入只有sos没有eos，所以padding要多一个token
        dec_num_padding_tokens = self.seq_len-len(dec_input_tokens)-1

        # 确保填充的token不为负数
        if enc_num_padding_tokens<0 or dec_num_padding_tokens<0:
            raise ValueError('Sentence is too long')
        
        # 为encoder输入填充tensor，添加[SOS] [EOS] [PAD]
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens,dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token]*enc_num_padding_tokens,dtype=torch.int64)
        ])

        # 为decoder的输入填充tensor，添加[SOS] [PAD]
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens,dtype=torch.int64),
            torch.tensor([self.pad_token]*dec_num_padding_tokens,dtype=torch.int64)
        ])

        # 为decoder的输出/label填充tensor，只添加[EOS]
        label = torch.cat([
            torch.tensor(dec_input_tokens,dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token]*dec_num_padding_tokens,dtype=torch.int64)
        ])

        # 检查填充厚的tensor长度是否满足seq_len
        assert encoder_input.size(0)==self.seq_len
        assert decoder_input.size(0)==self.seq_len
        assert label.size(0)==self.seq_len

        # encoder的mask用于屏蔽填充的token
        # decoder的mask用于屏蔽掉填充的token和未来的token
        return {
            'encoder_input':encoder_input, #(seq_len)
            'decoder_input':decoder_input, #(seq_len)
            'encoder_mask':(encoder_input!=self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1,1,seq_len)
            'decoder_mask':(decoder_input!=self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1,seq_len,seq_len)
            'label':label, # (seq_len)
            'src_text':src_text, # 原文，用于可视化
            'tgt_text':tgt_text  # 译文，用于可视化
        }
def causal_mask(size):
    # triu函数中的diagonal=1表示主对角线+1，生成上三角矩阵后主对角线再置为0
    mask = torch.triu(torch.ones(1,size,size),diagonal=1).type(torch.int)
    # 矩阵取反，生成下三角矩阵
    return mask == 0