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

    # ��ԭʼ����ת��Ϊ����
    def __getitem__(self,index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]
        # �ı�תtoken����תid��tokenizer�Ѿ��Ӳ��Ϊ�ʣ��ٰѴ�תΪ�ʱ��е�id
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        # �����ӣ��ó��ȴﵽseq_len��������eos��sos
        enc_num_padding_tokens = self.seq_len-len(enc_input_tokens)-2
        # decoder����ֻ��sosû��eos������paddingҪ��һ��token
        dec_num_padding_tokens = self.seq_len-len(dec_input_tokens)-1

        # ȷ������token��Ϊ����
        if enc_num_padding_tokens<0 or dec_num_padding_tokens<0:
            raise ValueError('Sentence is too long')
        
        # Ϊencoder�������tensor�����[SOS] [EOS] [PAD]
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens,dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token]*enc_num_padding_tokens,dtype=torch.int64)
        ])

        # Ϊdecoder���������tensor�����[SOS] [PAD]
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens,dtype=torch.int64),
            torch.tensor([self.pad_token]*dec_num_padding_tokens,dtype=torch.int64)
        ])

        # Ϊdecoder�����/label���tensor��ֻ���[EOS]
        label = torch.cat([
            torch.tensor(dec_input_tokens,dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token]*dec_num_padding_tokens,dtype=torch.int64)
        ])

        # ��������tensor�����Ƿ�����seq_len
        assert encoder_input.size(0)==self.seq_len
        assert decoder_input.size(0)==self.seq_len
        assert label.size(0)==self.seq_len

        # encoder��mask������������token
        # decoder��mask�������ε�����token��δ����token
        return {
            'encoder_input':encoder_input, #(seq_len)
            'decoder_input':decoder_input, #(seq_len)
            'encoder_mask':(encoder_input!=self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1,1,seq_len)
            'decoder_mask':(decoder_input!=self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1,seq_len,seq_len)
            'label':label, # (seq_len)
            'src_text':src_text, # ԭ�ģ����ڿ��ӻ�
            'tgt_text':tgt_text  # ���ģ����ڿ��ӻ�
        }
def causal_mask(size):
    # triu�����е�diagonal=1��ʾ���Խ���+1�����������Ǿ�������Խ�������Ϊ0
    mask = torch.triu(torch.ones(1,size,size),diagonal=1).type(torch.int)
    # ����ȡ�������������Ǿ���
    return mask == 0