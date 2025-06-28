# -*- coding: gbk -*-
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer

from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer  # tokenizer对应的trainer，通过给定的句子创建词表
from tokenizers.pre_tokenizers import Whitespace  # 按空格分词

from pathlib import Path

from dataset import BilingualDataset, causal_mask


def get_or_build_tokenizer(config, ds, lang):
    """
    创建分词器
    参数: 
        config: 模型的配置
        ds: 数据集
        lang: 分词器的语言
    返回: 
        Tokenizer分词器实例
    """
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))  # 分词器遇到不在词表的词替换为[UNK]
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_all_sentences(ds, lang):
    """
    从数据集中取指定语言的句子
    参数：
        ds：数据集
        lang：制定语言，en或者it（Italy）
    """
    for item in ds:
        yield item["translation"][lang]


def get_ds(config):
    """
    加载数据集并创建分词器
    """
    # 从huggingface下载opus_books数据集中'en-it'自己，选择'train'部分，之后再分出validation部分
    ds_raw = load_dataset(
        "opus_books", f'{config["lang_src"]}-{config["lang_tgt"]}', split="train"
    )

    # 创建分词器
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    # 90%训练，10%测试
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )
    val_ds = BilingualDataset(
        val_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )

    # 找出训练集中源语言和目标语言里最长的句子长度分别是多少
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
        tgt_ids = tokenizer_tgt.encode(item["translation"][config["lang_tgt"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source language: {max_len_src}")
    print(f"Max length of target language: {max_len_tgt}")

    train_dataloader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


# 训练模型
# 创建模型
from model import build_transformer


def get_model(config, vocab_src_len, vocab_tgt_len):
    """
    创建模型
    """
    model = build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config["seq_len"],
        config["seq_len"],
        config["d_model"],
    )
    return model


# 为预加载权重编写代码，若指定了预加载的模型，则直接加载。
# 指定loss函数，声明padding不参与loss计算
from torch.utils.tensorboard import SummaryWriter
from config import get_weights_file_path, get_config
from tqdm import tqdm


def train_model(config):
    """
    训练模型
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 确定权重文件目录存在
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    # 加载数据集和分词器
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(
        config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()
    ).to(device)

    # 启动Tensorboard可视化loss
    writer = SummaryWriter(config["experiment_name"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    # 若指定预加载权重，则预加载
    inital_epoch = 0
    global_step = 0
    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        inital_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
    ).to(device)

    # 训练循环 training loop
    for epoch in range(inital_epoch, config["num_epochs"]):
        model.train()
        # 画进度条
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)

            # 将tensor输入模型并计算
            encoder_ouput = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(
                encoder_ouput, encoder_mask, decoder_input, decoder_mask
            )
            proj_output = model.project(decoder_output)

            label = batch["label"].to(device)

            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)
            )
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            # 记录loss
            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            # loss反向传播
            loss.backward()

            # 更新参数
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
        run_validation(
            model,
            val_dataloader,
            tokenizer_src,
            tokenizer_tgt,
            config["seq_len"],
            device,
            lambda msg: batch_iterator.write(msg),
            global_step,
            writer,
        )
        model_filename = get_weights_file_path(config, f"{epoch:02d}")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )


# 测试循环
# validation loop
def run_validation(
    model,
    validatoin_ds,
    tokenizer_src,
    tokenizer_tgt,
    max_len,
    device,
    print_msg,
    global_state,
    writer,
    num_examples=2,
):
    # 告诉pytorch将模型置于eval模型，关闭dropout、batchnorm中的随机因素
    model.eval()
    count = 0
    source_texts = []
    expected = []
    predicted = []

    console_width = 80

    # 关闭pytorch梯度计算，with中只需要推理结果，不需要训练
    with torch.no_grad():
        for batch in validatoin_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            # 确定validation的batch为1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for valid"

            # 模型推理时，encoder_output只需要计算一次，然后重复为每个token计算decoder_output
            model_out = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                tokenizer_src,
                tokenizer_tgt,
                max_len,
                device,
            )

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]

            # 用目标语言的分词器把模型输出的token序列转为文本
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # 打印输出，为了不影响tqdm进度条正常刷新，使用print_msg
            print_msg("-" * console_width)
            print_msg(f"Source text: {source_text}")
            print_msg(f"Target text: {target_text}")
            print_msg(f"Predicted text: {model_out_text}")
            print_msg("-" * console_width)

            # 如果预测了足够的句子，停止循环
            if count >= num_examples:
                break


def greedy_decode(
    model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device
):
    """
    使用贪婪策略，缓存encoder输出，计算decoder输出

    参数: 
        model: 模型
        source: 源语言句子
        source_mask: 源语言句子的padding mask
        tokenizer_src: 源语言的分词器
        tokenizer_tgt: 目标语言的分词器
        max_len: 句子最大长度
        device: 运行模型的设备
    """
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    # 计算encoder输出，并将结果重用于计算每个token的decoder输出
    encoder_output = model.encode(source, source_mask)

    # 模型推理计算过程：
    # 首先输入句子起始符[SOS]作为decoder的第一个输入，然后得到模型输出的翻译句子的第一个词
    # 在每一步迭代中将上一步输出附加在输入中，使模型继续输出下一个词，直到遇到[EOS]或者达到句子最大长度

    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        # 判断是否达到句子最大长度
        if decoder_input.size(1) == max_len:
            break

        # 为target(decoder input)创建mask，让模型只能看到当前token之前的词
        decoder_mask = causal_mask(
            decoder_input.size(1).type_as(source_mask).to(device)
        )
        # 计算decoder输出，复用encoder_output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # 计算下一个词，使用projection(投影层)计算下一个token概率分布，只取最后一个词的projection
        prob = model.project(out[:, -1])
        # 取概率最大的词作为下一个token
        _, next_word = torch.max(prob, dim=1)
        # 再将next_word添加到decoder_input的末尾
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device),
            ],
            dim=1,
        )

        # 如果遇到[EOS]，则停止循环
        if next_word == eos_idx:
            break
    return decoder_input.squeeze(0)


import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
