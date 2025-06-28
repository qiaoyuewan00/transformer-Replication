from pathlib import Path

def get_config():
    return {
        "batch_size":8,
        "num_epochs":20,
        # 实际训练过程中，lr会调整。比如一开始设置较大lr，训练过程中逐渐减小lr
        "lr":10**-4,
        "seq_len":350,
        "d_model":512,
        "lang_src":"en",
        "lang_tgt":"it",
        "model_folder":"weights",
        "model_basename":"tmodel_",
        # 预加载模型，如果不为空则加载参数，否则从头训练
        "preload":None,
        # 分词文件，如_en.json
        "tokenizer_file":"tokenizer_{0}.json",
        # 为tensorboard提供log路径
        "experiment_name":"runs/tmodel"
    }

def get_weights_file_path(config,epoch:str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.')/model_folder/model_filename)