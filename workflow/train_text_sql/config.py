from peft import LoraConfig
import torch
from transformers import AutoTokenizer

# 配置参数
class Config:
    seed = 42
    model_name = "workflow/models/Tongyi-Finance-14B-Chat"    # lora微调该模型
    train_data_dir = "workflow/text2sql_dataset_generator"
    train_json_path = "train_text_sql.json"
    # train_json_path = "train_text_sql_add.json"
    val_data_dir = "workflow/text2sql_dataset_generator"
    val_json_path = "val_text_sql.json"
    # val_json_path = "val_text_sql_add.json"
    test_data_dir = "workflow/text2sql_dataset_generator"
    test_json_path = "test_text_sql.json"
    sql_model_save_path = "workflow/train_text_sql/model_save/sql_lora"
    epochs = 5  # 最大epoch数
    early_stop_patience = 50  # 早停耐心值(不能设置太小，不然会导致模型训练不充分)
    improvement_ratio = 0.8  # 早停改善比例(不能设置太大，不然会导致模型训练不充分)
    lora_config = LoraConfig(
        r=8,  # 低秩矩阵的秩
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none", 
        target_modules = ["c_attn", "c_proj", "w1", "w2"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    max_len = 512
    system_message = "You are a helpful assistant that translates natural language to SQL queries. " 
    

