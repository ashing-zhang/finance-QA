'''
    TRT-LLM version of SQL generation using Tongyi-Finance-14B model
'''

from modelscope import AutoTokenizer
from peft import PeftModel
import sqlite3
import os
import argparse
import torch
import gc
import torch.distributed as dist
from transformers import set_seed
from accelerate import Accelerator
from tensorrt_llm import LLM

def merge_model(args):
    # 初始化加速器
    accelerator = Accelerator()
    
    print("加载TRT-LLM引擎...")
    model = LLM(
        model_dir=args.model_name_path,
        lora_dir=args.lora_path,
        dtype='bfloat16',
        use_auto_parallel=True
    )
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_path,
        trust_remote_code=True
    )
    
    # 准备tokenizer
    tokenizer = accelerator.prepare(tokenizer)
    
    print("TRT-LLM模型加载完成")
    
    return model, tokenizer

def generate_sql(model, tokenizer, prompt, system_message):
    # Format input using special tokens like in training
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eod_id
    
    # Build input with system message and user prompt
    input_text = (
        f"<|im_start|>system\n{system_message}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    output = model.generate(
        input_text,
        tokenizer=tokenizer,
        max_new_tokens=128,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    return output[0]['generated_text']

def test_sql_lora(args):
    model, tokenizer = merge_model(args)
    
    conn = sqlite3.connect(args.db_path)
    system_message = "You are a helpful assistant that translates natural language to SQL queries."
    
    while True:
        print("\n用户输入SQL生成诉求：", end="")
        prompt = input().strip()
        if not prompt:
            continue
            
        with torch.no_grad():
            # 生成SQL查询
            response = generate_sql(model, tokenizer, prompt, system_message)
        
        print("当前TRT-LLM模型生成SQL语句为：", response)
        response = response.replace("”", '').replace("“", '')
        
        # 执行SQL查询
        cur = conn.cursor()
        print('执行SQL:', response)
        try:
            cur.execute(response)
            sql_answer = cur.fetchall()
        except Exception as e:
            # 检查并替换 '交易日' 和 '交易日期'
            if '交易日期' in response:
                alt_response = response.replace('交易日期', '交易日')
            elif '交易日' in response:
                alt_response = response.replace('交易日', '交易日期')
            else:
                print(f"SQL执行失败: {e}")
                continue
            print('尝试替换后的sql: ' + alt_response)
            try:
                cur.execute(alt_response)
                sql_answer = cur.fetchall()
                response = alt_response  # 更新为成功的SQL
            except Exception as e2:
                print(f"SQL执行仍然失败: {e2}")
                continue
        print('当前SQL语句查询结果：', sql_answer)
             

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_path", type=str, default='../models/Tongyi-Finance-14B-Chat', 
                        help="模型名称或路径")
    parser.add_argument("--lora_path", type=str, default='./model_save/sql_lora/merged', 
                        help="LoRA模型路径")
    parser.add_argument("--db_path", type=str, default='../../data/dataset/fund_data.db', 
                        help="数据库路径")
    parser.add_argument("--gen_len", type=int, default=128,  
                        help="生成的最大token数")
    parser.add_argument("--seed", type=int, default=42, 
                        help="随机种子")
    args = parser.parse_args()
    
    set_seed(args.seed)  # 设置随机种子
    test_sql_lora(args)
