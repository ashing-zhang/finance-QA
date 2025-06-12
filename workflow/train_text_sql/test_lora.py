from modelscope import AutoTokenizer,AutoModelForCausalLM,AutoConfig
from peft import get_peft_model, PeftModel
import sqlite3
import os
import argparse
import torch
import gc
from vllm import LLM
import torch.distributed as dist


def merge_model(args):
    num_gpus = torch.cuda.device_count()
    # Initialize distributed if needed
    if num_gpus > 1:
        dist.init_process_group(backend="nccl")

    # Load base model
    print("Before loading - Memory allocated:", torch.cuda.memory_allocated()/1024**2, "MB")
    print("Before loading - Memory reserved:", torch.cuda.memory_reserved()/1024**2, "MB")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_path,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    print("After loading - Memory allocated:", torch.cuda.memory_allocated()/1024**2, "MB")
    print("After loading - Memory reserved:", torch.cuda.memory_reserved()/1024**2, "MB")
    print("Model device:", next(model.parameters()).device)
    
    # Load and merge LoRA
    model = PeftModel.from_pretrained(model, args.lora_path)
    print("Loaded PEFT model. Merging...")
    model = model.merge_and_unload()
    print("Merge complete.")

    model = model.eval()
    
    # Initialize vLLM
    llm = LLM(
        model=model,
        tokenizer=model_name_path,
        tensor_parallel_size=num_gpus,
        dtype="float16",
        max_model_len=args.gen_len
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_path, trust_remote_code=True)
    return llm, tokenizer

def test_sql_lora(args):
    with torch.no_grad():
        model, tokenizer = merge_model(args)
    conn = sqlite3.connect(db_path)
    while True:
        print("用户输入SQL生成诉求：", end="")
        prompt = input()
        with torch.no_grad():
            # 由于训练数据按对话模板进行了处理，所以推理阶段不能用generate替代chat
            response, history = model.chat(tokenizer, prompt, history=None, system="You are a helpful assistant that translates natural language to SQL queries. ")
        # 推理的system message需要和训练时保持一致，不然输出结果是灾难性的
        # response, history = model.chat(tokenizer, prompt, history=None, system="")
        print("当前LORA模型生成SQL语句为：", response)
        response = response.replace("”", '').replace("“", '')
        cur = conn.cursor()
        print('sql: ' + response)
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

        # if len(sql_answer) > 10:
        #     raise ValueError("too many query results")

        print('当前SQL语句查询结果：', sql_answer)
    
def test_ner_lora(model_name,lora_path):
    model,tokenizer = merge_model(model_name,lora_path)
    while True:
        print("当前用户输入问题：", end="")
        prompt = input()
        response, history = model.module.chat(tokenizer, prompt, history=None,system='你是一个NER智能体') 
        print('当前用户问题识别结果：',response)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='../models/Tongyi-Finance-14B-Chat', help="model name or path")
    parser.add_argument("--lora_path", type=str, default='./model_save/sql_lora/merged', help="lora model path")
    parser.add_argument("--db_path", type=str, default='data/dataset/fund_data.db', help="database path")
    parser.add_argument("--pin-memory", type=int, default=0, help="whether to pinned CPU memory for ZeRO offloading")
    parser.add_argument("--cpu-offload", action="store_true", help="Use cpu offload.")
    parser.add_argument("--disk-offload", action="store_true", help="Use disk offload.")
    parser.add_argument("--offload-dir", type=str, default="./offload_dir", help="Directory to store offloaded cache.")
    parser.add_argument("--kv-offload", action="store_true", help="Use kv cache cpu offloading.")
    parser.add_argument("--use_gds", action="store_true", help="Use NVIDIA GPU DirectStorage to transfer between NVMe and GPU.")
    parser.add_argument("--local_rank", type=int, help="local rank for distributed inference")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gen-len", type=int, default=128,  help="number of tokens to generate")
    args = parser.parse_args()

    test_sql_lora(args.model_name, args.lora_path, args.db_path, args)
