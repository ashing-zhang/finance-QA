#!/bin/sh

MODEL_NAME=../models/Tongyi-Finance-14B-Chat # ONLY OPT AND BLOOM MODELS ARE SUPPORTED FOR NOW
BATCHSIZE=8 # batch size
GEN_LEN=512 # number of tokens to generate

USE_CPU_OFFLOAD=1 # whether to use model weights cpu offloading when running with deepspeed zero inference
USE_KV_OFFLOAD=1 # whether to use kv cache cpu offloading when running with deepspeed zero inference
USE_HF_MODEL=0 # whether to use the original HF model(no kv cache offloading support) or not
USE_QUANT=1 # whether to use model weigths quantization or not
QUANT_BITS=4

if [ $USE_CPU_OFFLOAD -eq 1 ]; then
    CPU_OFFLOAD="--cpu-offload"
else
    CPU_OFFLOAD=""
fi

if [ $USE_KV_OFFLOAD -eq 1 ]; then
    KV_OFFLOAD="--kv-offload"
else
    KV_OFFLOAD=""
fi

if [ $USE_HF_MODEL -eq 1 ]; then
    HF_MODEL="--hf-model"
else
    HF_MODEL=""
fi

if [ $USE_QUANT -eq 1 ]; then
    QUANT="--quant_bits"   
else
    QUANT=""
fi


deepspeed --num_gpus 1 test_lora.py --model ${MODEL_NAME} --batch-size ${BATCHSIZE} --gen-len ${GEN_LEN} ${CPU_OFFLOAD} ${KV_OFFLOAD} ${QUANT} ${QUANT_BITS}