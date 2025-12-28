#!/bin/bash

# 注意：在脚本中使用 conda activate 有时需要先初始化 shell hook
eval "$(conda shell.bash hook)"
conda activate qtorch


# 使用 source 可以确保在当前 shell 环境下运行，继承 conda 环境

echo "========================================================"
echo "🚀 Starting Binary (2-class) Experiments..."
echo "========================================================"
source scripts/run2.sh



echo "========================================================"
echo "🚀 Starting Multi-label (15-class) Experiments..."
echo "========================================================"
source scripts/run-m.sh


echo "========================================================"
echo "🚀 Starting BERT Experiments..."
echo "========================================================"
source scripts/run-bert.sh

# echo "========================================================"
# echo "🚀 Starting Multi-class (8-class) Experiments..."
# echo "========================================================"
# source scripts/run8.sh

echo "🎉 All experiments finished!"