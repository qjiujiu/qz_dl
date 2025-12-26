conda activate qtorch

# Define models and noise types
models=("tcn" "lstm" "conv1d" "conv2d")
noise_types=("ctx" "ctx_sa" "ctx_sape" "ctx_mlp" "ctx_gas")

# Loop over models and noise types to run the training scripts
for model in "${models[@]}"; do
  for noise in "${noise_types[@]}"; do
    python train_demo.py -c $model -n $noise
  done
done
