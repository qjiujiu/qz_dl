conda activate qtorch

# Define models and noise types
models=("tcn" "lstm" "conv1d" "conv2d")
noise_types=("ctxm" "ctxm_sa" "ctxm_sape" "ctxm_mlp" "ctxm_gas")

# Loop over models and noise types to run the training scripts
for model in "${models[@]}"; do
  for noise in "${noise_types[@]}"; do
    python train_mc.py -c $model -n $noise
  done
done
