conda activate qtorch

# Define models and noise types
# models=("tcn" "lstm" "conv1d" "conv2d")
models=("conv2d")
noise_types=("ctx8" "ctx8_sa" "ctx8_sape" "ctx8_mlp" "ctx8_gas")

# Loop over models and noise types to run the training scripts
for model in "${models[@]}"; do
  for noise in "${noise_types[@]}"; do
    python train_8c.py -c $model -n $noise
  done
done
