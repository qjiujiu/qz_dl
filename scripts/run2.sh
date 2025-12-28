conda activate qtorch

# Define models and noise types
# models=("tcn" "lstm" "conv1d" "conv2d")
# noise_types=("ctx2" "ctx2_sa" "ctx2_sape" "ctx2_mlp" "ctx2_gas")

models=("conv2d")
noise_types=("ctx2_sa" "ctx2_sape" "ctx2_mlp" "ctx2_gas")

# Loop over models and noise types to run the training scripts
for model in "${models[@]}"; do
  for noise in "${noise_types[@]}"; do
    python train_2c.py -c $model -n $noise
  done
done
