# python train_8c_adv.py -c tcn -n ctx8 --adv-type PGD  --steps 1 --epsilon 0.03 --epochs 30
# python train_8c_adv.py -c tcn -n ctx8 --adv-type PGD  --steps 2 --epsilon 0.03 --epochs 30
# python train_8c_adv.py -c tcn -n ctx8 --adv-type PGD  --steps 3 --epsilon 0.03 --epochs 30
# python train_8c_adv.py -c tcn -n ctx8 --adv-type PGD  --steps 4 --epsilon 0.03 --epochs 30
# python train_8c_adv.py -c tcn -n ctx8 --adv-type PGD  --steps 5 --epsilon 0.03 --epochs 30


conda activate qtorch

# Define models and noise types (取消 "conv2d")
models=("tcn" "lstm" "conv1d")
steps=(1 2 3 4 5)

# # Loop over models and steps to run the training scripts
# for step in "${steps[@]}"; do
#     for model in "${models[@]}"; do
#         python train_2c_adv.py -c "$model" -n ctx2 --adv-type PGD --steps "$step" --epsilon 0.03 --epochs 12
#     done
# done

# Loop over models and steps to run the training scripts
for step in "${steps[@]}"; do
    for model in "${models[@]}"; do
        python train_mc_adv.py -c "$model" -n ctxm --adv-type PGD --steps "$step" --epsilon 0.03 --epochs 30
    done
done