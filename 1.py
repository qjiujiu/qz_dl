# import thrember
# thrember.create_vectorized_features('./data/EMBER2024/APK_all', label_type="family")

from config.params_parser.parser import ArgsParser
from config.datasets.dataset_instance.mal_api_embed import get_embeding


# python 1.py --model lstm  --batch-size 8 --epochs 30 --lr 0.001 --dropout-prob 0.5  --embedding-dim 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278 -cp checkpoints/2025-07-09/LSTMTextClassifier/20250709-0954-ff28631f_weights.pth
if __name__ == "__main__":
    cfg = ArgsParser().create_nlp_config()
    get_embeding(cfg)
