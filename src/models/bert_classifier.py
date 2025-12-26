from transformers import AutoModelForSequenceClassification
import torch.nn as nn

class BertSeqClassifier(nn.Module):
    def __init__(self, output_dim: int = 8, **kwargs) -> None:
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path = 'microsoft/codebert-base', 
            num_labels= output_dim
        )
        
    def forward(self, x):
        return self.model(x)