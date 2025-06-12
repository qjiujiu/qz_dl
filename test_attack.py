import torch
from models.cv.lenet import LeNet

def test_attack():
    model = LeNet(num_classes=10)
    x = torch.randn(32, 1, 28, 28)
    
    y = model(x)

    print(y.size())

    z = model.fgsm_attack(x, y, epsilon=0.01)
    print(z.size())
    

test_attack()
