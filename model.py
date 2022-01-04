import os
import torch
import torchvision
import torch.nn as nn

def load_model(args, net, optim):
    ckpt_dir = args.ckpt_dir
    if not os.path.exists(ckpt_dir):
        epoch = 0

        return epoch, net, optim
    
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst = [p for p in ckpt_lst if p.endswith("pth")]
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load("%s/%s" % (ckpt_dir, ckpt_lst[-1]), map_location=device)

    net.load_state_dict(dict_model["net"])
    optim.load_state_dict(dict_model["optim"])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return epoch, net, optim

class ResNetRegressor(nn.Module):
    """
    Build a regression model by calling a constructor in torchvision.models subpackage.
    """
    def __init__(self, out_channels, pretrained=True):
        super(ResNetRegressor, self).__init__()

        self.resnet = torchvision.models.resnet50(pretrained=pretrained)
        
        self.fc = nn.Linear(1000, out_channels)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        sig = nn.Sigmoid()
        return sig(x)