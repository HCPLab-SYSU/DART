# Score Prediction Nets
from cv2 import mean
from sympy import false
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small
import torchvision.models as models
from .tools import MLP

class ScorePredNetV1(nn.Module):
    def __init__(self, features: nn.Module, feature_dim: int, hidden_dim=None):
        super(ScorePredNetV1, self).__init__()
        self.features = features
        self.mlp = MLP(feature_dim, hidden_dim or feature_dim, 1)

    def forward(self, x, shape=(14,14)):
        with torch.no_grad():
            x = self.features(x)

        x = x.permute(0,2,3,1)
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2)

        x = F.interpolate(x, size=shape, mode='bilinear', align_corners=False)
        x = x - x.mean()
        x = x / (x.std() + 1e-8)
        x = F.sigmoid(x)+0.1
        return x.view(x.size(0), -1)
    
class ScorePredNetV2(nn.Module):
    def __init__(self, features: nn.Module, feature_dim: int, hidden_dim=None):
        super(ScorePredNetV2, self).__init__()
        self.features = features
        self.mlp = MLP(feature_dim, hidden_dim or feature_dim, feature_dim)

    def forward(self, x, shape=(14,14)):
        with torch.no_grad():
            f = self.features(x)
        f_shape = f.shape[2:]
        n = x.size(0)
        d = f.size(1)
        f = f.view(n,d,-1).permute(0,2,1)

        s = (
            f*F.sigmoid( self.mlp(f.mean(dim=1,keepdim=True)) )
        ).sum(-1)

        s = F.interpolate(s.view(n,1,*f_shape), size=shape, mode='bilinear', align_corners=False)
        s = s - s.mean()
        s = s / (s.std() + 1e-8)
        s = F.sigmoid(s*2)
        return s.view(x.size(0), -1)
    
class ScorePredNet(nn.Module):
    def __init__(self, features: nn.Module, feature_dim: int, hidden_dim=None, version='v1'):
        super(ScorePredNet, self).__init__()
        SPN = ScorePredNetV1 if version == 'v1' else ScorePredNetV2
        self.spn = SPN(features, feature_dim, hidden_dim)
        self.version = version
    def forward(self, *args, **kwargs):
        return self.spn(*args, **kwargs)
    
class MobileNetSmallPred(nn.Module):
    def __init__(self, version='v1'):
        super(MobileNetSmallPred, self).__init__()
        mobilenet = models.mobilenet_v3_small(pretrained=True)
        if version == 'v1':
            features = mobilenet.features[:11]
            dims=(96,96)
        else:
            features = mobilenet.features[:13]
            dims=(576,96)

        for param in features.parameters():
            param.requires_grad = False

        self.spn = ScorePredNet(features, *dims, version=version)
    def forward(self, x, shape=(14,14)):
        return self.spn(x, shape)

class MobileNetLargePred(nn.Module):
    def __init__(self, version='v1'):
        super(MobileNetLargePred, self).__init__()
        mobilenet = models.mobilenet_v3_large(pretrained=True)
        if version == 'v1':
            features = mobilenet.features[:15]
            dims=(160,160)
        else:
            features = mobilenet.features[:17]
            dims=(960,96)

        for param in features.parameters():
            param.requires_grad = False

        self.spn = ScorePredNet(features, *dims, version=version)
    def forward(self, x, shape=(14,14)):
        return self.spn(x, shape)
    

class EfficientNetB0Pred(nn.Module):
    def __init__(self, version='v1'):
        super(EfficientNetB0Pred, self).__init__()
        model = models.efficientnet_b0(pretrained=True)
        features = model.features
        dims=(1280,96)
        for param in features.parameters():
            param.requires_grad = False

        self.spn = ScorePredNet(features, *dims, version=version)
    def forward(self, x, shape=(14,14)):
        return self.spn(x, shape)
    