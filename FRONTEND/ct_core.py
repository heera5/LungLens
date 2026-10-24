"""LungLens - shared deep-learning code (used by the Colab notebook AND the Flask app).
Needs: torch, torchvision."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

ARCHS = ('densenet121', 'resnet50', 'efficientnet_b0')
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_model(arch, num_classes=3, pretrained=True):
    """ImageNet-pretrained backbone with a new classification head."""
    if arch == 'densenet121':
        m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    elif arch == 'resnet50':
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == 'efficientnet_b0':
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError('Unknown architecture: ' + str(arch))
    return m


def head_parameters(model, arch):
    head = model.fc if arch == 'resnet50' else model.classifier
    return list(head.parameters())


def split_forward(model, arch):
    """Return (feature_fn, head_fn): the network split at its last convolutional feature map,
    which is where Grad-CAM needs the gradients."""
    if arch == 'densenet121':
        feat = lambda x: model.features(x)
        head = lambda f: model.classifier(torch.flatten(F.adaptive_avg_pool2d(F.relu(f), (1, 1)), 1))
    elif arch == 'resnet50':
        def feat(x):
            x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
            return model.layer4(model.layer3(model.layer2(model.layer1(x))))
        head = lambda f: model.fc(torch.flatten(model.avgpool(f), 1))
    elif arch == 'efficientnet_b0':
        feat = lambda x: model.features(x)
        head = lambda f: model.classifier(torch.flatten(model.avgpool(f), 1))
    else:
        raise ValueError('Unknown architecture: ' + str(arch))
    return feat, head


def grad_cam(model, arch, x, class_idx=None):
    """Grad-CAM for one image tensor x of shape [1,3,H,W].
    Returns (cam in [0,1] as a 2-D numpy array, class probabilities, class index explained)."""
    model.eval()
    feat_fn, head_fn = split_forward(model, arch)
    with torch.enable_grad():
        f = feat_fn(x)
        logits = head_fn(f)
        probs = torch.softmax(logits, dim=1)[0].detach()
        if class_idx is None:
            class_idx = int(probs.argmax())
        grads = torch.autograd.grad(logits[0, class_idx], f)[0]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * f).sum(dim=1))[0].detach()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.cpu().numpy(), probs.cpu().numpy(), class_idx
