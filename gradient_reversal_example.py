import torch
import torch.nn as nn
from torch.autograd import Function
from torchvision.models.utils import load_state_dict_from_url


''' 
Very easy template to start for developing your AlexNet with DANN 
Has not been tested, might contain incompatibilities with most recent versions of PyTorch (you should address this)
However, the logic is consistent
'''


class ReverseLayerF(Function):
    # Forwards identity
    # Sends backward reversed gradients
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class RandomNetworkWithReverseGrad(nn.Module):
    #def __init__(self, **kwargs):
    def __init__(self, NUM_CLASSES=1000, NUM_DOMAINS=10):
        super(RandomNetworkWithReverseGrad, self).__init__()
        # adding conv parameters, took from AlexNet self.features = nn.Sequentia(etc.
        self.features = nn.Sequential(
        	nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # adding fc parameters, took from AlexNet self.classifier = nn.Sequential( etc)
        self.classifier = nn.Sequential(
        	nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, NUM_CLASSES),
        )
        # same architecture of AlexNet's FC but adding a new densely connected branch with 2 output neurons (TO FINISH)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.dann_classifier = nn.Sequential(
        	nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, NUM_DOMAINS),
        )

    def update(self):
        self.domain_classifier[1].weight.data = self.classifier[1].weight.data
        self.domain_classifier[1].bias.data = self.classifier[1].bias.data

        self.domain_classifier[4].weight.data = self.classifier[4].weight.data
        self.domain_classifier[4].bias.data = self.classifier[4].bias.data
        

    def forward(self, x, alpha=None):
        features = self.features(x);
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        # If we pass alpha, we can assume we are training the discriminator
        if alpha is not None:
            # gradient reversal layer (backward gradients will be reversed)
            reverse_feature = ReverseLayerF.apply(features, alpha)
            discriminator_output = self.dann_classifier(features);
            return discriminator_output
        # If we don't pass alpha, we assume we are training with supervision
        else:
            # do something else
            class_outputs = self.classifier(features);
            return class_outputs

    def DANN_AlexNet(pretrained=False, progress=True, **kwargs):
        r"""AlexNet model architecture from the
        `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
    """
        model = RandomNetworkWithReverseGrad(**kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                                    progress = progress
                                                    )
            state_dict.popitem("classifier.6.weight")
            state_dict.popitem("classifier.6.bias")

            model.load_state_dict(state_dict, strict=False)
            model.update()
        return model
