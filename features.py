import torch
import torch.nn as nn
import torchvision.models as models
from config import DEVICE

# Implementation of perceptual loss for style transfer -- https://arxiv.org/pdf/1603.08155.pdf

def gram_matrix(x):
    (b, ch, h, w) = x.size()
    # flatten each feature map
    features = x.view(b, ch, h * w)
    features_t = features.transpose(1, 2)
    # take the dot product with the feature representation and its transpose
    gram = features.bmm(features_t)
    # normalize the result
    gram /= ch * h * w
    return gram

class VGG16Features(nn.Module):
    def __init__(self):
        super().__init__()

        # Import PyTorch's pretrained VGG16 model
        self.vgg16 = models.vgg16(pretrained=True).eval().to(DEVICE)

        # Get the slices of the layers we want to extract feature maps from
        self.relu1_1 = self.vgg16.features[0:4]
        self.relu2_2 = self.vgg16.features[4:9]
        self.relu3_3 = self.vgg16.features[9:16]
        self.relu4_3 = self.vgg16.features[16:23]

        # We don't want to modify the pre-trained vgg16 network, so freeze its parameters
        for param in self.vgg16.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Extract the feature maps from the images
        relu1_1 = self.relu1_1(x)
        relu2_2 = self.relu2_2(relu1_1)
        relu3_3 = self.relu3_3(relu2_2)
        relu4_3 = self.relu4_3(relu3_3)

        return (relu1_1, relu2_2, relu3_3, relu4_3)

if __name__ == '__main__':
    loss = VGG16Features()
    x = torch.randn((5, 3, 128, 128)).to(DEVICE)
    y = loss(x)
    print(x.shape, y[0].shape)
