'''
Distributed model for Neural Distributed Image Compression with Cross-Attention Feature Alignment
'''
import torch
import math
from models.balle2017 import entropy_model, gdn
from models.distributed_models.attention_block import CrossAttention
from torch import nn

lower_bound = entropy_model.lower_bound_fn.apply


'''
This model is based on balle2017 model.
'''


class CADistributedAutoEncoder(nn.Module):
    def __init__(self, num_filters=192,  image_size=(256, 256), bound=0.11):
        super(CADistributedAutoEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1 = gdn.GDN(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2 = gdn.GDN(num_filters)
        self.conv3 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3 = gdn.GDN(num_filters)
        self.conv4 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)

        self.conv1_cor = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1_cor = gdn.GDN(num_filters)
        self.conv2_cor = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2_cor = gdn.GDN(num_filters)
        self.conv3_cor = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3_cor = gdn.GDN(num_filters)
        self.conv4_cor = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)

        self.conv1_w = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1_w = gdn.GDN(num_filters)
        self.conv2_w = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2_w = gdn.GDN(num_filters)
        self.conv3_w = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3_w = gdn.GDN(num_filters)
        self.conv4_w = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)

        self.entropy_bottleneck = entropy_model.EntropyBottleneck(num_filters, quantize=False)
        self.entropy_bottleneck_hx = entropy_model.EntropyBottleneck(num_filters)
        self.entropy_bottleneck_hy = entropy_model.EntropyBottleneck(num_filters, quantize=False)

        self.deconv1 = nn.ConvTranspose2d(2 * num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn1 = gdn.GDN(num_filters, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(2 * num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn2 = gdn.GDN(num_filters, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(2 * num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn3 = gdn.GDN(num_filters, inverse=True)
        self.deconv4 = nn.ConvTranspose2d(2 * num_filters, 3, 5, stride=2, padding=2, output_padding=1)

        self.deconv1_cor = nn.ConvTranspose2d(2 * num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn1_cor = gdn.GDN(num_filters, inverse=True)
        self.deconv2_cor = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn2_cor = gdn.GDN(num_filters, inverse=True)
        self.deconv3_cor = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn3_cor = gdn.GDN(num_filters, inverse=True)
        self.deconv4_cor = nn.ConvTranspose2d(num_filters, 3, 5, stride=2, padding=2, output_padding=1)

        self.ca1 = CrossAttention(input_size=(image_size[0] // 8, image_size[1] // 8), num_filters=num_filters,
                                  dim=256, num_patches=4, heads=8, dropout=0.1)
        self.ca2 = CrossAttention(input_size=(image_size[0] // 4, image_size[1] // 4), num_filters=num_filters,
                                  dim=256, num_patches=4, heads=8, dropout=0.1)
        self.ca3 = CrossAttention(input_size=(image_size[0] // 2, image_size[1] // 2), num_filters=num_filters,
                                  dim=256, num_patches=4, heads=8, dropout=0.1)

        self.bound = bound

    def encode(self, x):
        x = self.conv1(x)
        x = self.gdn1(x)
        x = self.conv2(x)
        x = self.gdn2(x)
        x = self.conv3(x)
        x = self.gdn3(x)
        x = self.conv4(x)
        return x

    def encode_cor(self, x):
        x = self.conv1_cor(x)
        x = self.gdn1_cor(x)
        x = self.conv2_cor(x)
        x = self.gdn2_cor(x)
        x = self.conv3_cor(x)
        x = self.gdn3_cor(x)
        x = self.conv4_cor(x)
        return x

    def encode_w(self, x):
        x = self.conv1_w(x)
        x = self.gdn1_w(x)
        x = self.conv2_w(x)
        x = self.gdn2_w(x)
        x = self.conv3_w(x)
        x = self.gdn3_w(x)
        x = self.conv4_w(x)
        return x

    def decode(self, x, y, w):
        x = torch.cat((x, w), 1)
        y = torch.cat((y, w), 1)

        x = self.deconv1(x)
        x = self.igdn1(x)
        y = self.deconv1_cor(y)
        y = self.igdn1_cor(y)
        # Applying cross attention between outputs of decoder's first layer
        x = self.ca1(x, y)

        x = self.deconv2(x)
        x = self.igdn2(x)
        y = self.deconv2_cor(y)
        y = self.igdn2_cor(y)
        # Applying cross attention between outputs of decoder's second layer
        x = self.ca2(x, y)

        x = self.deconv3(x)
        x = self.igdn3(x)
        y = self.deconv3_cor(y)
        y = self.igdn3_cor(y)
        # Applying cross attention between outputs of decoder's third layer
        x = self.ca3(x, y)

        x = self.deconv4(x)
        y = self.deconv4_cor(y)

        return x, y

    def forward(self, x, y):
        w = self.encode_w(y)  # p(w|y), i.e. the "common variable "
        if self.training:
            w = w + math.sqrt(0.001) * torch.randn_like(w)  # Adding small Gaussian noise improves the stability of training
        hx = self.encode(x)  # p(hx|x), i.e. the "private variable" of the primary image
        hy = self.encode_cor(y)  # p(hy|y), i.e. the "private variable" of the correlated image

        hx_tilde, x_likelihoods = self.entropy_bottleneck_hx(hx)
        _, y_likelihoods = self.entropy_bottleneck_hy(hy)
        _, w_likelihoods = self.entropy_bottleneck(w)

        x_tilde, y_tilde = self.decode(hx_tilde, hy, w)
        return x_tilde, y_tilde, x_likelihoods, y_likelihoods, w_likelihoods


if __name__ == '__main__':
    net = CADistributedAutoEncoder().cuda()
    print(net(torch.randn(1, 3, 256, 256).cuda(), torch.randn(1, 3, 256, 256).cuda())[0].shape)
