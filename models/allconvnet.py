import torch.nn as nn
import torch.nn.functional as F
import activations as acts
import torchvision.transforms as transforms
import torch

__all__ = ['AllConvNet']


class AllConvNetBase(nn.Module):

    def __init__(self, activation=acts.softplus, dropout=True, nc=3, num_classes=10, bn_affine=False):
        super(AllConvNetBase, self).__init__()
        self.act = activation
        print(self.act)
        self.dropout = dropout
        self.conv1 = nn.Conv2d(nc, 96, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(96, affine=bn_affine)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(96, affine=bn_affine)

        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(96, affine=bn_affine)

        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(192, affine=bn_affine)

        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(192, affine=bn_affine)

        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.bn6 = nn.BatchNorm2d(192, affine=bn_affine)

        self.conv7 = nn.Conv2d(192, 192, 3)#, padding=1)
        self.bn7 = nn.BatchNorm2d(192, affine=bn_affine)

        self.conv8 = nn.Conv2d(192, 192, 1)
        self.bn8 = nn.BatchNorm2d(192, affine=bn_affine)

        self.class_conv = nn.Conv2d(192, num_classes, 1)
        self.bn9 = nn.BatchNorm2d(num_classes, affine=bn_affine)

        self.global_avg = nn.AvgPool2d(6)

    def forward(self, x):
        if self.dropout:
            x = F.dropout(x, .2)
        bn1_out = self.bn1(self.conv1(x))
        conv1_out = self.act(bn1_out)
        bn2_out = self.bn2(self.conv2(conv1_out))
        conv2_out = self.act(bn2_out)
        bn3_out = self.bn3(self.conv3(conv2_out))
        conv3_out = self.act(bn3_out)
        if self.dropout:
            conv3_out = F.dropout(conv3_out, .5)
        bn4_out = self.bn4(self.conv4(conv3_out))
        conv4_out = self.act(bn4_out)
        bn5_out = self.bn5(self.conv5(conv4_out))
        conv5_out = self.act(bn5_out)
        bn6_out = self.bn6(self.conv6(conv5_out))
        conv6_out = self.act(bn6_out)
        if self.dropout:
            conv6_out = F.dropout(conv6_out, .5)
        bn7_out = self.bn7(self.conv7(conv6_out))
        conv7_out = self.act(bn7_out)
        bn8_out = self.bn8(self.conv8(conv7_out))
        conv8_out = self.act(bn8_out)

        bn9_out = self.bn9(self.class_conv(conv8_out))
        class_out = self.act(bn9_out)
        #pool_out = class_out.reshape(class_out.size(0), class_out.size(1), -1).mean(-1)
        pool_out = self.global_avg(class_out)
        pool_out = torch.squeeze(pool_out)
        return pool_out


class Base:
    base = AllConvNetBase
    args = list()
    kwargs = dict()
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


class AllConvNet(Base):
    pass