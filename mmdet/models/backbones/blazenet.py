
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from ..registry import BACKBONES


class BlazeBlock(nn.Module):
    def __init__(self, inp, oup1, oup2=None, stride=1, kernel_size=5):
        super(BlazeBlock, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_double_block = oup2 is not None
        self.use_pooling = self.stride != 1

        if self.use_double_block:
            self.channel_pad = oup2 - inp
        else:
            self.channel_pad = oup1 - inp

        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=stride, padding=padding, groups=inp, bias=True),
            nn.BatchNorm2d(inp),
            # pw-linear
            nn.Conv2d(inp, oup1, 1, 1, 0, bias=True),
            nn.BatchNorm2d(oup1),
        )
        self.act = nn.ReLU(inplace=True)

        if self.use_double_block:
            self.conv2 = nn.Sequential(
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup1, oup1, kernel_size=kernel_size, stride=1, padding=padding, groups=oup1, bias=True),
                nn.BatchNorm2d(oup1),
                # pw-linear
                nn.Conv2d(oup1, oup2, 1, 1, 0, bias=True),
                nn.BatchNorm2d(oup2),
            )

        if self.use_pooling:
            self.mp = nn.MaxPool2d(kernel_size=self.stride, stride=self.stride)

    def forward(self, x):
        h = self.conv1(x)
        if self.use_double_block:
            h = self.conv2(h)

        # skip connection
        if self.use_pooling:
            x = self.mp(x)
        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), 'constant', 0)
        return self.act(h + x)


def initialize(module):
    # original implementation is unknown
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data)
        nn.init.constant_(module.bias.data, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight.data, 1)
        nn.init.constant_(module.bias.data, 0)
    elif isinstance(module,nn.Linear):
        nn.init.kaiming_normal_(module.weight.data)
        nn.init.constant_(module.bias.data, 0)

@BACKBONES.register_module
class BlazeNet(nn.Module):
    """Constructs a BlazeFace model
    the original paper
    https://sites.google.com/view/perception-cv4arvr/blazeface
    """

    def __init__(self,
                 input_width=64,
                 input_height=128,
                 num_single=5,
                 num_double=6):
        super(BlazeNet, self).__init__()

        self.input_width = input_width
        self.input_height = input_height
        self.num_single = num_single
        self.num_double = num_double

        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )

        idx_single=0
        if num_single==5:
            self.features.add_module('single_block_{}'.format(idx_single),BlazeBlock(24, 24))
            idx_single+=1
            self.features.add_module('single_block_{}'.format(idx_single),BlazeBlock(24, 24))
            idx_single += 1
            self.features.add_module('single_block_{}'.format(idx_single),BlazeBlock(24, 48, stride=2))
            idx_single += 1
            self.features.add_module('single_block_{}'.format(idx_single),BlazeBlock(48, 48))
            idx_single += 1
            self.features.add_module('single_block_{}'.format(idx_single),BlazeBlock(48, 48))
            #idx_single += 1
            #self.features.add_module('single_block_{}'.format(idx_single), BlazeBlock(48, 48, stride=2))
        else:
            raise "Only support 5 single blaze blocks now."

        idx_double = 0
        if num_double==6:
            self.features.add_module('double_block_{}'.format(idx_single),BlazeBlock(48, 24, 96, stride=2))
            idx_single += 1
            self.features.add_module('double_block_{}'.format(idx_single),BlazeBlock(96, 24, 96))
            idx_single += 1
            self.features.add_module('double_block_{}'.format(idx_single),BlazeBlock(96, 24, 96))
            idx_single += 1
            self.features2 = nn.Sequential()


            self.features2.add_module('double_block_{}'.format(idx_single),BlazeBlock(96, 24, 96, stride=2))
            idx_single += 1
            self.features2.add_module('double_block_{}'.format(idx_single),BlazeBlock(96, 24, 96))
            idx_single += 1
            self.features2.add_module('double_block_{}'.format(idx_single),BlazeBlock(96, 24, 96))
        else:
            raise "Only support 6 double blaze blocks now."

        self.apply(initialize)

    def init_weights(self, pretrained):
        warnings.warn("deprecated", DeprecationWarning)

    def forward(self, x):
        outs = []
        y1 = self.features(x)
        y2 = self.features2(y1)

        outs.append(y1)
        outs.append(y2)

        return tuple(outs)

# if __name__ == '__main__':
#     blf = BlazeFace()
#     blf.eval()
#     blf.cuda()
#     batchsize=1
#     w=64
#     h=128
#     input = torch.randn(batchsize,3,w,h).cuda()
#     blf.forward(input)


