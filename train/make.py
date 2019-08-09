################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# CNN setup and data normalization
#
################

import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

#Unet
def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, size=4, st=2, pad=1, dropout=0.):
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
        #block.add_module('%s_prelu' % name, nn.PReLU())
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
        #block.add_module('%s_prelu' % name, nn.PReLU())
    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, kernel_size=size, stride=st, padding=pad, bias=True))
    else:
        block.add_module('%s_upsam' % name, nn.Upsample(scale_factor=2))
        block.add_module('%s_tconv' % name, nn.Conv2d(in_c, out_c, kernel_size=(size-1), stride=1, padding=pad, bias=True))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout>0.:
        block.add_module('%s_dropout' % name, nn.Dropout2d( dropout, inplace=True))

    return block
    
# generator model
class TurbNetG(nn.Module):
    def __init__(self, channelExponent=6, dropout=0.):
        super(TurbNetG, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.up = nn.Upsample(scale_factor=2))
        nb_filter = [channels, channels*2, channels*4, channels*8]
        
        
        self.layer0_0 = nn.Sequential()
        self.layer0_0.add_module('layer1_0_conv', nn.Conv2d(3, nb_filter[0], 4, 2, 1, bias=True))
        self.layer1_0 = blockUNet(nb_filter[0], nb_filter[1], 'layer1_0', transposed=False, bn=True,  relu=False, dropout=dropout )
        self.layer2_0 = blockUNet(nb_filter[1], nb_filter[1], 'layer2_0',transposed=False, bn=True,  relu=False, dropout=dropout )
        self.layer3_0 = blockUNet(nb_filter[1], nb_filter[2], 'layer3_0', transposed=False, bn=True,  relu=False, dropout=dropout )
        self.layer4_0 = blockUNet(nb_filter[2], nb_filter[3], 'layer4_0', transposed=False, bn=True,  relu=False, dropout=dropout , size=2,pad=0)
        self.layer5_0 = blockUNet(nb_filter[3], nb_filter[3], 'layer5_0', transposed=False, bn=False,  relu=False, dropout=dropout , size=2,pad=0)
        self.layer6_0 = blockUNet(nb_filter[3], nb_filter[3], 'layer6_0', transposed=False, bn=False, relu=False, dropout=dropout , size=2,pad=0)

        self.layer0_1 = blockUNet(nb_filter[0]+nn_filter[1], nb_filter[0], 'layer0_1', transposed=True, bn=True,  relu=True, dropout=dropout )
        self.layer1_1 = blockUNet(nb_filter[1]+nn_filter[1], nb_filter[1], 'layer1_1', transposed=True, bn=True,  relu=True, dropout=dropout )
        self.layer2_1 = blockUNet(nb_filter[1]+nb_filter[2], nb_filter[1], 'layer2_1', transposed=True, bn=True,  relu=True, dropout=dropout )
        self.layer3_1 = blockUNet(nb_filter[2]+nb_filter[3], nb_filter[2], 'layer3_1', transposed=True, bn=True,  relu=True, dropout=dropout )
        self.layer4_1 = blockUNet(nb_filter[3]+nb_filter[3], nb_filter[3], 'layer4_1', transposed=True, bn=True,  relu=True, dropout=dropout )
        self.layer5_1 = blockUNet(nb_filter[3]+nb_filter[3], nb_filter[3], 'layer5_1', transposed=True, bn=False, relu=True, dropout=dropout , size=2,pad=0)
        self.layer6_1 = blockUNet(nb_filter[3], nb_filter[3], 'layer6_1', transposed=True, bn=False, relu=True, dropout=dropout , size=2,pad=0)

        self.layer0_1 = blockUNet(nb_filter[0], nb_filter[1], 'layer1_0', transposed=False, bn=True,  relu=False, dropout=dropout )
        self.layer1_2 = blockUNet(nb_filter[0]*2+nb_filter[1], nb_filter[0], 'layer1_2', transposed=True, bn=True,  relu=True, dropout=dropout )
        self.layer2_2 = blockUNet(nb_filter[1]*2+nb_filter[1], nb_filter[1], 'layer2_2', transposed=True, bn=True,  relu=True, dropout=dropout )
        self.layer3_2 = blockUNet(nb_filter[1]*2+nb_filter[2], nb_filter[1], 'layer3_2', transposed=True, bn=True,  relu=True, dropout=dropout )
        self.layer4_2 = blockUNet(nb_filter[2]*2+nb_filter[3], nb_filter[2], 'layer4_1', transposed=True, bn=True,  relu=True, dropout=dropout )
        self.layer5_2 = blockUNet(nb_filter[3]*3, nb_filter[2], 'layer5_2', transposed=True, bn=False, relu=True, dropout=dropout , size=2,pad=0)

        self.layer1_3 = blockUNet(nb_filter[0]*3+nb_filter[1], nb_filter[0], 'layer1_3', transposed=True, bn=True,  relu=True, dropout=dropout )
        self.layer2_3 = blockUNet(nb_filter[1]*3+nb_filter[1], nb_filter[1], 'layer2_3', transposed=True, bn=True,  relu=True, dropout=dropout )
        self.layer3_3 = blockUNet(nb_filter[1]*3+nb_filter[2], nb_filter[1], 'layer3_3', transposed=True, bn=True,  relu=True, dropout=dropout )
        self.layer4_3 = blockUNet(nb_filter[2]*4, nb_filter[1], 'layer4_3', transposed=True, bn=True,  relu=True, dropout=dropout)
        
        self.layer1_4 = blockUNet(nb_filter[0]*4+nb_filter[1], nb_filter[0], 'layer1_4', transposed=True, bn=True,  relu=True, dropout=dropout )
        self.layer2_4 = blockUNet(nb_filter[1]*4+nb_filter[1], nb_filter[1], 'layer2_4', transposed=True, bn=True,  relu=True, dropout=dropout )
        self.layer3_4 = blockUNet(nb_filter[1]*5, nb_filter[1], 'layer3_4', transposed=True, bn=True,  relu=True, dropout=dropout )
        
        self.layer1_5 = blockUNet(nb_filter[0]*5+nb_filter[1], nb_filter[0], 'layer1_5', transposed=True, bn=True,  relu=True, dropout=dropout )
        self.layer2_5 = blockUNet(nb_filter[1]*6, nb_filter[0], 'layer2_5', transposed=True, bn=True,  relu=True, dropout=dropout )

        #output
        self.layer1_6 = nn.Sequential()
        self.layer1_6.add_module('layer1_6_relu', nn.ReLU(inplace=True))
        self.layer1_6.add_module('layer1_6_tconv', nn.ConvTranspose2d(nb_filter[0]*7, 3, 4, 2, 1, bias=True))

     

    def forward(self, x):
       
        #input
        out0_0 = self.layer0_0(x)

        out1_0 = self.layer1_0(out0_0)
        out2_0 = self.layer2_0(out1_0)
        out3_0 = self.layer3_0(out2_0)
        out4_0 = self.layer4_0(out3_0)
        out5_0 = self.layer5_0(out4_0)
        out6_0 = self.layer6_0(out5_0)

        out1_1 = self.layer1_1(torch.cat([out1_0, out5], 1))
        out2_0 = self.layer2_0(out1_0)
        out3_0 = self.layer3_0(out2_0)
        out4_0 = self.layer4_0(out3_0)
        out5_0 = self.layer5_0(out4_0)
        out6_0 = self.layer6_0(out5_0)

        out6 = self.dlayer6(out6)
        out6_out5 = torch.cat([dout6, out5], 1)
        out5 = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2b = torch.cat([dout3, out2b], 1)
        dout2b = self.dlayer2b(dout3_out2b)
        dout2b_out2 = torch.cat([dout2b, out2], 1)
        dout2 = self.dlayer2(dout2b_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        #dout1b = self.dlayer1(dout2_out1b)
        #dout1b_out1 = torch.cat([dout1b, out1], 1)
        dout1 = self.dlayer1(dout2_out1)
        dout1_out0 = torch.cat([dout1, out0], 1)
        dout0 = self.dlayer0(dout1_out0)
        return dout0




