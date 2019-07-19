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

        #self.layer1 = nn.Sequential()
        #self.layer1.add_module('layer1_conv', nn.Conv2d(3, channels, 4, 2, 1, bias=True))

        self.layer0 = nn.Sequential()
        self.layer0.add_module('layer0_conv', nn.Conv2d(3, channels, 4, 2, 1, bias=True))
        self.layer1 = blockUNet(channels, channels, 'layer1', transposed=False, bn=False,  relu=False, dropout=dropout )
        #self.layer1b = blockUNet(channels, channels, 'layer1b', transposed=False, bn=True,  relu=False, dropout=dropout )

        self.layer2 = blockUNet(channels, channels*2, 'layer2', transposed=False, bn=False,  relu=False, dropout=dropout )
        self.layer2b= blockUNet(channels*2, channels*2, 'layer2b',transposed=False, bn=False,  relu=False, dropout=dropout )
        self.layer3 = blockUNet(channels*2, channels*4, 'layer3', transposed=False, bn=False,  relu=False, dropout=dropout )
        self.layer4 = blockUNet(channels*4, channels*8, 'layer4', transposed=False, bn=False,  relu=False, dropout=dropout , size=2,pad=0)
        self.layer5 = blockUNet(channels*8, channels*8, 'layer5', transposed=False, bn=False,  relu=False, dropout=dropout , size=2,pad=0)
        self.layer6 = blockUNet(channels*8, channels*8, 'layer6', transposed=False, bn=False, relu=False, dropout=dropout , size=2,pad=0)
#############################################################################################      
        
############################################################################################# 
        self.dlayer6 = blockUNet(channels*8, channels*8, 'dlayer6', transposed=True, bn=False, relu=True, dropout=dropout , size=2,pad=0)
        self.dlayer5 = blockUNet(channels*16,channels*8, 'dlayer5', transposed=True, bn=False, relu=True, dropout=dropout , size=2,pad=0)
        self.dlayer4 = blockUNet(channels*16,channels*4, 'dlayer4', transposed=True, bn=False, relu=True, dropout=dropout )
        self.dlayer3 = blockUNet(channels*8, channels*2, 'dlayer3', transposed=True, bn=False, relu=True, dropout=dropout )
        self.dlayer2b= blockUNet(channels*4, channels*2, 'dlayer2b',transposed=True, bn=False, relu=True, dropout=dropout )
        self.dlayer2 = blockUNet(channels*4, channels, 'dlayer2', transposed=True, bn=False, relu=True, dropout=dropout )

#outputpadding=0
        #self.dlayer1b = blockUNet(channels*2, channels, 'dlayer1b', transposed=True, bn=True, relu=True, dropout=dropout )
        self.dlayer1 = blockUNet(channels*2, channels, 'dlayer1', transposed=True, bn=False, relu=True, dropout=dropout )

        self.dlayer0 = nn.Sequential()
        self.dlayer0.add_module('dlayer0_relu', nn.ReLU(inplace=True))
        #self.dlayer0.add_module('dlayer0_prelu', nn.Sigmoid())
        self.dlayer0.add_module('dlayer0_tconv', nn.ConvTranspose2d(channels*2, 3, 4, 2, 1, bias=True))

        #self.dlayer1 = nn.Sequential()
        #self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=True))
        #self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose2d(channels*2, 3, 4, 2, 1, bias=True))

        #up-conv(pooling or unpooling)
        #self.ulayer6 = blockHFCN('ulayer6', size=2, st=2, pad=0)
        #self.ulayer5 = blockHFCN('ulayer5', size=2, st=2, pad=0)
        #self.ulayer4 = blockHFCN('ulayer4')
        #self.ulayer3 = blockHFCN('ulayer3')
        #self.ulayer2b = blockHFCN('ulayer2b')
        #self.ulayer2 = blockHFCN('ulayer2')
        #self.ulayer1 = blockHFCN('ulayer1')
        #self.ulayer0 = blockHFCN('ulayer0')

        #self.layer = nn.Sequential()
        #self.layer.add_module('layer_relu', nn.ReLu(inplace=True))
        #self.layer.add_module('layer0_conv', nn.Conv2d(3, channels, 4, 2, 1, bias=True))



    def forward(self, x):
        out0 = self.layer0(x)

        out1 = self.layer1(out0)
        #out1b = self.layer1b(out1)
        out2 = self.layer2(out1)
        out2b= self.layer2b(out2)
        out3 = self.layer3(out2b)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)

        dout6 = self.dlayer6(out6)
        dout6_out5 = torch.cat([dout6, out5], 1)
        dout5 = self.dlayer5(dout6_out5)
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

# discriminator (only for adversarial training, currently unused)
class TurbNetD(nn.Module):
    def __init__(self, in_channels1, in_channels2,ch=64):
        super(TurbNetD, self).__init__()

        self.c0 = nn.Conv2d(in_channels1 + in_channels2, ch, 4, stride=2, padding=2)
        self.c1 = nn.Conv2d(ch  , ch*2, 4, stride=2, padding=2)
        self.c2 = nn.Conv2d(ch*2, ch*4, 4, stride=2, padding=2)
        self.c3 = nn.Conv2d(ch*4, ch*8, 4, stride=2, padding=2)
        self.c4 = nn.Conv2d(ch*8, 1   , 4, stride=2, padding=2)

        self.bnc1 = nn.BatchNorm2d(ch*2)
        self.bnc2 = nn.BatchNorm2d(ch*4)
        self.bnc3 = nn.BatchNorm2d(ch*8)        

    def forward(self, x1, x2):
        h = self.c0(torch.cat((x1, x2),1))
        h = self.bnc1(self.c1(F.leaky_relu(h, negative_slope=0.2)))
        h = self.bnc2(self.c2(F.leaky_relu(h, negative_slope=0.2)))
        h = self.bnc3(self.c3(F.leaky_relu(h, negative_slope=0.2)))
        h = self.c4(F.leaky_relu(h, negative_slope=0.2))
        h = F.sigmoid(h) 
        return h



#Nested Unet
class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super(VGGBlock, self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        block.add_module('%s_dropout' % name, nn.Dropout2d( dropout, inplace=True))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_func(out)

        return out
class NestedUNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(args.input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.args.deepsupervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output
#############################################################################################      
        
############################################################################################# 
#Nested Unet
def BUNblock(in_c, out_c, name, transposed=False, size=4, st=2, pad=1, dropout=0.):
    block = nn.Sequential()
    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, kernel_size=size, stride=st, padding=pad, bias=True))
    else:
        block.add_module('%s_tconv' % name, nn.Conv2d(in_c, out_c, kernel_size=(size-1), stride=1, padding=pad, bias=True))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout>0.:
        block.add_module('%s_dropout' % name, nn.Dropout2d( dropout, inplace=True))

    return block

def SKCdence(in_c, middle_channels, out_c, name, act_func=nn.ReLU(inplace=True)):
    block = nn.Sequential()
    self.act_func = act_func
    self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
    self.bn1 = nn.BatchNorm2d(middle_channels)
    self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
    self.bn2 = nn.BatchNorm2d(out_channels)
    block.add_module('%s_dropout' % name, nn.Dropout2d( dropout, inplace=True))

    return out

class NestedUNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        nb_filter = [32, 64, 128, 256, 512]

        self.seq = nn.Sequential()
        self.relu = nn.ReLU(inplace=True)
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.up = nn.Upsample(scale_factor=2)


        #down_sampling
        self.layer0_0 = nn.Sequential()
        self.layer0_0.add_module('block0_0_conv', nn.Conv2d(3, nb_filter[0], 4, 2, 1, bias=True))

        self.layer1_0 = BlockUNet(nb_filter[0], nb_filter[1], 'block1_0', transposed=False, bn=True,  relu=False, dropout=dropout)
        self.layer2_0 = BlockUNet(nb_filter[1], nb_filter[1], 'block2_0', transposed=False, bn=True,  relu=False, dropout=dropout)
        self.layer3_0 = BlockUNet(nb_filter[1], nb_filter[2], 'block3_0', transposed=False, bn=True,  relu=False, dropout=dropout)
        self.layer4_0 = BlockUNet(nb_filter[2], nb_filter[3], 'block4_0', transposed=False, bn=True,  relu=False, dropout=dropout , size=2,pad=0)
        self.layer5_0 = BlockUNet(nb_filter[3], nb_filter[3], 'block5_0', transposed=False, bn=True,  relu=False, dropout=dropout , size=2,pad=0)
        self.layer6_0 = BlockUNet(nb_filter[3], nb_filter[3], 'block6_0', transposed=False, bn=True,  relu=False, dropout=dropout , size=2,pad=0)

        #up_sampling
        #self.dlayer6_0d = BlockUNet(nb_filter[3], nb_filter[3], 'block6_0d', transposed=True, bn=True, relu=True, dropout=dropout , size=2,pad=0)
        self.dlayer5_1 = BlockUNet(nb_filter[3]*3, nb_filter[3], 'block5_1', transposed=True, bn=True, relu=True, dropout=dropout , size=2,pad=0)
        self.dlayer4_2 = BlockUNet(nb_filter[3]*4, nb_filter[2], 'block4_2', transposed=True, bn=True, relu=True, dropout=dropout)
        self.dlayer3_3 = BlockUNet(nb_filter[2]*5, nb_filter[1], 'block3_3', transposed=True, bn=True, relu=True, dropout=dropout )
        self.dlayer2_4 = BlockUNet(nb_filter[1]*6, nb_filter[1], 'block2_4', transposed=True, bn=True, relu=True, dropout=dropout )
        self.conv1_5 = BlockUNet(nb_filter[1]*7, nb_filter[0], 'block1_5', transposed=True, bn=True, relu=True, dropout=dropout )

        self.dlayer0_6 = nn.Sequential()
        self.dlayer0_6.add_module('block0_6_relu', nn.ReLU(inplace=True))
        #self.dlayer0.add_module('dlayer0_prelu', nn.Sigmoid())
        self.dlayer0.add_module('block0_6_tconv', nn.ConvTranspose2d(nb_filter[1]*8, 3, 4, 2, 1, bias=True))

        #skip-connection
        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[1], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv3_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv4_1 = VGGBlock(nb_filter[3]+nb_filter[3], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[1], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv3_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])
        
        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[1], nb_filter[1], nb_filter[1])
        self.conv2_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])
        
        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_4 = VGGBlock(nb_filter[1]*4+nb_filter[1], nb_filter[1], nb_filter[1])
        
        self.conv0_5 = VGGBlock(nb_filter[0]*5+nb_filter[1], nb_filter[0], nb_filter[0])
        
        if self.args.deepsupervision:
            self.final1 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.leaky(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.leaky(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.leaky(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.leaky(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        x5_0 = self.conv5_0(self.leaky(x4_0))
        x4_1 = self.conv4_1(torch.cat([x4_0, self.up(x5_0)], 1))
        x3_2 = self.conv3_2(torch.cat([x3_0, x3_1, self.up(x4_1)], 1))
        x2_3 = self.conv2_3(torch.cat([x2_0, x2_1, x2_2, self.up(x3_2)], 1))
        x1_4 = self.conv1_4(torch.cat([x1_0, x1_1, x1_2, x1_3, self.up(x2_3)], 1))
        x0_5 = self.conv0_5(torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, self.up(x1_4)], 1))

        x6_0 = self.conv6_0(self.leaky(x5_0))
        x5_1 = self.conv5_1(torch.cat([x5_0, self.up(x6_0)], 1))
        x4_2 = self.conv4_2(torch.cat([x4_0, x4_1, self.up(x5_1)], 1))
        x3_3 = self.conv3_3(torch.cat([x3_0, x3_1, x3_2, self.up(x4_2)], 1))
        x2_4 = self.conv2_4(torch.cat([x2_0, x2_1, x2_2, x2_3, self.up(x3_3)], 1))
        x1_5 = self.conv1_5(torch.cat([x1_0, x1_1, x1_2, x1_3, x1_4, self.up(x2_4)], 1))
        x0_6 = self.conv0_6(torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, x0_5, self.up(x1_5)], 1))

        if self.args.deepsupervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            output5 = self.final5(x0_5)
            output6 = self.final6(x0_6)
            return [output1, output2, output3, output4, output5, output6]

        else:
            output = self.final(x0_6)
            return output




