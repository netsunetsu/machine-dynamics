################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# Main training script
#
################

import os, sys, random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from DfpNet import TurbNetG, weights_init
import dataset
import utils

######## Settings ########

# number of training iterations
iterations = 10000
# batch size
batch_size = 30
# learning rate, generator
lrG = 0.0006
# decay learning rate?
decayLr = True
# channel exponent to control network size
expo = 5
# data set config
prop=None # by default, use all from "../data/train"
#prop=[1000,0.75,0,0.25] # mix data from multiple directories
# save txt files with per epoch loss?
saveL1 = False

##########################

# NT_DEBUG , remove
iterations = 5000
res = 256
prefix = ""
if len(sys.argv)>1:
    prefix = sys.argv[1]
    print("Output prefix: {}".format(prefix))

autoIter   = False
dropout    = 0.1
doLoad     = ""
fout = open('train.txt', 'wt')

print("LR: {}".format(lrG))
print("LR decay: {}".format(decayLr))
print("Iterations: {}".format(iterations))
print("Dropout: {}".format(dropout))


print("LR: {}".format(lrG), file=fout)
print("LR decay: {}".format(decayLr), file=fout)
print("Iterations: {}".format(iterations), file=fout)
print("Dropout: {}".format(dropout), file=fout)
##########################

seed = random.randint(0, 2**32 - 1)
print("Random seed: {}".format(seed))
print("Random seed: {}".format(seed), file=fout)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
#torch.backends.cudnn.deterministic=True # warning, slower

# create pytorch data object with dfp dataset
data = dataset.TurbDataset(prop, shuffle=1, dataDir="../data/make-datas/256/train/", dataDirTest="../data/make-datas/256/test/", res=res)
trainLoader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
print("Training batches: {}".format(len(trainLoader)))
print("Training batches: {}".format(len(trainLoader)), file=fout)
dataValidation = dataset.ValiDataset(data)
valiLoader = DataLoader(dataValidation, batch_size=batch_size, shuffle=False, drop_last=True) 
print("Validation batches: {}".format(len(valiLoader)))
print("Validation batches: {}".format(len(valiLoader)), file=fout)

# setup training
epochs = int((iterations/len(trainLoader))*2 + 0.5)
netG = TurbNetG(channelExponent=expo, dropout=dropout)
print(netG) # print full net
print(netG, file=fout) # print train.txt
model_parameters = filter(lambda p: p.requires_grad, netG.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Initialized TurbNet with {} trainable params ".format(params))
print("Initialized TurbNet with {} trainable params ".format(params), file=fout)

netG.apply(weights_init)
if len(doLoad)>0:
    netG.load_state_dict(torch.load(doLoad))
    print("Loaded model "+doLoad)
netG.cuda()

#cost function(L1, MSE, SmoothL1)
#criterionLoss = nn.L1Loss()
criterionLoss = nn.MSELoss()
#criterionLoss = nn.SmoothL1Loss()
criterionLoss.cuda()



optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(0.5, 0.999), weight_decay=0.0)

targets = Variable(torch.FloatTensor(batch_size, 3, res, res))
inputs  = Variable(torch.FloatTensor(batch_size, 3, res, res))
targets = targets.cuda()
inputs  = inputs.cuda()

##########################
L_list = []
accum_list = []

for epoch in range(epochs):
    print("Starting epoch {} / {}".format((epoch+1),epochs))

    netG.train()
    Loss_accum = 0.0
    samples_accum = 0
    for i, traindata in enumerate(trainLoader, 0):
        inputs_cpu, targets_cpu = traindata
        current_batch_size = targets_cpu.size(0)

        targets_cpu, inputs_cpu = targets_cpu.float().cuda(), inputs_cpu.float().cuda()
        inputs.data.resize_as_(inputs_cpu).copy_(inputs_cpu)
        targets.data.resize_as_(targets_cpu).copy_(targets_cpu)

        # compute LR decay
        if decayLr:
            currLr = utils.computeLR(epoch, epochs, lrG*0.1, lrG)
            if currLr < lrG:
                for g in optimizerG.param_groups:
                    g['lr'] = currLr

        netG.zero_grad()
        gen_out = netG(inputs)

        loss = criterionLoss(gen_out, targets)
        loss.backward()

        optimizerG.step()

        lossviz = loss.item()
        Loss_accum += lossviz
       
        samples_accum += current_batch_size

        if i==len(trainLoader)-1:
            logline = "Epoch: {}, batch-idx: {}, Loss: {}\n".format(epoch, i, lossviz)
            print(logline)
            L_list.append(lossviz)
            accum_list.append(Loss_accum)
        
    # validation
    netG.eval()
    Lossval_accum = 0.0
 
    for i, validata in enumerate(valiLoader, 0):
        inputs_cpu, targets_cpu = validata
        current_batch_size = targets_cpu.size(0)

        targets_cpu, inputs_cpu = targets_cpu.float().cuda(), inputs_cpu.float().cuda()
        inputs.data.resize_as_(inputs_cpu).copy_(inputs_cpu)
        targets.data.resize_as_(targets_cpu).copy_(targets_cpu)

        outputs = netG(inputs)
        outputs_cpu = outputs.data.cpu().numpy()

        loss = criterionLoss(outputs, targets)
        Lossval_accum += loss.item()
     

        if i==0:
            input_ndarray = inputs_cpu.cpu().numpy()[0]
            v_norm = ( np.max(np.abs(input_ndarray[0,:,:]))**2 + np.max(np.abs(input_ndarray[1,:,:]))**2 )**0.5

            outputs_denormalized = data.denormalize(outputs_cpu[0], v_norm)
            targets_denormalized = data.denormalize(targets_cpu.cpu().numpy()[0], v_norm)
            utils.makeDirs(["results_train"])
            utils.imageOut("results_train/epoch{}_{}".format(epoch, i), outputs_denormalized, targets_denormalized, saveTargets=True)

    # data for graph plotting
    Loss_accum    /= len(trainLoader)
    Lossval_accum /= len(valiLoader)
 
    if saveL1:
        if epoch==0: 
            utils.resetLog(prefix + "L.txt"   )
            utils.resetLog(prefix + "Lval.txt")
        utils.log(prefix + "L.txt"   , "{} ".format(Loss_accum), False)
        utils.log(prefix + "Lval.txt", "{} ".format(Lossval_accum), False)
        

torch.save(netG.state_dict(), prefix + "modelG" )

plt.figure()
plt.plot(range(epochs), L_list, 'r-', label='train_loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.savefig('train.png')







