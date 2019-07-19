################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# Compute errors for a test set and visualize. This script can loop over a range of models in 
# order to compute an averaged evaluation. 
#
################

import os,sys,random,math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import TurbDataset
from DfpNet import TurbNetG, weights_init
import utils
from utils import log

suffix = "" # customize loading & output if necessary
prefix = ""
if len(sys.argv)>1:
    prefix = sys.argv[1]
    print("Output prefix: {}".format(prefix))

expo = 5
res = 256
dataset = TurbDataset(None, mode=TurbDataset.TEST, dataDir="../data/make-datas/256/train/", dataDirTest="../data/make-datas/256/test/", res=res)
testLoader = DataLoader(dataset, batch_size=1, shuffle=False)

targets = torch.FloatTensor(1, 3, res, res)
targets = Variable(targets)
targets = targets.cuda()
inputs = torch.FloatTensor(1, 3, res, res)
inputs = Variable(inputs)
inputs = inputs.cuda()

targets_dn = torch.FloatTensor(1, 3, res, res)
targets_dn = Variable(targets_dn)
targets_dn = targets_dn.cuda()
outputs_dn = torch.FloatTensor(1, 3, res, res)
outputs_dn = Variable(outputs_dn)
outputs_dn = outputs_dn.cuda()

netG = TurbNetG(channelExponent=expo)
lf = "./" + prefix + "testout{}.txt".format(suffix) 
utils.makeDirs(["results_test"])

# loop over different trained models
avgLoss = 0.
losses = []
models = []
loss_p_list = []
loss_v_list = []
accum_list = []

for si in range(25):
    s = chr(96+si)
    if(si==0): 
        s = "" # check modelG, and modelG + char
    modelFn = "./" + prefix + "modelG{}{}".format(suffix,s)
    if not os.path.isfile(modelFn):
        continue

    models.append(modelFn)
    log(lf, "Loading " + modelFn )
    netG.load_state_dict( torch.load(modelFn) )
    log(lf, "Loaded " + modelFn )
    netG.cuda()

    criterionLoss = nn.L1Loss()
    #criterionLoss = nn.MSELoss()
    #criterionLoss = nn.SmoothL1Loss()
    criterionLoss.cuda()
    Lossval_accum = 0.0
    Lossval_dn_accum = 0.0
    lossPer_p_accum = 0
    lossPer_v_accum = 0
    lossPer_accum = 0

    netG.eval()

    for i, data in enumerate(testLoader, 0):
        inputs_cpu, targets_cpu = data
        targets_cpu, inputs_cpu = targets_cpu.float().cuda(), inputs_cpu.float().cuda()
        inputs.data.resize_as_(inputs_cpu).copy_(inputs_cpu)
        targets.data.resize_as_(targets_cpu).copy_(targets_cpu)

        outputs = netG(inputs)
        outputs_cpu = outputs.data.cpu().numpy()[0]
        targets_cpu = targets_cpu.cpu().numpy()[0]

        loss = criterionLoss(outputs, targets)
        Lossval_accum += loss.item()
        samples = 0

        # precentage loss by ratio of means which is same as the ratio of the sum
        lossPer_p = np.sum(np.abs(outputs_cpu[0] - targets_cpu[0]))/np.sum(np.abs(targets_cpu[0]))
        lossPer_v = ( np.sum(np.abs(outputs_cpu[1] - targets_cpu[1])) + np.sum(np.abs(outputs_cpu[2] - targets_cpu[2])) ) / ( np.sum(np.abs(targets_cpu[1])) + np.sum(np.abs(targets_cpu[2])) )
        lossPer = np.sum(np.abs(outputs_cpu - targets_cpu))/np.sum(np.abs(targets_cpu))
        lossPer_p_accum += lossPer_p.item()
        lossPer_v_accum += lossPer_v.item()
        lossPer_accum += lossPer.item()  
        #loss_p_list.append(np.sum(np.abs(outputs_cpu[0] - targets_cpu[0]))/np.sum(np.abs(targets_cpu[0])))
        p = lossPer_p.item()
        v = lossPer_v.item()
        if p >= 0.4:
            p = 0.
        loss_p_list.append(p)
        if v >= 0.1:
            v = 0.
        loss_v_list.append(v)
        accum_list.append(np.sum(np.abs(outputs_cpu - targets_cpu)))
        samples += i

        log(lf, "Test sample %d"% i )
        log(lf, "    pressure:  abs. difference, ratio: %f , %f " % (np.sum(np.abs(outputs_cpu[0] - targets_cpu[0])), lossPer_p.item()) )
        log(lf, "    velocity:  abs. difference, ratio: %f , %f " % (np.sum(np.abs(outputs_cpu[1] - targets_cpu[1])) + np.sum(np.abs(outputs_cpu[2] - targets_cpu[2])) , lossPer_v.item() ) )
        log(lf, "    aggregate: abs. difference, ratio: %f , %f " % (np.sum(np.abs(outputs_cpu    - targets_cpu   )), lossPer.item()) )

        # Calculate the norm
        input_ndarray = inputs_cpu.cpu().numpy()[0]
        v_norm = ( np.max(np.abs(input_ndarray[0,:,:]))**2 + np.max(np.abs(input_ndarray[1,:,:]))**2 )**0.5

        outputs_denormalized = dataset.denormalize(outputs_cpu, v_norm)
        targets_denormalized = dataset.denormalize(targets_cpu, v_norm)

        # denormalized error 
        outputs_denormalized_comp=np.array([outputs_denormalized])
        outputs_denormalized_comp=torch.from_numpy(outputs_denormalized_comp)
        targets_denormalized_comp=np.array([targets_denormalized])
        targets_denormalized_comp=torch.from_numpy(targets_denormalized_comp)

        targets_denormalized_comp, outputs_denormalized_comp = targets_denormalized_comp.float().cuda(), outputs_denormalized_comp.float().cuda()

        outputs_dn.data.resize_as_(outputs_denormalized_comp).copy_(outputs_denormalized_comp)
        targets_dn.data.resize_as_(targets_denormalized_comp).copy_(targets_denormalized_comp)

        loss_dn = criterionLoss(outputs_dn, targets_dn)
        Lossval_dn_accum += loss_dn.item()

        # write output image, note - this is currently overwritten for multiple models
        os.chdir("./results_test/")
        utils.imageOut("%04d"%(i), outputs_cpu, targets_cpu, normalize=False, saveMontage=True) # write normalized with error
        os.chdir("../")

    log(lf, "\n") 
    Lossval_accum     /= len(testLoader)
    lossPer_p_accum /= len(testLoader)
    lossPer_v_accum /= len(testLoader)
    lossPer_accum   /= len(testLoader)
    Lossval_dn_accum  /= len(testLoader)
    log(lf, "Loss percentage (p, v, combined): %f %%    %f %%    %f %% " % (lossPer_p_accum*100, lossPer_v_accum*100, lossPer_accum*100 ) )
    log(lf, "Loss error: %f" % (Lossval_accum) )
    log(lf, "Denormalized error: %f" % (Lossval_dn_accum) )
    log(lf, "\n") 

    avgLoss += lossPer_accum
    losses.append(lossPer_accum)

avgLoss /= len(losses)
lossStdErr = np.std(losses) / math.sqrt(len(losses))
log(lf, "Averaged relative error and std dev:   %f , %f " % (avgLoss,lossStdErr) )

aoa_128 = np.array([13.15, 20.38, -12.90, 2.592, -17.25, -19.32, -18.852, 
                    10.26, 13.34, 17.79, 2.746, -2.06, -0.07, -6.24, 
                   -10.291, 6.21, 10.23, 12.50, 4.059, -15.103, 9.683,
                    -17.56, 6.19, 6.43, 19.93, -2.27, 8.28, -11.10, 
                   -15.431, -22.10, -17.14, -7.75, 13.06, -12.28, -17.68, 
                    17.065, 7.31, 11.69, -18.844, 16.191, -6.67, -4.62, 
                     0.07, 15.12, -16.006, -9.132, -16.12, 10.066, 4.137, 
                    19.66, -21.34, -6.37, -16.33, 6.53, 1.267, -13.10, 
                    -20.10, -5.31, 18.72, 6.16, 7.27, -17.030, 5.343, 
                    18.402, -1.086, -6.41, -5.07, -14.802, 10.56, -19.86,
                    3.248, 21.10, 8.443, -3.39, -20.144, 19.52, -6.709, 
                    13.647, -11.82, 11.71, 16.92, 11.385, 6.34, -13.86, 
                    10.63, -3.852, 10.47, 16.16, 3.455, -18.868])

aoa_256 = np.array([0.029, -0.059,  0.247, -0.362,  0.282, -0.079, -0.178, 
               -0.100, -0.132, -0.135,  0.353, -0.333, -0.299, -0.227, 
               -0.020, -0.333,  0.222,  0.385, -0.094,  0.142,  0.301, 
                0.190,  0.369,  0.098,  0.030, -0.187, -0.229,  0.152, 
                0.290,  0.044, -0.076,  0.018,  0.193, -0.385,  0.328, 
               -0.355, -0.109, -0.133, -0.231, -0.208, -0.298,  0.173, 
               -0.160,  0.115, -0.142])
#x = aoa_128
x = aoa_256[:] * (8 / math.pi) * 22.5


plt.figure()

plt.scatter(x,loss_p_list)

plt.title('difference between CNN to CFD')
plt.xlabel('AoA')
plt.ylabel('loss_p')

#plt.show()
plt.savefig('loss_p')

plt.figure()

plt.scatter(x,loss_v_list)

plt.title('difference between CNN to CFD')
plt.xlabel('AoA')
plt.ylabel('loss_v')

#plt.show()
plt.savefig('loss_v')



