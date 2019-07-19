################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# Generate training data via OpenFOAM
#
################

import os, math, uuid, sys, random
import numpy as np
import utils
from func import genMesh, runSim

samples           = 100       # no. of datasets to produce
freestream_angle  = math.pi / 8.  # -angle ... angle
freestream_length = 10.           # len * (1. ... factor)
freestream_length_factor = 10.    # length factor

airfoil_database  = "./object_database/"
output_dir        = "./train/"

seed = random.randint(0, 2**32 - 1)
np.random.seed(seed)
print("Seed: {}".format(seed))

def outputProcessing(basename, freestreamX, freestreamY, dataDir=output_dir, pfile='OpenFOAM/postProcessing/internalCloud/500/cloud_p.xy', ufile='OpenFOAM/postProcessing/internalCloud/500/cloud_U.xy', res=256, imageIndex=0):
    # output layout channels:
    # [0] freestream field X + boundary
    # [1] freestream field Y + boundary
    # [2] binary mask for boundary
    # [3] pressure output
    # [4] velocity X output
    # [5] velocity Y output
    npOutput = np.zeros((6, res, res))

    ar = np.loadtxt(pfile)
    curIndex = 0

    for y in range(res):
        for x in range(res):
            xf = (x / res - 0.5) * 2 + 0.5
            yf = (y / res - 0.5) * 2
            if abs(ar[curIndex][0] - xf)<1e-4 and abs(ar[curIndex][1] - yf)<1e-4:
                npOutput[3][x][y] = ar[curIndex][3]
                curIndex += 1
                # fill input as well
                npOutput[0][x][y] = freestreamX
                npOutput[1][x][y] = freestreamY
            else:
                npOutput[3][x][y] = 0
                # fill mask
                npOutput[2][x][y] = 1.0

    ar = np.loadtxt(ufile)
    curIndex = 0

    for y in range(res):
        for x in range(res):
            xf = (x / res - 0.5) * 2 + 0.5
            yf = (y / res - 0.5) * 2
            if abs(ar[curIndex][0] - xf)<1e-4 and abs(ar[curIndex][1] - yf)<1e-4:
                npOutput[4][x][y] = ar[curIndex][3]
                npOutput[5][x][y] = ar[curIndex][4]
                curIndex += 1
            else:
                npOutput[4][x][y] = 0
                npOutput[5][x][y] = 0

    utils.saveAsImage('data_pictures/pressure_%04d.png'%(imageIndex), npOutput[3])
    utils.saveAsImage('data_pictures/velX_%04d.png'  %(imageIndex), npOutput[4])
    utils.saveAsImage('data_pictures/velY_%04d.png'  %(imageIndex), npOutput[5])
    utils.saveAsImage('data_pictures/inputX_%04d.png'%(imageIndex), npOutput[0])
    utils.saveAsImage('data_pictures/inputY_%04d.png'%(imageIndex), npOutput[1])

    #fileName = dataDir + str(uuid.uuid4()) # randomized name
    fileName = dataDir + "%s_%d_%d" % (basename, int(freestreamX*100), int(freestreamY*100) )
    print("\tsaving in " + fileName + ".npz")
    np.savez_compressed(fileName, a=npOutput)


files = os.listdir(airfoil_database)
files.sort()
if len(files)==0:
	print("error - no airfoils found in %s" % airfoil_database)
	exit(1)

utils.makeDirs( ["./data_pictures", "./train", "./OpenFOAM/constant/polyMesh/sets", "./OpenFOAM/constant/polyMesh"] )


# main

fout = open('train.txt', 'wt')
for n in range(samples):
    print("Run {}:".format(n))
    print("Run {}:".format(n), file=fout)

    #fileNumber = np.random.randint(0, len(files))
    #basename = os.path.splitext( os.path.basename(files[fileNumber]) )[0]
    #print("\tusing {}".format(files[fileNumber]))
    #print("\tusing {}".format(files[fileNumber]), file=fout)

    basename = 'cylinder.dat'

    length = freestream_length * np.random.uniform(1.,freestream_length_factor)*2
    angle  = np.random.uniform(-freestream_angle, freestream_angle)
    AoA = angle * (8 / math.pi) * 22.5
    fsX =  math.cos(angle) * length
    fsY = -math.sin(angle) * length

    print("\tUsing len %5.3f angle %+5.3f " %( length,AoA ))
    print("\tUsing len %5.3f angle %+5.3f " %( length,AoA )  , file=fout)
    print("\tResulting freestream vel x,y: {},{}".format(fsX,fsY))
    print("\tResulting freestream vel x,y: {},{}".format(fsX,fsY), file=fout)

    os.chdir("./OpenFOAM/")
    #if genMesh("../" + object_database + files[fileNumber]) != 0:
        #print("\tmesh generation failed, aborting");
        #os.chdir("..")
        #continue

    runSim(fsX, fsY)
    os.chdir("..")

    outputProcessing(basename, fsX, fsY)
    print("\tdone")
fout.close()
