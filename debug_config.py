''' Configuration File.
'''

##
# Learning Loss for Active Learning
NUM_TRAIN = 50000 # N
NUM_VAL   = 50000 - NUM_TRAIN
BATCH     = 128 # B
SUBSET    = 10000 # M
INITIALQUERY = 1000
ADDENDUM  = 1000 # K

MARGIN = 1.0 # xi
WEIGHT = 1.0 # lambda
WEIGHT2 = 1.0 # TripletLoss Metric Loss for Loss module embeddings

TRIALS = 5
CYCLES = 10

EPOCH = 200
LR = 0.1
MILESTONES = [160]
EPOCHL = 120 # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model

MOMENTUM = 0.9
WDECAY = 5e-4


## parsing
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--aux1', type=str, default = "None")
parser.add_argument('--aux2', type=str, default = "None")
parser.add_argument('--aux3', type=str, default = "None")
parser.add_argument('--picked_plot', action='store_true', default = False)
parser.add_argument('--rule', type=str, default = "Random")
args = parser.parse_args()

if args.aux1 == 'MSE':
    LWDECAY = 5e-4
else:
    LWDECAY = WDECAY

''' CIFAR-10 | ResNet-18 | 93.6%
NUM_TRAIN = 50000 # N
NUM_VAL   = 50000 - NUM_TRAIN
BATCH     = 128 # B
SUBSET    = NUM_TRAIN # M
ADDENDUM  = NUM_TRAIN # K

MARGIN = 1.0 # xi
WEIGHT = 0.0 # lambda

TRIALS = 1
CYCLES = 1

EPOCH = 50
LR = 0.1
MILESTONES = [25, 35]
EPOCHL = 40

MOMENTUM = 0.9
WDECAY = 5e-4
'''
