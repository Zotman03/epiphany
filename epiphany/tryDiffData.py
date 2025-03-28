from utils import *
from data_loader_10kb import *
from model_10kb import *
from dataset import HiCDiffusionDataset
test_chroms = ['chr3', 'chr11', 'chr17']
train_chroms = ['chr1', 'chr2', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22']

TRAIN_SEQ_LENGTH = 200 
TEST_SEQ_LENGTH = 200 
train_set = HiCDiffusionDataset(seq_length=TRAIN_SEQ_LENGTH, chroms=train_chroms) 
test_set = HiCDiffusionDataset(seq_length=TEST_SEQ_LENGTH, chroms=test_chroms)

train_set_x = train_set[0] # training data from h5
train_set_y_noisy = train_set[1] # training data from pickle with noise
train_set_y_target = train_set[2] # real training data from pickle

test_set_x = train_set[0] # testing data from h5
test_set_y_noisy = train_set[1] # testing data from pickle with noise
test_set_y_target = train_set[2] # real testing data from pickle

print(len(train_set))
print(test_set)
