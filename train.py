import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.mydataset import MyDataset
from model.networks import heatmap_Encoder, flow_Encoder, encoder_3D, Decoder
from model.losses import contra_loss, recons_loss

def main(data_dir = 'dataset'):
    pass

if __name__ == '__main__':
    main()