from utils.mydataset import MyDataset
from model.networks import Encoder, Decoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm

writer = SummaryWriter()

def main():
    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    # device = 'cpu'
    #model, batch, label 부분에 to device 붙이기
    data_dir = 'adjusted_dataset'
    batch = 12
    # start_time = time.time()
    dataset = MyDataset(data_dir, 1)
    train_loader = DataLoader(dataset, batch, shuffle=False, num_workers=8)
    # end_time = time.time()
    # print(end_time-start_time)
    # train_len = len(train_loader)

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    mse = nn.MSELoss()
    cE = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam([{'params':encoder.parameters(),'lr':0.001},
    #                             {'params':decoder.parameters(),'lr':0.001}])
    optimizer = torch.optim.Adam([
        *encoder.parameters(),
        *decoder.parameters()
    ], lr=0.001)
    
    for epoch in range(100):  # loop over the dataset multiple times

        # running_loss = 0.0
        total_loss = 0.0
        total_mse_loss = 0.0
        total_cE_loss = 0.0

        for i, train_item in enumerate(tqdm(train_loader, 0)):
            # get the inputs; data is a list of [inputs, labels]
            print('speed_test')
            anchor = train_item['anchor'].to(device)
            h_input = train_item['anchor_heatmap'].to(device)
            f_input = train_item['anchor_flow'].to(device)
            class_num = train_item['class'].to(device)
            print('Load train data to GPU')
            # print(anchor.size())
            # print(h_input.size())
            # print(f_input.size())
            # print(class_num, class_num.size())

            motion_feature = encoder(h_input,f_input)
            mean_motion_feature = motion_feature.mean(dim=1)
            decoded = decoder(motion_feature)
            print('Model foward finish')

            optimizer.zero_grad()

            mse_loss = mse(anchor, decoded)
            cE_loss = cE(mean_motion_feature, class_num)
            loss = mse_loss + cE_loss
            loss.backward()
            optimizer.step()

            total_mse_loss += mse_loss
            total_cE_loss += cE_loss
            total_loss += loss
            
            # print statistics
            # running_loss += loss.item()
            # if i % 100 == 99:    # print every 100 mini-batches
            #     # print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            #     # running_loss = 0.0

            #     writer.add_scalar("Loss/train_total_loss", total_loss/100 , i)
            #     writer.add_scalar("Loss/train_mse_loss", total_mse_loss/100, i)
            #     writer.add_scalar("Loss/train_cE_loss", total_cE_loss/100, i)

            #     total_loss = 0.0
            #     total_mse_loss = 0.0
            #     total_cE_loss = 0.0


        print('end of epoch', epoch)
        writer.add_scalar("total_Loss/train", total_loss/len(train_loader), epoch)
        writer.add_scalar("total_mse_Loss/train", total_mse_loss/len(train_loader), epoch)
        writer.add_scalar("total_cE_Loss/train", total_cE_loss/len(train_loader), epoch)

    torch.save(encoder,'/data/onwood/sba_project_3D/model.pt')

if __name__ == '__main__':
    main()