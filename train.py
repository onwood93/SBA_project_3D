import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.mydataset import MyDataset
from model.networks import heatmap_Encoder, flow_Encoder, encoder_3D, Decoder
from model.losses import contra_loss, recons_loss

def main(data_dir = 'dataset'):
    train_data = MyDataset(data_dir)
    # trainloader = DataLoader(train_data, batch_size = 3)
    train_data_item = train_data[12000]
    heatmap = train_data_item['heatmap'][1:]
    optical_flow = train_data_item['optical_flow']

    hE = heatmap_Encoder()
    h_feat = hE(heatmap)

    fE = flow_Encoder()
    f_feat = fE(optical_flow)

    hf_feat = torch.cat((h_feat, f_feat), dim=1).transpose(1,0)[None,...]

    e3D = encoder_3D()
    motion_feature = e3D(hf_feat)

    decoder = Decoder()
    decoded_motion = decoder(motion_feature)
    recon_origin = train_data_item['origin'][1:]

    recon_loss = recons_loss()
    optimizer = optim.Adam(learning_rate=0.01)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = recon_loss(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

if __name__ == '__main__':
    main()