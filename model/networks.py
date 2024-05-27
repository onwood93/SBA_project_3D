import torch
import torch.nn as nn
import torch.nn.functional as F

class heatmap_Encoder(nn.Module):
    def __init__(self):
        super(heatmap_Encoder, self).__init__()
        self.conv_h1 = nn.Conv2d(17, 16, kernel_size=3, padding=1)
        self.conv_h2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.conv_h3 = nn.Conv2d(8, 3, kernel_size=3, padding=1)

    def forward(self, h_feat):
        h_feat = F.leaky_relu(self.conv_h1(h_feat))
        h_feat = F.leaky_relu(self.conv_h2(h_feat))
        h_feat = F.leaky_relu(self.conv_h3(h_feat))

        return h_feat

class flow_Encoder(nn.Module):
    def __init__(self):
        super(flow_Encoder, self).__init__()
        self.conv_f1 = nn.Conv2d(34, 16, kernel_size=3, padding=1)
        self.conv_f2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.conv_f3 = nn.Conv2d(8, 3, kernel_size=3, padding=1)

    def forward(self, f_feat):
        f_feat = F.leaky_relu(self.conv_f1(f_feat))
        f_feat = F.leaky_relu(self.conv_f2(f_feat))
        f_feat = F.leaky_relu(self.conv_f3(f_feat))

        return f_feat

class main_model(nn.Module):
    def __init__(self):
        super(main_model, self).__init__()
        self.conv3d_1 = nn.Conv3d(6, 8, kernel_size=3, padding=1)
        self.conv3d_2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.conv3d_3 = nn.Conv3d(16, 32, kernel_size=3, padding=1)

        # self.conv1d_1 = nn.Conv1d(input_channel, )

    def forward(self, h_feat, f_feat):
        hf_feat = torch.cat((h_feat, f_feat), dim=1).transpose(1,0)
        motion_feature = F.leaky_relu(self.conv3d_1(hf_feat))
        motion_feature = F.leaky_relu(self.conv3d_2(motion_feature))
        motion_feature = F.leaky_relu(self.conv3d_3(motion_feature)).reshape(32,-1)

        return motion_feature