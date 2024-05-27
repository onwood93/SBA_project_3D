import torch
import torch.nn as nn
import torch.nn.functional as F

class heatmap_Encoder(nn.Module):
    def __init__(self):
        super(heatmap_Encoder, self).__init__()
        self.conv_h1 = nn.Conv2d(17, 16, kernel_size=11)
        self.conv_h2 = nn.Conv2d(16, 8, kernel_size=7)
        self.conv_h3 = nn.Conv2d(8, 3, kernel_size=5)

    def forward(self, h_feat):
        h_feat = F.leaky_relu(self.conv_h1(h_feat))
        h_feat = F.leaky_relu(self.conv_h2(h_feat))
        h_feat = F.leaky_relu(self.conv_h3(h_feat))

        return h_feat

class flow_Encoder(nn.Module):
    def __init__(self):
        super(flow_Encoder, self).__init__()
        self.conv_f1 = nn.Conv2d(34, 16, kernel_size=11)
        self.conv_f2 = nn.Conv2d(16, 8, kernel_size=7)
        self.conv_f3 = nn.Conv2d(8, 3, kernel_size=5)

    def forward(self, f_feat):
        f_feat = F.leaky_relu(self.conv_f1(f_feat))
        f_feat = F.leaky_relu(self.conv_f2(f_feat))
        f_feat = F.leaky_relu(self.conv_f3(f_feat))

        return f_feat
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        pass

    def forward(self, x):
        return None

class encoder_3D(nn.Module):
    def __init__(self):
        super(encoder_3D, self).__init__()
        self.conv3d_1 = nn.Conv3d(6, 8, kernel_size=11)
        self.conv3d_2 = nn.Conv3d(8, 12, kernel_size=9)
        self.conv3d_3 = nn.Conv3d(12, 16, kernel_size=7)
        self.conv3d_4 = nn.Conv3d(16, 24, kernel_size=5)
        self.conv3d_5 = nn.Conv3d(24, 32, kernel_size=3)

        # self.conv3d = nn.Conv3d(6, 32, kernel_size=3)

        self.conv1d_1 = nn.Conv1d(32, 32, kernel_size=3)

    def forward(self, motion_feature):
        motion_feature = F.leaky_relu(self.conv3d_1(motion_feature))
        motion_feature = F.leaky_relu(self.conv3d_2(motion_feature))
        motion_feature = F.leaky_relu(self.conv3d_3(motion_feature))
        motion_feature = F.leaky_relu(self.conv3d_4(motion_feature))
        motion_feature = F.leaky_relu(self.conv3d_5(motion_feature)).reshape(32,-1)
        # motion_feature = F.leaky_relu(self.conv3d(hf_feat))
        # motion_feature = F.leaky_relu(self.conv3d(hf_feat)).reshape(32,-1)

        motion_feature = self.conv1d_1(motion_feature)

        return motion_feature