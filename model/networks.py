import torch
import torch.nn as nn
import torch.nn.functional as F

# 프레임 개수 고정(32개, uniform하게 33장 추출), mydataset에서 코드 추가 작성 필요, interpolation
# gpu에 올려서 샘플 동시에 몇 개까지 가능한지 확인(최소 5개는 되면 좋겠음)

class heatmap_Encoder(nn.Module):
    def __init__(self):
        super(heatmap_Encoder, self).__init__()
        self.h_seq = nn.Sequential(nn.Conv2d(17, 16, kernel_size=11),
                                    nn.LeakyReLU(0.2),
                                    nn.BatchNorm2d(16),
                                    # nn.MaxPool2d((2,2)),
                                    nn.Conv2d(16, 8, kernel_size=7),
                                    nn.LeakyReLU(0.2),
                                    nn.BatchNorm2d(8),
                                    nn.MaxPool2d((2,2)),
                                    nn.Conv2d(8, 3, kernel_size=3))

    def forward(self, h_feat):
        h_feat = self.h_seq(h_feat)

        return h_feat

class flow_Encoder(nn.Module):
    def __init__(self):
        super(flow_Encoder, self).__init__()
        self.f_seq = nn.Sequential(nn.Conv2d(34, 16, kernel_size=11),
                                    nn.LeakyReLU(0.2),
                                    nn.BatchNorm2d(16),
                                    # nn.MaxPool2d((2,2)),
                                    nn.Conv2d(16, 8, kernel_size=7),
                                    nn.LeakyReLU(0.2),
                                    nn.BatchNorm2d(8),
                                    nn.MaxPool2d((2,2)),
                                    nn.Conv2d(8, 3, kernel_size=3))

    def forward(self, f_feat):
        f_feat = self.f_seq(f_feat)

        return f_feat

class encoder_3D(nn.Module):
    def __init__(self):
        super(encoder_3D, self).__init__()
        self.e3D_seq = nn.Sequential(nn.Conv3d(6,8,kernel_size=11),
                                     nn.LeakyReLU(0.2),
                                     nn.BatchNorm3d(8),
                                     nn.Conv3d(8,12,kernel_size=9),
                                     nn.LeakyReLU(0.2),
                                     nn.BatchNorm3d(12),
                                     nn.Conv3d(12,16,kernel_size=7),
                                     nn.LeakyReLU(0.2),
                                     nn.BatchNorm3d(16),
                                     nn.Conv3d(16,24,kernel_size=5),
                                     nn.LeakyReLU(0.2),
                                     nn.BatchNorm3d(24),
                                     nn.Conv3d(24,32,kernel_size=3))

        self.lin_seq = nn.Sequential(nn.Linear(38000, 512),
                                     nn.Linear(512,128))
        # self.linear = nn.Linear(38000, 128)
        # self.lin_seq = nn.Sequential(nn.Linear(4920, 512),
        #                              nn.Linear(512,128))
        
    def forward(self, motion_feature):
        motion_feature = self.e3D_seq(motion_feature).reshape(32,-1)
        motion_feature = self.lin_seq(motion_feature)

        return motion_feature
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # self.linears = nn.ModuleList([nn.Linear(128, 2) for i in range(17)])
        self.linears_1 = nn.ModuleList([nn.Linear(128, 32) for i in range(17)])
        self.linears_2 = nn.ModuleList([nn.Linear(32, 2) for i in range(17)])

    def forward(self, x):
        x_list = []
        for i in range(len(x)):
            for j in range(17):
                # tmp_x = self.linears[j](x[i])
                tmp_x = self.linears_1[j](x[i])
                tmp_x = self.linears_2[j](tmp_x)
                x_list.append(tmp_x)

        return torch.stack(x_list, dim=0).reshape(32,17,2)