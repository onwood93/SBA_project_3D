import torch
import torch.nn as nn

# motion_feature에서 Avg를 구해서 global motion(d, )로 만들어서 contra loss에 넣기

def contra_loss(input):
    loss = nn.LogSoftmax(dim=1)
    output = -loss(input)

    return output

# # MSE 사용
# def recons_loss(origin, target):
#     loss = nn.MSELoss()
#     output = loss(origin, target)

#     return output