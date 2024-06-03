import numpy as np
import torch
import torch.nn as nn

# motion_feature에서 Avg를 구해서 global motion(d, )로 만들어서 contra loss에 넣기

def contra_loss(origin, decoded):
    output = np.dot(origin, decoded)

    return output

# # MSE 사용
# def recons_loss(origin, target):
#     loss = nn.MSELoss()
#     output = loss(origin, target)

#     return output