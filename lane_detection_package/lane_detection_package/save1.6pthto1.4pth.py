# -*- coding: utf-8 -*-

# save 1.6 pth to pickle dict
import pickle as pkl
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

info_dict = torch.load('./checkpoints/torch1.6model/CP_epoch6.pth', map_location=device)

with open('pkl_1.4model.pth', 'wb') as f:
    pkl.dump(info_dict, f)


# in torch 1.4 load pickle model
# model = ....
# with open( 'pkl_1.4model.pth', 'rb') as f:
#     info_dict = pkl.load(f)

# model.load_state_dict(info_dict['model'])