import torch
import torch.nn as nn
import os
from U_NET_BF_w8 import UNET 
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from gen_samples_final_test import create_phantom_bmodes_att_conv


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
    print(device)

    # ###################CLUSTER####################
    # #DATA DIRS
    sim_dir = '/mnt/nfs/isalazar/datasets/simulatedCystDataset/raw_0.0Att/'
    att_dir = '/mnt/nfs/isalazar/datasets/simulatedCystDataset/raw_0.5Att/'

    # # Get the list of all files and directories
    path = '/mnt/nfs/efernandez/datasets/dataRF/RF_test/'
    # path = '/TESIS/DATOS_1/rf_test/'
    dir_test = os.listdir(path)

    save_dir = '/mnt/nfs/efernandez/generated_samples/Nair/gen_test/'
    save_dir2 = '/mnt/nfs/efernandez/generated_samples/Nair/gen_att/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # model_dir='/mnt/nfs/efernandez/trained_models/DDPM_model/v6_TT_50epoch'
    # model_dir='/CODIGOS_TESIS/T2/trained_models/Unet_Nair/'
    # model_dir = '/mnt/nfs/efernandez/trained_models/UNet_Nair/' 
    # model_dir = '/CODIGOS_TESIS/T2/trained_models/Unet_Nair/'
    model_dir = '/mnt/nfs/efernandez/trained_models/UNet_Nair/'
    # model_dir='/CODIGOS_TESIS/T2/trained_models/UNet_difusiva'
    training_epochs = 100#10
    model = UNET(2,64,1).to(device)
    # print(model)
    model.load_state_dict(torch.load(f"{model_dir}/model_{training_epochs}.pth", map_location=device))

    depth_ini=30
    num_samples = 500

    for simu in dir_test:
        simu_name = simu[:-4]

    # for simu in range(1,num_samples+1):
    #     simu_name = "simu" + str(simu).zfill(5)
        # simu_name = simu[:-4]
        # simu_name = simu
        # bmode, grid_full=create_phantom_bmodes2("/CODIGOS_TESIS/T2/shape",h5name,simu_name, depth_ini, device, model,diffusion)
        #bmode, grid_full=create_phantom_bmodes_att_diff(sim_dir,simu_name, depth_ini, device, model,diffusion)
        bmode, grid_full=create_phantom_bmodes_att_conv(sim_dir,simu_name, depth_ini, device, model)
        np.save(save_dir+simu_name+".npy", bmode)  ##modelo ft 300

    for simu in range(1,num_samples+1):
        simu_name = "simu" + str(simu).zfill(5)
        bmode, grid_full=create_phantom_bmodes_att_conv(att_dir,simu_name, depth_ini, device, model)
        np.save(save_dir2+simu_name+".npy", bmode)  ##modelo ft 300
        

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()