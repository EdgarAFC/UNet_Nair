import torch
import torch.nn as nn
import os
from U_NET_BF import UNET 
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

class ONEPW_Dataset(Dataset):
    def __init__(self, data, onepw_img):
        '''
        data - train data path
        enh_img - train enhanced images path
        '''
        self.train_data = data
        self.train_onepw_img = onepw_img

        self.images = sorted(os.listdir(self.train_data))
        self.onepw_images = sorted(os.listdir(self.train_onepw_img))
  
    #regresar la longitud de la lista, cuantos elementos hay en el dataset
    def __len__(self):
        if self.onepw_images is not None:
          assert len(self.images) == len(self.onepw_images), 'not the same number of images ans enh_images'
        return len(self.images)

    def __getitem__(self, idx):
        rf_image_name = os.path.join(self.train_data, self.images[idx])
        rf_image = np.load(rf_image_name)
        rf_image = torch.Tensor(rf_image)
        rf_image = rf_image.permute(2, 0, 1)

        onepw_image_name = os.path.join(self.train_onepw_img, self.onepw_images[idx])
        onepw_img = np.load(onepw_image_name)
        onepw_img = torch.Tensor(onepw_img)
        onepw_img = onepw_img.unsqueeze(0)
        # new_min = -1
        # new_max = 1
        # onepw_img = onepw_img * (new_max - new_min) + new_min

        # enh_image_name = os.path.join(self.train_enh_img, self.enh_images[idx])
        # enh_img = np.load(enh_image_name)
        # enh_img = torch.Tensor(enh_img)
        # enh_img = enh_img.unsqueeze(0)
        # new_min = -1
        # new_max = 1
        # enh_img = enh_img * (new_max - new_min) + new_min

        return rf_image, onepw_img

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
    print(device)


    # # att_dir = '/CODIGOS_TESIS/T2/data_att0.5'
    # # shape_dir = '/CODIGOS_TESIS/T2/shape'
    # input_folder = '/TESIS/DATOS_1/rf_test'
    # # output_folder = '/TESIS/DATOS_TESIS2/onepw_test'

    # output_folder = '/TESIS/DATOS_1/enh_test'
    # Define the model and train with scheduler
    test_path = '/mnt/nfs/efernandez/datasets/dataRF/RF_test'
    test_enh_path = '/mnt/nfs/efernandez/datasets/dataENH/ENH_test'
    test_onepw_path= '/mnt/nfs/efernandez/datasets/dataONEPW/ONEPW_test'

    test_dataset = ONEPW_Dataset(test_path, test_enh_path)


    # save_dir = '/CODIGOS_TESIS/T2/generated_samples/DDPM_model/v6_TT_100steps_gen_att/'
    # save_dir = '/CODIGOS_TESIS/T2/trained_models/Udiffusive/v1_50epoch_gen_att/'
    # save_dir = '/CODIGOS_TESIS/T2/generated_samples/Unet_Nair/v1_70epoch/gen_test/'
    save_dir = '/mnt/nfs/efernandez/generated_samples/Nair/gen_test/'    # save_dir = '/CODIGOS_TESIS/T2/trained_models/Udiffusive/v1_300epoch_gen_att/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # model_dir='/mnt/nfs/efernandez/trained_models/DDPM_model/v6_TT_50epoch'
    # model_dir='/CODIGOS_TESIS/T2/trained_models/Unet_Nair/'
    model_dir = '/mnt/nfs/efernandez/trained_models/UNet_Nair/' 
    # model_dir='/CODIGOS_TESIS/T2/trained_models/UNet_difusiva'
    training_epochs = 100#10
    model = UNET(2,64,1).to(device)
    model.load_state_dict(torch.load(f"{model_dir}/model_{training_epochs}.pth", map_location=device))
    print("Num params: ", sum(p.numel() for p in model.parameters()))

    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    i = 0
    for x, y in dataloader:
        
        x_sample = x.to(device)
        y_sample = y.to(device)
        intermediate = []
        # for step in diffusion.p_sample_loop_progressive(model13A, y_sample.shape, x_sample, progress=True, clip_denoised=True):
        #     intermediate.append(step['sample'].cpu().detach())

        bmode=model(x_sample)
        np.save(save_dir+'simu',str(i),".npy", bmode)
        i = i +1
        # extent=[-20,20,50,0]
        # plt.figure(figsize=(9, 3))
        # # plt.subplot(1, 5, 1)
        # plt.imshow(bmode.detach().cpu().numpy().squeeze(), cmap="gray", extent=extent, origin="upper")
        # plt.colorbar()
        # plt.title('V7')
        # plt.show()

        # plt.subplot(1, 2, 1)
        # # print(y_sample[:,1,:].shape)
        # show_tensor_image(bmode.detach().cpu().numpy().squeeze())
        # plt.colorbar()
        # plt.title('deep')
        # plt.show()
        # show_reverse_process(intermediate[::10])

        # plt.figure(figsize=(9, 3))
        # plt.subplot(1, 2, 1)
        # show_tensor_image(intermediate[-1])
        # plt.colorbar()
        # plt.title('Diffusion')

        # plt.subplot(1, 2, 2)
        # # print(y_sample[:,1,:].shape)
        # plt.imshow(y_sample.detach().cpu().numpy().squeeze(), cmap="gray", extent=extent, origin="upper")
        # plt.colorbar()
        # plt.title('Objective')
        # plt.show()

if __name__ == '__main__':
    main()