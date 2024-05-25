import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch import nn, optim
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset, random_split
#
import PIL
from PIL import Image
#
from datetime import datetime
from model_diff import UNETv13

import torch.nn.functional as func

import guided_diffusion_v3 as gd
from model_diff import UNETv13

TRAIN_PATH = '/mnt/nfs/efernandez/datasets/dataRF/RF_train'
TRAIN_ENH_PATH= '/mnt/nfs/efernandez/datasets/dataENH/ENH_train'
TRAIN_ONEPW_PATH= '/mnt/nfs/efernandez/datasets/dataONEPW/ONEPW_train'


# ###############################
# file_loss = open("/mnt/nfs/efernandez/projects/UNet_Nair/log_w1.txt", "w")
# file_loss.close()
# ################################
# def write_to_file(input): 
#     with open("/mnt/nfs/efernandez/projects/UNet_Nair/log_w1.txt", "a") as textfile: 
#         textfile.write(str(input) + "\n") 
#     textfile.close()


# TRAIN_PATH='/TESIS/DATOS_1/rf_train'
# TRAIN_ONEPW_PATH = '/TESIS/DATOS_TESIS2/onepw_train'


#creating our own Dataset
#esta clase va a heredar de la clase Dataset de Pytorch
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

class Conv_3_k(nn.Module):
  def __init__(self, channels_in, channels_out):
    super().__init__()
    self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=1, bias=False)
  def forward(self, x):
    return self.conv1(x)  
  
class Double_Conv(nn.Module):
  '''
  Double convolution block for U-Net
  '''
  def __init__(self, channels_in, channels_out):
    super().__init__()
    self.double_conv = nn.Sequential(
                       Conv_3_k(channels_in, channels_out),
                       nn.BatchNorm2d(channels_out),
                       nn.ReLU(inplace=True),

                       Conv_3_k(channels_out, channels_out),
                       nn.BatchNorm2d(channels_out),
                       nn.ReLU(inplace=True),
                        )
  def forward(self, x):
    return self.double_conv(x)
    
class Down_Conv(nn.Module):
  '''
  Down convolution part
  maxPool + double convolution
  '''
  def __init__(self, channels_in, channels_out):
    super().__init__()
    self.encoder = nn.Sequential(
                  nn.MaxPool2d(2,2), #size 2x2 and stride 2 para dividir la imagen en 2
                  Double_Conv(channels_in, channels_out)
                  )
  def forward(self, x):
    return self.encoder(x)

class Up_Conv(nn.Module):
  '''
  Up convolution part
  '''
  def __init__(self, channels_in, channels_out):
    super().__init__()
    self.upsample_layer = nn.Sequential(
                          # nn.Upsample(scale_factor = 2, mode ='bicubic'),
                          nn.ConvTranspose2d(channels_in/2, channels_in/2, kernel_size=2, stride=2),
                          # nn.Conv2d(channels_in, channels_in//2, kernel_size=1, stride=1)              
                          )
    self.decoder = Double_Conv(channels_in, channels_out)

  def forward(self, x1, x2):
    '''
    x1 - upsampled volumen
    x2 - volume from down sample to concatenate
    '''
    x1 = self.upsample_layer(x1)
    x  = torch.cat([x2, x1], dim=1)
    return self.decoder(x)
  
class UNET(nn.Module):
  '''
  UNET model
  '''
  def __init__(self, channels_in, channels, num_classes):
    super().__init__()
    self.first_conv = Double_Conv(channels_in, channels) #64, 800, 128
    self.down_conv1 = Down_Conv(channels, 2*channels) #128, 400, 64
    self.down_conv2 = Down_Conv(2*channels, 4*channels) #256, 200, 32
    self.down_conv3 = Down_Conv(4*channels, 8*channels) #512, 100, 16
    
    self.middle_conv = Down_Conv(8*channels, 8*channels) #512, 50, 8

    self.up_conv1 = Up_Conv(16*channels, 4*channels)
    self.up_conv2 = Up_Conv(8*channels, 2*channels)
    self.up_conv3 = Up_Conv(4*channels, 1*channels)
    self.up_conv4 = Up_Conv(2*channels, 1*channels)

    self.last_conv =  nn.Sequential(
                      nn.Conv2d(channels, num_classes, kernel_size = 1, stride=1),
                      nn.Sigmoid()
                      )
    
  def forward(self, x):
    x1= self.first_conv(x)
    x2 = self.down_conv1(x1)
    x3 = self.down_conv2(x2)
    x4 = self.down_conv3(x3)

    x5 = self.middle_conv(x4)

    u1_bf = self.up_conv1(x5, x4)
    u2_bf = self.up_conv2(u1_bf, x3)
    u3_bf = self.up_conv3(u2_bf, x2)
    u4_bf = self.up_conv4(u3_bf, x1)

    return self.last_conv(u4_bf)

# '''
# Checkpoint
# '''
# def save_checkpoint(state, filename="my_checkpoint.pth"):
#     print("=> Saving checkpoint")
#     torch.save(state, filename)

# def load_checkpoint(checkpoint,model,optimizer=None):
#     print("=> Loading checkpoint")
#     model.load_state_dict(checkpoint['state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer']) 

# def dice_coeff(pred, target):
#     smooth = 1e-8
#     intersection = (pred*target).sum()
#     denom = pred.sum() + target.sum()
#     dice = 2.0*(intersection/(denom + smooth))
#     return float(dice)

def main():

  device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
  print(device)

  save_dir = '/mnt/nfs/efernandez/trained_models/UNet_Nair/'
  # save_dir = '/mnt/nfs/efernandez/trained_models/UNet_difusiva/v1_300epoch'
  # save_dir = '/CODIGOS_TESIS/T2/trained_models/UNet_difusiva/v1_50epoch'
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)

  # Training hyperparameters
  batch_size = 16  # 4 for testing, 16 for training
  n_epoch = 150
  l_rate = 1e-5  # changing from 1e-5 to 1e-6, new lr 1e-7

  # Define the model and train with scheduler
  train_dataset = ONEPW_Dataset(TRAIN_PATH, TRAIN_ENH_PATH)
  train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)

  # nn_model = UNETv13(residual=True, attention_res=[], group_norm=True).to(device)
  nn_model = UNET(2,64,1).to(device)
  print("Num params: ", sum(p.numel() for p in nn_model.parameters() if p.requires_grad))
  optimizer_unet = torch.optim.Adam(nn_model.parameters(), lr=l_rate)

  # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer_unet, 
  #                                               max_lr = 1e-1,
  #                                               steps_per_epoch=len(train_loader),
  #                                               epochs=n_epoch, pct_start=0.43, div_factor=10, final_div_factor=1000,
  #                                               three_phase=True)
  
  # Testing Dataloader
  for i, (x, y) in enumerate(train_loader):
    print(i, x.shape,y.shape)
    if i==9: break

  trained_epochs = 100
  if trained_epochs > 0:
    nn_model.load_state_dict(torch.load(save_dir+f"/model_{trained_epochs}.pth", map_location=device))  # From last model
    # load_checkpoint(torch.load(save_dir+f"/model_{trained_epochs}.pth", map_location=device))
    loss_arr = np.load(save_dir+f"/loss_{trained_epochs}.npy").tolist()  # From last model
  else:
    loss_arr = []

  # Training

  nn_model.train()
  print(f' Epoch {trained_epochs}/{n_epoch}, {datetime.now()}')

  for ep in range(trained_epochs+1, n_epoch+1):
    for x, y in train_loader:
      optimizer_unet.zero_grad()
      x = x.to(device)
      y = y.to(device)

      score1 = nn_model(x)

      # print("score shape ",score1.shape)
      # print("y shape ",y.shape)
      # loss=nn.L1Loss()
      cost1 = func.l1_loss(score1, y)
      cost1.backward()

      # print("cost: ",cost1)
      
      loss_arr.append(cost1.item())
      optimizer_unet.step()

    # print(f' Epoch {ep:03}/{n_epoch}, loss: {loss_arr[-1]:.2f}, {datetime.now()}')
    # write_to_file("Epoch:")
    # write_to_file(ep)
    # write_to_file(datetime.now())

    if ep % 10 == 0 or ep == int(n_epoch) or ep == 1:
      # checkpoint = {'state_dict' : nn_model.state_dict(), 'optimizer': optimizer_unet.state_dict()}
      # save_checkpoint(checkpoint,save_dir+f"/model_{ep}.pth")
      torch.save(nn_model.state_dict(), save_dir+f"/model_{ep}.pth")
      np.save(save_dir+f"/loss_{ep}.npy", np.array(loss_arr))  


if __name__ == '__main__':
  main()