import os
import  torch
import numpy as np
from matplotlib import pyplot as plt
import functools
import torch.nn as nn
import torch
import pickle
from U_NET_BF_w8 import UNET2
import copy

from model_diff import UNETv13

from gen_samples import LoadData_nair2020, downsample_channel_data

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

class Wang2020UnetGenerator(nn.Module):
    """Create a Unet-based generator"""
    def __init__(self, input_nc, output_nc, num_downs=6, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        """
        super(Wang2020UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        return self.model(input)



def load_gen_model(model_dir=None, epoch=None, num_downs=8, norm_layer=nn.BatchNorm2d, device=None):
    # Instantiate the model architecture
    # generator = Wang2020UnetGenerator(input_nc=3, #channel data (2) + latent space z (1)
                                    #   output_nc=1,
                                    #   num_downs=num_downs,
                                    #   ngf=64,
                                    #   norm_layer=norm_layer,
                                    #   use_dropout=False).to(device)

    # generator = UNET2(3,64,1).to(device)

    generator =  UNETv13(in_channels=3,out_channels=1,residual=True, attention_res=[], group_norm=True).to(device)

    if isinstance(epoch, int) and epoch == -1:
        gen_history = {"train_loss": [], "val_loss": []}
        genL1_history = {"train_loss": [], "val_loss": []}
        disc_history = {"train_loss": [], "val_loss": []}
        print("Model initialized with random weights")
    else:
        if isinstance(epoch, int):
            generator_filename = os.path.join(model_dir, 'model_gen_%.4d.t7' % epoch)
            discriminator_filename = os.path.join(model_dir, 'model_disc_%.4d.t7' % epoch)
            gen_history_filename = os.path.join(model_dir, 'history_gen_%.4d.pkl' % epoch)
            genL1_history_filename = os.path.join(model_dir, 'history_genL1_%.4d.pkl' % epoch)
            disc_history_filename = os.path.join(model_dir, 'history_disc_%.4d.pkl' % epoch)
            print(f"Loading models {epoch}...")
        elif isinstance(epoch, str):
            generator_filename = os.path.join(model_dir, 'model_gen_%s.t7' % epoch)
            print(generator_filename)

            discriminator_filename = os.path.join(model_dir, 'model_disc_%s.t7' % epoch)
            print(discriminator_filename)

            gen_history_filename = os.path.join(model_dir, 'history_gen_%s.pkl' % epoch)
            print(gen_history_filename)

            genL1_history_filename = os.path.join(model_dir, 'history_genL1_%s.pkl' % epoch)
            print(genL1_history_filename)

            disc_history_filename = os.path.join(model_dir, 'history_disc_%s.pkl' % epoch)
            print(disc_history_filename)

            print(f"Loading models {epoch}...")
        else:
            raise ValueError("Invalid epoch type. Should be either int or str.")

        # If weights path is provided, load the weights from the saved model
        if (os.path.isfile(generator_filename) and os.path.isfile(gen_history_filename)
                and os.path.isfile(genL1_history_filename)):
            checkpoint = torch.load(generator_filename, map_location=device)
            generator.load_state_dict(checkpoint)
            with open(gen_history_filename, 'rb') as f:
                gen_history = pickle.load(f)
            with open(genL1_history_filename, 'rb') as f:
                genL1_history = pickle.load(f)
            print(f"Generator {epoch} loaded.")
        else:
            raise ValueError(f" (Generator) No weights or history found at {model_dir}.")

        # If weights path is provided, load the weights from the saved model
        if os.path.isfile(discriminator_filename) and os.path.isfile(disc_history_filename):
            checkpoint = torch.load(discriminator_filename, map_location=device)
            # discriminator.load_state_dict(checkpoint)
            discriminator = 0
            with open(disc_history_filename, 'rb') as f:
                disc_history = pickle.load(f)
            print(f"Discriminator {epoch} loaded.")
        else:
            raise ValueError(f" (Discriminator) No weights or history found at {model_dir}.")

    return generator, discriminator, gen_history, genL1_history, disc_history




if __name__ == '__main__':
    ##################LOCAL########################
    # sim_dir = '/CODIGOS_TESIS/T2/dataset'
    # att_dir = '/CODIGOS_TESIS/T2/data_att0.5'
    # path = '/TESIS/DATOS_1/rf_test/'
    # dir_test = os.listdir(path)

    # this_dir = '/CODIGOS_TESIS/T2/wang/trained_models/L1_LOSS_udiff'

    # save_dir = '/CODIGOS_TESIS/T2/generated_samples/WANG/L1_LOSS_udiff/test/'

    # ###################CLUSTER####################
    #DATA DIRS
    sim_dir = '/mnt/nfs/isalazar/datasets/simulatedCystDataset/raw_0.0Att/'
    att_dir = '/mnt/nfs/isalazar/datasets/simulatedCystDataset/raw_0.5Att/'

    path = '/mnt/nfs/efernandez/datasets/dataRF/RF_test/'
    dir_test = os.listdir(path)
    #MODEL DIR
    # this_dir = '/nfs/privileged/isalazar/projects/ultrasound-image-formation/exploration/Journal2023/models/wang/'
    this_dir = '/mnt/nfs/efernandez/trained_models/WANG/L1_LOSS_udiff/'
    #SAVE_SAMPLES
    #data_with_attenuation
    # save_dir = '/mnt/nfs/efernandez/generated_samples/WANG/gen_att/'
    save_dir = '/mnt/nfs/efernandez/generated_samples/WANG/L1_LOSS_udiff/gen_att/'

    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
    print(device)

    model_gen, _, _, _, _ = load_gen_model(model_dir=this_dir,
                                             epoch='last',
                                             num_downs=5,
                                             device=device)

    model_gen.eval()

    # for simu in dir_test:
    #     simu_name = simu[:-4]
    num_samples = 500
    for simu in range(1,num_samples+1):
        simu_name = "simu" + str(simu).zfill(5)
        print(simu_name)
        ## Code to read channel data (2, 800, 128)
        P = LoadData_nair2020(att_dir, simu_name)
        # P,_,_=makeAberration(P, fwhm=1, rms_strength=50, seed=50)
        max_value = np.max(np.abs(np.array([P.idata, P.qdata])))
        P.idata = P.idata / max_value
        P.qdata = P.qdata / max_value

        laterals = np.linspace(P.grid_xlims[0], P.grid_xlims[-1], num=128)
        depths = np.linspace(30, 80, num=800) / 1000
        # P.grid_xlims = P.phantom_xlims
        P.grid_zlims = np.array([30, 80]) / 1000
        # Downsample channel data
        channel_data_phantom = downsample_channel_data(copy.deepcopy(P),
                                                    laterals=laterals,
                                                    depths=depths,
                                                    device=device)
        ##

        channel_data = channel_data_phantom / channel_data_phantom.abs().max()


        N, C, H, W = channel_data.size()
        z = torch.randn(N, 1, H, W).to(device)
        wang_phantom = model_gen(torch.cat((channel_data, z), dim=1))

        output_in_bmode_format = lambda x: (x * 60 - 60).detach().cpu().numpy().squeeze()

        wang_phantom = output_in_bmode_format(wang_phantom)

        np.save(save_dir+simu_name+".npy", wang_phantom)

        # extent=[-20,20,50,0]
        # plt.figure(figsize=(9, 3))
        # # plt.subplot(1, 5, 1)
        # plt.imshow(wang_phantom, cmap="gray", vmin=-60, vmax=0, extent=extent, origin="upper")
        # plt.colorbar()
        # plt.title('WANG')
        # plt.show()