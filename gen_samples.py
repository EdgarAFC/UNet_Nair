# import OS module
import os

import csv

import numpy as np
import h5py
import os

from scipy.signal import hilbert
from scipy import interpolate
import copy

import torch

from model7_shift_scale import UNETv13
import guided_diffusion_v3 as gd
from matplotlib import pyplot as plt

class PlaneWaveData:
    """ A template class that contains the plane wave data.

    PlaneWaveData is a container or dataclass that holds all of the information
    describing a plane wave acquisition. Users should create a subclass that reimplements
    __init__() according to how their data is stored.

    The required information is:
    idata       In-phase (real) data with shape (nangles, nchans, nsamps)
    qdata       Quadrature (imag) data with shape (nangles, nchans, nsamps)
    angles      List of angles [radians]
    ele_pos     Element positions with shape (N,3) [m]
    fc          Center frequency [Hz]
    fs          Sampling frequency [Hz]
    fdemod      Demodulation frequency, if data is demodulated [Hz]
    c           Speed of sound [m/s]
    time_zero   List of time zeroes for each acquisition [s]

    Correct implementation can be checked by using the validate() method. See the
    PICMUSData class for a fully implemented example.
    """

    def __init__(self):
        """ Users must re-implement this function to load their own data. """
        # Do not actually use PlaneWaveData.__init__() as is.
        raise NotImplementedError
        # We provide the following as a visual example for a __init__() method.
        nangles, nchans, nsamps = 2, 3, 4
        # Initialize the parameters that *must* be populated by child classes.
        self.idata = np.zeros((nangles, nchans, nsamps), dtype="float32")
        self.qdata = np.zeros((nangles, nchans, nsamps), dtype="float32")
        self.angles = np.zeros((nangles,), dtype="float32")
        self.ele_pos = np.zeros((nchans, 3), dtype="float32")
        self.fc = 5e6
        self.fs = 20e6
        self.fdemod = 0
        self.c = 1540
        self.time_zero = np.zeros((nangles,), dtype="float32")

    def validate(self):
        """ Check to make sure that all information is loaded and valid. """
        # Check size of idata, qdata, angles, ele_pos
        assert self.idata.shape == self.qdata.shape
        assert self.idata.ndim == self.qdata.ndim == 3
        nangles, nchans, nsamps = self.idata.shape
        assert self.angles.ndim == 1 and self.angles.size == nangles
        assert self.ele_pos.ndim == 2 and self.ele_pos.shape == (nchans, 3)
        # Check frequencies (expecting more than 0.1 MHz)
        assert self.fc > 1e5
        assert self.fs > 1e5
        assert self.fdemod > 1e5 or self.fdemod == 0
        # Check speed of sound (should be between 1000-2000 for medical imaging)
        assert 1000 <= self.c <= 2000
        # Check that a separate time zero is provided for each transmit
        assert self.time_zero.ndim == 1 and self.time_zero.size == nangles
        # print("Dataset successfully loaded")

class LoadData_nair2020(PlaneWaveData):
    def __init__(self, h5_dir, simu_name):
        # raw_dir = 'D:\Itamar\\datasets\\fieldII\\simulation\\nair2020\\raw'
        # raw_dir = 'D:\\Itamar\\datasets\\fieldII\\simulation\\nair2020\\raw12500_0.5attenuation'
        simu_number = int(simu_name[4:])
        lim_inf = 1000*((simu_number-1)//1000) + 1
        lim_sup = lim_inf + 999
        h5_name = 'simus_%.5d-%.5d.h5' % (lim_inf, lim_sup)
        h5filename = os.path.join(h5_dir, h5_name)
        # print(h5filename)
        with h5py.File(h5filename, "r") as g:
        # g = h5py.File(filename, "r")
            f = g[simu_name]
            self.idata = np.expand_dims(np.array(f["signal"], dtype="float32"), 0)
            self.qdata = np.imag(hilbert(self.idata, axis=-1))
            self.angles = np.array([0])
            self.fc = np.array(f['fc']).item()
            self.fs = np.array(f['fs']).item()
            self.c = np.array(f['c']).item()
            self.time_zero = np.array([np.array(f['time_zero']).item()])
            self.fdemod = 0
            xs = np.squeeze(np.array(f['ele_pos']))
            self.grid_xlims = [xs[0], xs[-1]]
            self.grid_zlims = [30*1e-3, 80*1e-3]
            self.ele_pos = np.array([xs, np.zeros_like(xs), np.zeros_like(xs)]).T
            self.pos_lat = np.array(f['lat_pos']).item()
            self.pos_ax = np.array(f['ax_pos']).item()
            self.radius = np.array(f['r']).item()
        super().validate()

class ShapeVariationDataLoader(PlaneWaveData):
    def __init__(self, h5_dir, h5_name, simu_name):
        h5filename = os.path.join(h5_dir, h5_name)
        print(h5filename)
        with h5py.File(h5filename, "r") as g:
            print(g.keys())
            f = g[simu_name]
            self.idata = np.expand_dims(np.array(f["signal"], dtype="float32"), 0)
            self.qdata = np.imag(hilbert(self.idata, axis=-1))
            self.angles = np.array([0])
            self.fc = np.array(f['fc']).item()
            self.fs = np.array(f['fs']).item()
            self.c = np.array(f['c']).item()
            self.time_zero = np.array([np.array(f['time_zero']).item()])
            self.phantom_xlims = [-0.02, 0.02]
            self.phantom_zlims = [0.03, 0.08]
            self.grid_xlims = [-0.02, 0.02]
            self.grid_zlims = [0.03, 0.08]
            self.fdemod = 0
            xs = np.squeeze(np.array(f['ele_pos']))
            self.ele_pos = np.array([xs, np.zeros_like(xs), np.zeros_like(xs)]).T
            # self.pos_lat = np.array(f['lat_pos']).item()
            # self.pos_ax = np.array(f['ax_pos']).item()
        super().validate()

def create_gaussian_diffusion(
        *,
        steps=1000,
        learn_sigma=False,
        sigma_small=False,
        noise_schedule="linear",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return gd.SpacedDiffusion(
        use_timesteps=gd.space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )

def make_pixel_grid_from_pos(x_pos, z_pos):
    zz, xx = np.meshgrid(z_pos, x_pos, indexing="ij") # 'ij' -> rows: z, columns: x
    yy = xx * 0
    grid = np.stack((xx, yy, zz), axis=-1)  # [nrows, ncols, 3]
    return grid

def roi_channel_data(P, laterals, depths):

    nLats, nDepths = len(laterals), len(depths)
    depth_samples = 2 * (depths / P.c) * P.fs
    tzero_samples = int(P.time_zero * P.fs)

    donwsample_depths = np.zeros((nDepths, P.idata.shape[1], 2))
    for idx in np.arange(P.idata.shape[1]):
        iplane = np.concatenate((np.zeros(tzero_samples), P.idata[0, idx, :]))  # fill with zeros
        qplane = np.concatenate((np.zeros(tzero_samples), P.qdata[0, idx, :]))  # fill with zeros

        data_axis = np.arange(len(iplane))

        func_i = interpolate.interp1d(data_axis, iplane, kind='slinear')
        func_q = interpolate.interp1d(data_axis, qplane, kind='slinear')

        donwsample_depths[:, idx, 0] = func_i(depth_samples)
        donwsample_depths[:, idx, 1] = func_q(depth_samples)

    XDC_elements_laterals = np.linspace(P.grid_xlims[0], P.grid_xlims[-1], P.idata.shape[1])
    donwsample_total = np.zeros((nDepths, nLats, 2))
    for idx in np.arange(nDepths):
        iplane = donwsample_depths[idx, :, 0]
        qplane = donwsample_depths[idx, :, 1]

        func_i = interpolate.interp1d(XDC_elements_laterals, iplane, kind='slinear')
        func_q = interpolate.interp1d(XDC_elements_laterals, qplane, kind='slinear')

        donwsample_total[idx, :, 0] = func_i(laterals)
        donwsample_total[idx, :, 1] = func_q(laterals)

    # iq = np.sqrt(input_Id[:, :, 0] ** 2 + input_Id[:, :, 1] ** 2)
    return donwsample_total

def downsample_channel_data(H, laterals, depths, device):
    norm_value = np.max((np.abs(H.idata), np.abs(H.qdata)))
    H.idata = H.idata / norm_value
    H.qdata = H.qdata / norm_value
    # 0.002, 0.052
    # x, depth_samples, tzero_samples = create_input_Id(H,npoints, depths)  # x: npointsx128x2
    x = roi_channel_data(H,laterals=laterals, depths=depths)  # x: npointsx128x2
    # norm_value_x = np.max(np.abs(x))
    # print(f'nair_input_corrected: -> norm_value_x: {norm_value_x}')

    # x = x/norm_value_x
    x = torch.from_numpy(x)[None, :].permute(0, 3, 1, 2).to(torch.float).to(device)
    # x: 1, 2, npoints, 128
    return x

def create_phantom_shape_diff(h5_dir, h5_name, simu_name, depth_ini, device, model,diffusion):
    
    # depth_ini = get_data_from_name(h5name)
    P = ShapeVariationDataLoader(h5_dir, h5_name,simu_name)
    max_value = np.max(np.abs(np.array([P.idata, P.qdata])))
    P.idata = P.idata / max_value
    P.qdata = P.qdata / max_value

    laterals = np.linspace(P.grid_xlims[0], P.grid_xlims[-1], num=128)
    depths = np.linspace(depth_ini, depth_ini + 50, num=800) / 1000
    # P.grid_xlims = P.phantom_xlims
    P.grid_zlims = np.array([depth_ini, depth_ini + 50]) / 1000
    # Downsample channel data
    channel_data_phantom = downsample_channel_data(copy.deepcopy(P),
                                                   laterals=laterals,
                                                   depths=depths,
                                                   device=device)
    channel_data_phantom = channel_data_phantom / channel_data_phantom.abs().max()
    laterals = np.linspace(P.grid_xlims[0], P.grid_xlims[-1], 128)
    grid_full = make_pixel_grid_from_pos(x_pos=laterals, z_pos=depths)

    # bmode = model(channel_data_phantom)
    # y = torch.empty(1, 1, 800,128)
    # bmode = model(channel_data_phantom)
    y = torch.empty(1, 1, 800,128)
    bmode = diffusion.p_sample_loop(model, y.shape, channel_data_phantom, progress=False, clip_denoised=True)

    # output_in_bmode_format = lambda x: np.clip((x * 60 - 60).detach().cpu().numpy().squeeze(), a_min=-60, a_max=0)
    output_in_bmode_format = lambda x: np.clip(((x + 1) * 30 - 60).detach().cpu().numpy().squeeze(), a_min=-60, a_max=0)

    bmode = output_in_bmode_format(bmode)

    return bmode, grid_full

def create_phantom_shape_conv(h5_dir, h5_name, simu_name, depth_ini, device, model):
    
    # depth_ini = get_data_from_name(h5name)
    P = ShapeVariationDataLoader(h5_dir, h5_name,simu_name)
    max_value = np.max(np.abs(np.array([P.idata, P.qdata])))
    P.idata = P.idata / max_value
    P.qdata = P.qdata / max_value

    laterals = np.linspace(P.grid_xlims[0], P.grid_xlims[-1], num=128)
    depths = np.linspace(depth_ini, depth_ini + 50, num=800) / 1000
    # P.grid_xlims = P.phantom_xlims
    P.grid_zlims = np.array([depth_ini, depth_ini + 50]) / 1000
    # Downsample channel data
    channel_data_phantom = downsample_channel_data(copy.deepcopy(P),
                                                   laterals=laterals,
                                                   depths=depths,
                                                   device=device)
    channel_data_phantom = channel_data_phantom / channel_data_phantom.abs().max()
    laterals = np.linspace(P.grid_xlims[0], P.grid_xlims[-1], 128)
    grid_full = make_pixel_grid_from_pos(x_pos=laterals, z_pos=depths)

    # bmode = model(channel_data_phantom)
    # y = torch.empty(1, 1, 800,128)
    bmode = model(channel_data_phantom)
    # y = torch.empty(1, 1, 800,128)
    # bmode = diffusion.p_sample_loop(model, y.shape, channel_data_phantom, progress=False, clip_denoised=True)

    # output_in_bmode_format = lambda x: np.clip((x * 60 - 60).detach().cpu().numpy().squeeze(), a_min=-60, a_max=0)
    output_in_bmode_format = lambda x: np.clip(((x + 1) * 30 - 60).detach().cpu().numpy().squeeze(), a_min=-60, a_max=0)

    bmode = output_in_bmode_format(bmode)

    return bmode, grid_full

def create_phantom_bmodes_att_conv(h5_dir, simu_name, depth_ini, device, model):
    # depth_ini = get_data_from_name(h5name)
    P = LoadData_nair2020(h5_dir, simu_name)
    max_value = np.max(np.abs(np.array([P.idata, P.qdata])))
    P.idata = P.idata / max_value
    P.qdata = P.qdata / max_value

    laterals = np.linspace(P.grid_xlims[0], P.grid_xlims[-1], num=128)
    depths = np.linspace(depth_ini, depth_ini + 50, num=800) / 1000
    # P.grid_xlims = P.phantom_xlims
    P.grid_zlims = np.array([depth_ini, depth_ini + 50]) / 1000
    # Downsample channel data
    channel_data_phantom = downsample_channel_data(copy.deepcopy(P),
                                                   laterals=laterals,
                                                   depths=depths,
                                                   device=device)
    channel_data_phantom = channel_data_phantom / channel_data_phantom.abs().max()
    laterals = np.linspace(P.grid_xlims[0], P.grid_xlims[-1], 128)
    grid_full = make_pixel_grid_from_pos(x_pos=laterals, z_pos=depths)

    # bmode = model(channel_data_phantom)
    # y = torch.empty(1, 1, 800,128)
    bmode = model(channel_data_phantom)

    # output_in_bmode_format = lambda x: np.clip((x * 60 - 60).detach().cpu().numpy().squeeze(), a_min=-60, a_max=0)
    output_in_bmode_format = lambda x: np.clip(((x + 1) * 30 - 60).detach().cpu().numpy().squeeze(), a_min=-60, a_max=0)

    bmode = output_in_bmode_format(bmode)

    return bmode, grid_full

def create_phantom_bmodes_att_diff(h5_dir, simu_name, depth_ini, device, model,diffusion):
    # depth_ini = get_data_from_name(h5name)
    P = LoadData_nair2020(h5_dir, simu_name)
    max_value = np.max(np.abs(np.array([P.idata, P.qdata])))
    P.idata = P.idata / max_value
    P.qdata = P.qdata / max_value

    laterals = np.linspace(P.grid_xlims[0], P.grid_xlims[-1], num=128)
    depths = np.linspace(depth_ini, depth_ini + 50, num=800) / 1000
    # P.grid_xlims = P.phantom_xlims
    P.grid_zlims = np.array([depth_ini, depth_ini + 50]) / 1000
    # Downsample channel data
    channel_data_phantom = downsample_channel_data(copy.deepcopy(P),
                                                   laterals=laterals,
                                                   depths=depths,
                                                   device=device)
    channel_data_phantom = channel_data_phantom / channel_data_phantom.abs().max()
    laterals = np.linspace(P.grid_xlims[0], P.grid_xlims[-1], 128)
    grid_full = make_pixel_grid_from_pos(x_pos=laterals, z_pos=depths)

    # bmode = model(channel_data_phantom)
    y = torch.empty(1, 1, 800,128)
    bmode = diffusion.p_sample_loop(model, y.shape, channel_data_phantom, progress=False, clip_denoised=True)

    # output_in_bmode_format = lambda x: np.clip((x * 60 - 60).detach().cpu().numpy().squeeze(), a_min=-60, a_max=0)
    output_in_bmode_format = lambda x: np.clip(((x + 1) * 30 - 60).detach().cpu().numpy().squeeze(), a_min=-60, a_max=0)

    bmode = output_in_bmode_format(bmode)

    return bmode, grid_full

def main():

    # Get the list of all files and directories
    path = "/mnt/nfs/efernandez/datasets/dataRF/RF_test/"
    dir_list = sorted(os.listdir(path))
    # print("Files and directories in '", path, "' :")
    # prints all files
    # print(dir_list)

    for i in range(0, len(dir_list)):
        dir_list[i] = dir_list[i][:-4]

    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
    print(device)


    att_dir = '/mnt/nfs/isalazar/datasets/simulatedCystDataset/raw_0.5Att/'
    shape_dir = ''


    save_dir = '/mnt/nfs/efernandez/generated_samples/DDPM_model/v6_TT_50epoch_gen_att/'
    # save_dir = '/CODIGOS_TESIS/T2/trained_models/Udiffusive/v1_50epoch_gen_att/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    model_dir='/mnt/nfs/efernandez/trained_models/DDPM_model/v6_TT_50epoch'
    # model_dir='/CODIGOS_TESIS/T2/trained_models/Udiffusive'
    training_epochs = 50#10
    model = UNETv13(residual=True, attention_res=[], group_norm=True).to(device)
    model.load_state_dict(torch.load(f"{model_dir}/model_{training_epochs}.pth", map_location=device))
    
    depth_ini=30

    num_samples = 100
    diffusion = create_gaussian_diffusion(noise_schedule="linear")

    # inclusion_shape = 'triangle'
    # h5name = "inclusion_%s.h5" % inclusion_shape

    for simu in range(1,num_samples+1):
        simu_name = "simu" + str(simu).zfill(5)
        # bmode, grid_full=create_phantom_bmodes2("/CODIGOS_TESIS/T2/shape",h5name,simu_name, depth_ini, device, model,diffusion)
        bmode, grid_full=create_phantom_bmodes_att_diff(att_dir,simu_name, depth_ini, device, model,diffusion)
        np.save(save_dir+simu_name+".npy", bmode)
        # np.save(save_dir+"grid0000"+str(simu)+".npy", grid_full)
        # extent=[-20,20,50,0]
        # plt.figure(figsize=(9, 3))
        # # plt.subplot(1, 5, 1)
        # plt.imshow(bmode, cmap="gray", vmin=-60, vmax=0, extent=extent, origin="upper")
        # plt.colorbar()
        # plt.title('V7')
        # plt.show()

    

if __name__ == '__main__':
    main()
    