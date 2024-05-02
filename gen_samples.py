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

from model_diff import UNETv13
import guided_diffusion_v3 as gd
from matplotlib import pyplot as plt

from torch.nn.functional import grid_sample
from scipy import signal
from scipy.interpolate import interp1d

PI = 3.14159265359

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

def delay_plane(grid, angles):
    # Use broadcasting to simplify computations
    x = grid[:, 0].unsqueeze(0)
    z = grid[:, 2].unsqueeze(0)
    # For each element, compute distance to pixels
    dist = x * torch.sin(angles) + z * torch.cos(angles)
    # Output has shape [nangles, npixels]
    return dist

def delay_focus(grid, ele_pos):
    # Compute distance to user-defined pixels from elements
    # Expects all inputs to be torch tensors specified in SI units.
    # grid    Pixel positions in x,y,z    [npixels, 3]
    # ele_pos Element positions in x,y,z  [nelems, 3]
    # Get norm of distance vector between elements and pixels via broadcasting
    dist = torch.norm(grid - ele_pos.unsqueeze(0), dim=-1)
    # Output has shape [nelems, npixels]
    return dist

def make_pixel_grid(xlims, zlims, dx, dz):
    x_pos = np.arange(xlims[0], xlims[1], dx)
    z_pos = np.arange(zlims[0], zlims[1], dz)
    zz, xx = np.meshgrid(z_pos, x_pos, indexing="ij") # 'ij' -> rows: z, columns: x
    yy = xx * 0
    grid = np.stack((xx, yy, zz), axis=-1)  # [nrows, ncols, 3]
    return grid

## Simple phase rotation of I and Q component by complex angle theta
def complex_rotate(I, Q, theta):
    Ir = I * torch.cos(theta) - Q * torch.sin(theta)
    Qr = Q * torch.cos(theta) + I * torch.sin(theta)
    return Ir, Qr

class DAS_PW(torch.nn.Module):
    def __init__(
        self,
        P,
        grid,
        ang_list=None,
        ele_list=None,
        rxfnum=2,
        dtype=torch.float,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        """ Initialization method for DAS_PW.

        All inputs are specified in SI units, and stored in self as PyTorch tensors.
        INPUTS
        P           A PlaneWaveData object that describes the acquisition
        grid        A [ncols, nrows, 3] numpy array of the reconstruction grid
        ang_list    A list of the angles to use in the reconstruction
        ele_list    A list of the elements to use in the reconstruction
        rxfnum      The f-number to use for receive apodization
        dtype       The torch Tensor datatype (defaults to torch.float)
        device      The torch Tensor device (defaults to GPU execution)
        """
        super().__init__()
        # If no angle or element list is provided, delay-and-sum all
        if ang_list is None:
            ang_list = range(P.angles.shape[0])
        elif not hasattr(ang_list, "__getitem__"):
            ang_list = [ang_list]
        if ele_list is None:
            ele_list = range(P.ele_pos.shape[0])
        elif not hasattr(ele_list, "__getitem__"):
            ele_list = [ele_list]

        # Convert plane wave data to tensors
        self.angles = torch.tensor(P.angles, dtype=dtype, device=device)
        self.ele_pos = torch.tensor(P.ele_pos, dtype=dtype, device=device)
        self.fc = torch.tensor(P.fc, dtype=dtype, device=device)
        self.fs = torch.tensor(P.fs, dtype=dtype, device=device)
        self.fdemod = torch.tensor(P.fdemod, dtype=dtype, device=device)
        self.c = torch.tensor(P.c, dtype=dtype, device=device)
        self.time_zero = torch.tensor(P.time_zero, dtype=dtype, device=device)

        # Convert grid to tensor
        self.grid = torch.tensor(grid, dtype=dtype, device=device).reshape(-1, 3)
        self.out_shape = grid.shape[:-1]

        # Store other information as well
        self.ang_list = torch.tensor(ang_list, dtype=torch.long, device=device)
        self.ele_list = torch.tensor(ele_list, dtype=torch.long, device=device)
        self.dtype = dtype
        self.device = device

    def forward(self, x, accumulate=False):
        """ Forward pass for DAS_PW neural network. """
        dtype, device = self.dtype, self.device

        # Load data onto device as a torch tensor

        idata, qdata = x
        idata = torch.tensor(idata, dtype=dtype, device=device)
        qdata = torch.tensor(qdata, dtype=dtype, device=device)

        # Compute delays in meters
        nangles = len(self.ang_list)
        nelems = len(self.ele_list)
        npixels = self.grid.shape[0]
        xlims = (self.ele_pos[0, 0], self.ele_pos[-1, 0])  # Aperture width
        txdel = torch.zeros((nangles, npixels), dtype=dtype, device=device)
        rxdel = torch.zeros((nelems, npixels), dtype=dtype, device=device)
        txapo = torch.ones((nangles, npixels), dtype=dtype, device=device)
        rxapo = torch.ones((nelems, npixels), dtype=dtype, device=device)
        for i, tx in enumerate(self.ang_list):
            txdel[i] = delay_plane(self.grid, self.angles[[tx]])
            # txdel[i] += self.time_zero[tx] * self.c   # ORIGINAL
            txdel[i] -= self.time_zero[tx] * self.c     # IT HAS TO BE "-"
            # txapo[i] = apod_plane(self.grid, self.angles[tx], xlims)
        for j, rx in enumerate(self.ele_list):
            rxdel[j] = delay_focus(self.grid, self.ele_pos[[rx]])
            # rxapo[i] = apod_focus(self.grid, self.ele_pos[rx])

        # Convert to samples
        txdel *= self.fs / self.c
        rxdel *= self.fs / self.c

        # Initialize the output array
        idas = torch.zeros(npixels, dtype=self.dtype, device=self.device)
        qdas = torch.zeros(npixels, dtype=self.dtype, device=self.device)
        iq_cum = None
        if accumulate:
            iq_cum = torch.zeros(nangles, npixels, nelems, 2, dtype=dtype, device='cpu')
        # Loop over angles and elements
        # for t, td, ta in zip(self.ang_list, txdel, txapo):
        # for idx1, (t, td, ta) in tqdm(enumerate(zip(self.ang_list, txdel, txapo)), total=nangles):
        for idx1, (t, td, ta) in enumerate(zip(self.ang_list, txdel, txapo)):
            for idx2, (r, rd, ra) in enumerate(zip(self.ele_list, rxdel, rxapo)):
                # Grab data from t-th Tx, r-th Rx
                # Avoiding stack because of autograd problems
                # iq = torch.stack((idata[t, r], qdata[t, r]), dim=0).view(1, 2, 1, -1)
                i_iq = idata[t, r].view(1, 1, 1, -1)
                q_iq = qdata[t, r].view(1, 1, 1, -1)
                # Convert delays to be used with grid_sample
                delays = td + rd
                dgs = (delays.view(1, 1, -1, 1) * 2 + 1) / idata.shape[-1] - 1
                dgs = torch.cat((dgs, 0 * dgs), axis=-1)
                # Interpolate using grid_sample and vectorize using view(-1)
                # ifoc, qfoc = grid_sample(iq, dgs, align_corners=False).view(2, -1)
                ifoc = grid_sample(i_iq, dgs, align_corners=False).view(-1)
                qfoc = grid_sample(q_iq, dgs, align_corners=False).view(-1)
                # torch.Size([144130])
                # Apply phase-rotation if focusing demodulated data
                if self.fdemod != 0:
                    tshift = delays.view(-1) / self.fs - self.grid[:, 2] * 2 / self.c
                    theta = 2 * PI * self.fdemod * tshift
                    ifoc, qfoc = complex_rotate(ifoc, qfoc, theta)
                # Apply apodization, reshape, and add to running sum
                # apods = ta * ra
                # idas += ifoc * apods
                # qdas += qfoc * apods
                idas += ifoc
                qdas += qfoc
                # torch.Size([355*406])
                if accumulate:
                    # 1, npixels, nelems, 2
                    iq_cum[idx1, :, idx2, 0] = ifoc.cpu()
                    iq_cum[idx1, :, idx2, 1] = qfoc.cpu()

        # Finally, restore the original pixel grid shape and convert to numpy array
        idas = idas.view(self.out_shape)
        qdas = qdas.view(self.out_shape)

        env = torch.sqrt(idas**2 + qdas**2)
        bimg = 20 * torch.log10(env + torch.tensor(1.0*1e-25))
        bimg = bimg - torch.max(bimg)
        return bimg, env, idas, qdas, iq_cum
class LoadData_phantomLIM_ATSmodel539(PlaneWaveData):
    def __init__(self, h5_dir, h5_name):
        # super().__init__(self)
        h5filename = os.path.join(h5_dir, h5_name)
        with h5py.File(h5filename, "r") as f:
            self.angles = np.array([0])
            # self.angles = np.array(f['angles']).squeeze()
            self.fc = np.array(f['fc']).item()
            # self.fc = np.array(5*1e6).item()
            self.fs = np.array(f['fs']).item()
            # self.fs = np.array(29.6*1e6).item()
            self.c = np.array(f['c']).item()
            self.x = np.array(f['x'])      # x 1x128 in MATLAB
            self.z = np.array(f['z'])      # z 1x2048 in MATLAB
            xs = np.squeeze(self.x)
            self.ele_pos = np.array([xs, np.zeros_like(xs), np.zeros_like(xs)]).T
            self.data_IQ = np.array(f["rf"], dtype="float32")[3]    # 2048x128 in MATLAB
            # print(f["rf"][3])
            # print(f["rf"][3].shape)
            #
            # self.data_IQ = f["rf"][3]    # 2048x128 in MATLAB
            # self.idata = np.expand_dims(np.array(f["signal"], dtype="float32"), 0)
            # self.qdata = np.imag(hilbert(self.idata, axis=-1))
            # print(self.data_IQ.shape)
            # self.idata = self.data_IQ[len(self.data_IQ)//2][None]   # 1 x 128 x 3328
            # self.qdata = np.imag(hilbert(self.idata, axis=-1))      # 1 x 128 x 3328
            self.idata = self.data_IQ[None]  # 1 x 128 x 3328
            # self.idata = self.data_IQ   # 1 x 128 x 3328
            self.qdata = np.imag(hilbert(self.idata, axis=-1))      # 1 x 128 x 3328
            # self.time_zero = np.array([np.array(1e-6).item()])
            self.time_zero = np.array([np.array(f['time_zero'][4]).item()])
            # self.time_zero = np.array(f['time_zero']).squeeze()
            self.fdemod = 0
            r = 8 * 1e-3
            xctr = 0.0
            zctr = 40 * 1e-3
            self.pos_lat = xctr
            self.pos_ax = zctr
            self.radius = r
            # self.grid_zlims = [0.002, 0.052]
            # self.grid_zlims = [0.002, 0.055]
            self.phantom_zlims = [0.03, 0.08]
            # self.phantom_xlims = [-0.019, 0.019]
            self.phantom_xlims = [xs[0], xs[-1]]
        self.validate()
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
        print("Dataset successfully loaded")

def makeAberration(P, fwhm, rms_strength, seed):
    Fs = P.fs
    pitch = (P.ele_pos[1, 0] - P.ele_pos[0, 0]) * 1000  # pitch in mm
    recorded_signals = P.idata.squeeze()
    num_elements, num_samples = recorded_signals.shape

    # Step 1: Create profile
    np.random.seed(seed)
    rand_arr = np.random.normal(loc=0, scale=1, size=(num_elements, 1))
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Standard deviation of the Gaussian function (in mm)
    sigma_in_px = sigma / pitch
    gauss_kernel = signal.gaussian(num_elements, std=sigma_in_px)[:, None]
    profile = signal.fftconvolve(rand_arr, gauss_kernel, mode='same')
    rms_value = np.sqrt(np.mean(profile ** 2, axis=0))
    profile = (1e-9)*rms_strength*profile/rms_value   # from ns to seconds
    # profile = (1e-9)*rms_strength*np.ones_like(profile)

    aberrated_signals = np.zeros_like(recorded_signals)
    start, end = 0, (num_samples/Fs)     # in seconds
    time = np.linspace(start, end, num_samples)        # in seconds
    for idx, delay in enumerate(profile):
        new_time = time+delay
        curr_signal = recorded_signals[idx, :]
        interpolated_signal = interp1d(time, curr_signal,
                                       kind='linear',
                                       fill_value='extrapolate')(new_time)
        aberrated_signals[idx, :] = interpolated_signal

    P.idata = aberrated_signals[None, :]
    P.qdata = np.imag(hilbert(P.idata, axis=-1))

    dct = {'gauss_kernel': gauss_kernel,
           'signals': recorded_signals,
           'aberrated_signals': aberrated_signals}

    # return P, phase_screen, np.mean(phase_screen, axis=1), dct
    return P, profile, dct

def make_bimg_das1(h5_dir, simu_name, device):

    P = LoadData_nair2020(h5_dir, simu_name)
    # P,_,_=makeAberration(P, fwhm=2, rms_strength=30, seed=50)
    # P = LoadData_phantomLIM_ATSmodel539(h5_dir=h5_dir, h5_name=simu_name)
    norm_value = np.max((np.abs(P.idata), np.abs(P.qdata)))
    P.idata = P.idata / norm_value
    P.qdata = P.qdata / norm_value
    # norm = np.max(np.sqrt(P.idata ** 2 + P.qdata ** 2))
    # P.idata = P.idata/norm
    # P.qdata = P.qdata/norm

    wvln = P.c / P.fc
    dx, dz = wvln / 3, wvln / 3

    laterals = np.linspace(P.grid_xlims[0], P.grid_xlims[-1], num=128)
    depth_ini=30
    depths = np.linspace(depth_ini, depth_ini + 50, num=800) / 1000
    grid_full = make_pixel_grid_from_pos(x_pos=laterals, z_pos=depths)
    # grid = make_pixel_grid(P.grid_xlims, P.grid_zlims, dx, dz)
    print(dx)
    print(dz)
    # grid = make_pixel_grid(P.phantom_xlims, P.phantom_zlims, dx, dz)
    id_angle = len(P.angles) // 2
    dasNet = DAS_PW(P, grid_full, ang_list=id_angle, device=device)
    bimg, env, _, _, _ = dasNet((P.idata, P.qdata), accumulate=False)
    bimg = bimg.detach().cpu().numpy()
    env = env.detach().cpu().numpy()
    return bimg, env

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
    print(device)

    sim_dir = '/nfs/privileged/isalazar/datasets/simulatedCystDataset/raw_0.0Att/'
    att_dir = '/mnt/nfs/isalazar/datasets/simulatedCystDataset/raw_0.5Att/'
    shape_dir = ''

    # sim_dir = '/CODIGOS_TESIS/T2/dataset'
    # att_dir = '/CODIGOS_TESIS/T2/data_att0.5'
    # shape_dir = '/CODIGOS_TESIS/T2/shape'
    # phantom_dir = '/CODIGOS_TESIS/T2/phantom_data'

    # save_dir = '/mnt/nfs/efernandez/generated_samples/DDPM_model/v6_TT_100steps/380epochs/gen_test/'
    # save_dir = '/mnt/nfs/efernandez/generated_samples/DAS/gen_att/'
    save_dir = '/mnt/nfs/efernandez/generated_samples/UNet_difusiva/v1_380epoch/gen_att/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    model_dir='/mnt/nfs/efernandez/trained_models/UNet_difusiva/v1_300epoch/'
    # model_dir='/mnt/nfs/efernandez/trained_models/DDPM_model/v6_TT_100steps/'
    training_epochs = 380#10
    model = UNETv13(residual=True, attention_res=[], group_norm=True).to(device)
    model.load_state_dict(torch.load(f"{model_dir}/model_{training_epochs}.pth", map_location=device))
    # diffusion = create_gaussian_diffusion(steps=100,noise_schedule="linear")

    depth_ini=30

    num_samples = 500

    # inclusion_shape = 'triangle'
    # h5name = "inclusion_%s.h5" % inclusion_shape

    # Get the list of all files and directories
    path = '/mnt/nfs/efernandez/datasets/dataRF/RF_test/'
    # path = '/TESIS/DATOS_1/rf_test/'
    dir_test = os.listdir(path)

    phantom_names=["IS_L11-4v_data1_RF.h5","IS_L11-4v_data2_RF.h5","IS_L11-4v_data3_RF.h5","IS_L11-4v_data4_RF.h5"]

    for simu in range(1,num_samples+1):
        simu_name = "simu" + str(simu).zfill(5)
        # bmode, grid_full=create_phantom_bmodes2("/CODIGOS_TESIS/T2/shape",h5name,simu_name, depth_ini, device, model,diffusion)
        # bmode, grid_full=create_phantom_bmodes_att_diff(att_dir,simu_name, depth_ini, device, model,diffusion)
        bmode, grid_full=create_phantom_bmodes_att_conv(att_dir,simu_name, depth_ini, device, model)
        # bmode, _ = make_bimg_das1(att_dir, simu_name, device=device)
        np.save(save_dir+simu_name+".npy", bmode)
        # np.save(save_dir+"grid0000"+str(simu)+".npy", grid_full)
        # extent=[-20,20,50,0]
        # plt.figure(figsize=(9, 3))
        # # plt.subplot(1, 5, 1)
        # plt.imshow(bmode, cmap="gray", vmin=-60, vmax=0, extent=extent, origin="upper")
        # plt.colorbar()
        # plt.title('V7')
        # plt.show()

    # for simu in dir_test:
    #     simu_name = simu[:-4]
    #     # bmode, grid_full=create_phantom_bmodes2("/CODIGOS_TESIS/T2/shape",h5name,simu_name, depth_ini, device, model,diffusion)
    #     # bmode, grid_full=create_phantom_bmodes_att_diff(att_dir,simu_name, depth_ini, device, model,diffusion)
    #     # bmode, grid_full=create_phantom_bmodes_att_conv(att_dir,simu_name, depth_ini, device, model)
    #     bmode, _ = make_bimg_das1(sim_dir, simu_name, device=device)
    #     print(bmode.shape)
    #     np.save(save_dir+simu_name+".npy", bmode)
    #     # np.save(save_dir+"grid0000"+str(simu)+".npy", grid_full)
    #     # extent=[-20,20,50,0]
    #     # plt.figure(figsize=(9, 3))
    #     # # plt.subplot(1, 5, 1)
    #     # plt.imshow(bmode, cmap="gray", vmin=-60, vmax=0, extent=extent, origin="upper")
    #     # plt.colorbar()
    #     # plt.title('V7')
    #     # plt.show()

    

if __name__ == '__main__':
    main()
    