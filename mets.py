# import OS module
import os

import csv

import numpy as np
import h5py
import os

from scipy.signal import hilbert

from metrics import compute_metrics

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

def make_pixel_grid_from_pos(x_pos, z_pos):
    zz, xx = np.meshgrid(z_pos, x_pos, indexing="ij") # 'ij' -> rows: z, columns: x
    yy = xx * 0
    grid = np.stack((xx, yy, zz), axis=-1)  # [nrows, ncols, 3]
    return grid

def main():

    # field names
    # fields = ['id', 'r', 'cx', 'cz', 'c', 'v6_contrast', 'v6_cnr', 'v6_gcnr', 'v6_snr', 'v7_contrast', 'v7_cnr', 'v7_gcnr', 'v7_snr', 'v9_contrast', 'v9_cnr', 'v9_gcnr', 'v9_snr', 'v10_contrast', 'v10_cnr', 'v10_gcnr', 'v10_snr']
    fields = ['id', 'r', 'cx', 'cz', 'c', 'v6_contrast', 'v6_cnr', 'v6_gcnr', 'v6_snr', 'v9_contrast', 'v9_cnr', 'v9_gcnr', 'v9_snr', 'udiff_contrast', 'u_diff_cnr', 'udiff_gcnr', 'udiff_snr']

    rows = []
    n_sample = 0

    depths = np.linspace(30*1e-3, 80*1e-3, num=800)
    laterals = np.linspace(-19*1e-3, 19*1e-3, num=128)
    grid = make_pixel_grid_from_pos(x_pos=laterals, z_pos=depths)

    model6_met=[]
    # model7_met=[]
    model9_met=[]
    model_udiff_met=[]

    model6_contrast=[]
    model6_cnr=[]
    model6_gcnr=[]
    model6_snr=[]

    # model7_contrast=[]
    # model7_cnr=[]
    # model7_gcnr=[]
    # model7_snr=[]

    model9_contrast=[]
    model9_cnr=[]
    model9_gcnr=[]
    model9_snr=[]

    model_udiff_contrast=[]
    model_udiff_cnr=[]
    model_udiff_gcnr=[]
    model_udiff_snr=[]

    num_samples = 100
    for simu in range(1,num_samples+1):
        simu_name = "simu" + str(simu).zfill(5)
        # filename=simu_name+".npy"

        sub_row = []

        P = LoadData_nair2020(h5_dir='/mnt/nfs/isalazar/datasets/simulatedCystDataset/raw_0.5Att/',
                            simu_name=simu_name)
        sub_row.append(n_sample)
        sub_row.append(int(simu[4:]))
        r = P.radius
        cx = P.pos_lat
        cz = P.pos_ax
        sub_row.append(P.radius)
        sub_row.append(P.pos_lat)
        sub_row.append(P.pos_ax)
        sub_row.append(P.c)

        #testing model v6
        dir_model_v6 = '/mnt/nfs/efernandez/generated_samples/DDPM_model/v6_TT_50epoch_gen_att/'
        bmode_output = np.load(dir_model_v6+filename).squeeze()
        bmode_output = (bmode_output + 1) * 30 - 60
        contrast, cnr, gcnr, snr = compute_metrics(cx, cz, r, bmode_output, grid)
        sub_row.append(contrast)
        sub_row.append(cnr)
        sub_row.append(gcnr)
        sub_row.append(snr)
        model6_contrast.append(contrast)
        model6_cnr.append(cnr)
        model6_gcnr.append(gcnr)
        model6_snr.append(snr)

        # #testing model v7
        # dir_model_v7 = '/mnt/nfs/efernandez/generated_samples/DDPM_model/v7_TT_50epoch_gen_att/'
        # bmode_output = np.load(dir_model_v7+filename).squeeze()
        # bmode_output = (bmode_output + 1) * 30 - 60
        # contrast, cnr, gcnr, snr = compute_metrics(cx, cz, r, bmode_output, grid)
        # sub_row.append(contrast)
        # sub_row.append(cnr)
        # sub_row.append(gcnr)
        # sub_row.append(snr) 
        # model7_contrast.append(contrast)
        # model7_cnr.append(cnr)
        # model7_gcnr.append(gcnr)
        # model7_snr.append(snr)

        #testing model v9
        dir_model_v9 = '/mnt/nfs/efernandez/generated_samples/DDPM_model/v9_TT_50epoch_gen_att/'
        bmode_output = np.load(dir_model_v9+filename).squeeze()
        bmode_output = (bmode_output + 1) * 30 - 60
        contrast, cnr, gcnr, snr = compute_metrics(cx, cz, r, bmode_output, grid)
        sub_row.append(contrast)
        sub_row.append(cnr)
        sub_row.append(gcnr)
        sub_row.append(snr)
        model9_contrast.append(contrast)
        model9_cnr.append(cnr)
        model9_gcnr.append(gcnr)
        model9_snr.append(snr)

        #testing model udiff
        dir_model_udiff = '/mnt/nfs/efernandez/generated_samples/UNet_difusiva/v1_50epoch_gen_att/'
        bmode_output = np.load(dir_model_v9+filename).squeeze()
        bmode_output = (bmode_output + 1) * 30 - 60
        contrast, cnr, gcnr, snr = compute_metrics(cx, cz, r, bmode_output, grid)
        sub_row.append(contrast)
        sub_row.append(cnr)
        sub_row.append(gcnr)
        sub_row.append(snr)
        model_udiff_contrast.append(contrast)
        model_udiff_cnr.append(cnr)
        model_udiff_gcnr.append(gcnr)
        model_udiff_snr.append(snr)

        rows.append(sub_row)

    model6_met.append(model6_contrast)
    model6_met.append(model6_cnr)
    model6_met.append(model6_gcnr)
    model6_met.append(model6_snr)

    # model7_met.append(model7_contrast)
    # model7_met.append(model7_cnr)
    # model7_met.append(model7_gcnr)
    # model7_met.append(model7_snr)

    model9_met.append(model9_contrast)
    model9_met.append(model9_cnr)
    model9_met.append(model9_gcnr)
    model9_met.append(model9_snr)

    # model10_met.append(model10_contrast)
    # model10_met.append(model10_cnr)
    # model10_met.append(model10_gcnr)
    # model10_met.append(model10_snr)

    model_udiff_met.append(model_udiff_contrast)
    model_udiff_met.append(model_udiff_cnr)
    model_udiff_met.append(model_udiff_gcnr)
    model_udiff_met.append(model_udiff_snr)

    save_dir='/mnt/nfs/efernandez/generated_samples/UNet_difusiva/'

    np.save(save_dir+"/met_6.npy", np.array(model6_met))
    # np.save(save_dir+"/met_7.npy", np.array(model7_met))
    np.save(save_dir+"/met_9.npy", np.array(model9_met))
    # np.save(save_dir+"/met_10.npy", np.array(model10_met))
    np.save(save_dir+"/met_udiff.npy", np.array(model_udiff_met))

    
    # name of csv file
    filename = "/mnt/nfs/efernandez/datasets/test_models_att.csv"
 
    # writing to csv file
    with open(filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
 
        # writing the fields
        csvwriter.writerow(fields)
 
        # writing the data rows
        csvwriter.writerows(rows)

if __name__ == '__main__':
    main()