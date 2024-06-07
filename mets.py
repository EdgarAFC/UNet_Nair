# import OS module
import os

import csv

import numpy as np
import h5py
import os

from scipy.signal import hilbert

from metrics import compute_metrics

###############################
file_loss = open("/mnt/nfs/efernandez/projects/UNet_Nair/log_metrics_att.txt", "w")
file_loss.close()
################################
def write_to_file(input): 
    with open("/mnt/nfs/efernandez/projects/UNet_Nair/log_metrics_att.txt", "a") as textfile: 
        textfile.write(str(input) + "\n") 
    textfile.close()

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
    fields = ['#','id', 'r', 'cx', 'cz', 'c', 'das_contrast', 'das_cnr', 'das_gcnr', 'das_snr', 'das_decay', 'das_contrast_att','std_contrast', 'std_cnr', 'std_gcnr', 'std_snr', 'std_decay', 'std_contrast_att','diff_contrast', 'diff_cnr', 'diff_gcnr', 'diff_snr', 'diff_decay', 'diff_contrast_att']

    rows = []

    das_met=[]
    std_met=[]
    diff_met=[]

    das_contrast=[]
    das_cnr=[]
    das_gcnr=[]
    das_snr=[]
    das_decay=[]
    das_contrast_att=[]

    std_contrast=[]
    std_cnr=[]
    std_gcnr=[]
    std_snr=[]
    std_decay=[]
    std_contrast_att=[]

    diff_contrast=[]
    diff_cnr=[]
    diff_gcnr=[]
    diff_snr=[]
    diff_decay=[]
    diff_contrast_att=[]

    num_samples = 500

    sim_dir = '/nfs/privileged/isalazar/datasets/simulatedCystDataset/raw_0.0Att/'
    att_dir = '/mnt/nfs/isalazar/datasets/simulatedCystDataset/raw_0.5Att/'

    # Get the list of all files and directories
    path = '/mnt/nfs/efernandez/datasets/dataRF/RF_test/'
    # path = '/TESIS/DATOS_1/rf_test/'
    dir_test = sorted(os.listdir(path))

    n_sample = 0

    for simu in range(1,num_samples+1):
        simu_name = "simu" + str(simu).zfill(5)
        # filename=simu_name+".npy"

    # for simu in dir_test:
    #     simu_name = simu[:-4]
        filename=simu_name+".npy"

        write_to_file('File')
        write_to_file(filename)
        write_to_file('---')

        sub_row = []

        P = LoadData_nair2020(h5_dir=att_dir,
                            simu_name=simu_name)
        
        depths = np.linspace(30*1e-3, 80*1e-3, num=800)
        laterals = np.linspace(P.grid_xlims[0], P.grid_xlims[-1], num=128)
        grid = make_pixel_grid_from_pos(x_pos=laterals, z_pos=depths)

        sub_row.append(n_sample)
        sub_row.append(int(simu_name[4:]))
        r = P.radius
        cx = P.pos_lat
        cz = P.pos_ax
        sub_row.append(P.radius)
        sub_row.append(P.pos_lat)
        sub_row.append(P.pos_ax)
        sub_row.append(P.c)

        columns_id = []
        for i in range(len(grid[:,:,0][1])):
            id_col = grid[:,:,0][1][i]
            # print("Id_col", id_col)
            
            if id_col < (cx-r) or id_col > (cx+r):
                columns_id.append(i)

        # write_to_file('Columns available' + str(columns_id)),
        number_columns = 50
        found_region = 0
        region = []
        i = 0
        id = 0
        while found_region == 0:  
            while id < number_columns-1:
                if columns_id[i:i+number_columns][id] == (columns_id[i:i+number_columns][id+1] -1):
                    found_region = 1
                    region.append(columns_id[i:i+number_columns][id])
                    id = id + 1
                else:
                    found_region = 0
                    # print('no corresponde', i)
                    region = []
                    i = i+1
                    id = 0
            
        region.append(columns_id[i:i+number_columns][id])

        write_to_file(region)

        #testing model DAS
        test_DAS = '/mnt/nfs/efernandez/generated_samples/DAS/gen_att/'
        bmode_output = np.load(test_DAS+filename).squeeze()
        bmode_output = np.clip(bmode_output, a_min=-60, a_max=0)
        contrast, cnr, gcnr, snr, decay_param, contrast_att = compute_metrics(cx, cz, r, bmode_output, grid, region)
        sub_row.append(contrast)
        sub_row.append(cnr)
        sub_row.append(gcnr)
        sub_row.append(snr)
        sub_row.append(decay_param)
        sub_row.append(contrast_att)
        das_contrast.append(contrast)
        das_cnr.append(cnr)
        das_gcnr.append(gcnr)
        das_snr.append(snr)
        das_decay.append(decay_param)
        das_contrast_att.append(contrast_att)
        write_to_file('DAS: ' + str(contrast_att))

        #testing standard training
        test_std = '/mnt/nfs/efernandez/generated_samples/UNet_difusiva/v1_380epoch/gen_att/'
        bmode_output = np.load(test_std+filename).squeeze()
        # bmode_output = (bmode_output + 1) * 30 - 60
        contrast, cnr, gcnr, snr, decay_param, contrast_att = compute_metrics(cx, cz, r, bmode_output, grid, region)
        sub_row.append(contrast)
        sub_row.append(cnr)
        sub_row.append(gcnr)
        sub_row.append(snr)
        sub_row.append(decay_param)
        sub_row.append(contrast_att)
        std_contrast.append(contrast)
        std_cnr.append(cnr)
        std_gcnr.append(gcnr)
        std_snr.append(snr)
        std_decay.append(decay_param)
        std_contrast_att.append(contrast_att)
        write_to_file('STD: ' + str(contrast_att))

        #testing model udiff
        dir_model_udiff = '/mnt/nfs/efernandez/generated_samples/DDPM_model/v6_TT_100steps/380epoch/gen_att/'
        bmode_output = np.load(dir_model_udiff+filename).squeeze()
        # bmode_output = (bmode_output + 1) * 30 - 60
        contrast, cnr, gcnr, snr, decay_param, contrast_att = compute_metrics(cx, cz, r, bmode_output, grid, region)
        sub_row.append(contrast)
        sub_row.append(cnr)
        sub_row.append(gcnr)
        sub_row.append(snr)
        sub_row.append(decay_param)
        sub_row.append(contrast_att)
        diff_contrast.append(contrast)
        diff_cnr.append(cnr)
        diff_gcnr.append(gcnr)
        diff_snr.append(snr)
        diff_decay.append(decay_param)
        diff_contrast_att.append(contrast_att)
        write_to_file('DIFF: ' + str(contrast_att))

        rows.append(sub_row)
        n_sample = n_sample +1

    das_met.append(das_contrast)
    das_met.append(das_cnr)
    das_met.append(das_gcnr)
    das_met.append(das_snr)
    das_met.append(das_decay)
    das_met.append(das_contrast_att)

    std_met.append(std_contrast)
    std_met.append(std_cnr)
    std_met.append(std_gcnr)
    std_met.append(std_snr)
    std_met.append(std_decay)
    std_met.append(std_contrast_att)

    diff_met.append(diff_contrast)
    diff_met.append(diff_cnr)
    diff_met.append(diff_gcnr)
    diff_met.append(diff_snr)
    diff_met.append(diff_decay)
    diff_met.append(diff_contrast_att)

    save_dir='/mnt/nfs/efernandez/generated_samples/mets/att_fin'

    np.save(save_dir+"/met_das_att_fin.npy", np.array(das_met))
    # np.save(save_dir+"/met_7.npy", np.array(model7_met))
    np.save(save_dir+"/met_std_att_fin.npy", np.array(std_met))
    # np.save(save_dir+"/met_10.npy", np.array(model10_met))
    np.save(save_dir+"/met_diff_att_fin.npy", np.array(diff_met))
    
    # name of csv file
    filename = '/mnt/nfs/efernandez/generated_samples/mets/att_fin.csv'
 
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