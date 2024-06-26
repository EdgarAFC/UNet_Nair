# import OS module
import os

import csv

import numpy as np
import h5py
import os

from scipy.signal import hilbert

from metrics import compute_metrics

from gen_samples import LoadData_nair2020, downsample_channel_data

###############################
file_loss = open("/mnt/nfs/efernandez/projects/UNet_Nair/log_metrics_diffs.txt", "w")
file_loss.close()
################################
def write_to_file(input): 
    with open("/mnt/nfs/efernandez/projects/UNet_Nair/log_metrics_diffs.txt", "a") as textfile: 
        textfile.write(str(input) + "\n") 
    textfile.close()

def make_pixel_grid_from_pos(x_pos, z_pos):
    zz, xx = np.meshgrid(z_pos, x_pos, indexing="ij") # 'ij' -> rows: z, columns: x
    yy = xx * 0
    grid = np.stack((xx, yy, zz), axis=-1)  # [nrows, ncols, 3]
    return grid

def main():

    # field names
    fields = ['#','id', 'r', 'cx', 'cz', 'c', 
              'das_contrast', 'das_cnr', 'das_gcnr', 'das_snr', 'das_decay', 'das_contrast_att',
              'diff_contrast', 'diff_cnr', 'diff_gcnr', 'diff_snr', 'diff_decay', 'diff_contrast_att',
              'diff2_contrast', 'diff2_cnr', 'diff2_gcnr', 'diff2_snr', 'diff2_decay', 'diff2_contrast_att']

    rows = []

    das_met=[]
    diff_met=[]
    diff2_met=[]

    das_contrast=[]
    das_cnr=[]
    das_gcnr=[]
    das_snr=[]
    das_decay=[]
    das_contrast_att=[]

    diff_contrast=[]
    diff_cnr=[]
    diff_gcnr=[]
    diff_snr=[]
    diff_decay=[]
    diff_contrast_att=[]

    diff2_contrast=[]
    diff2_cnr=[]
    diff2_gcnr=[]
    diff2_snr=[]
    diff2_decay=[]
    diff2_contrast_att=[]

    num_samples = 500
    n_sample = 0

    sim_dir = '/mnt/nfs/isalazar/datasets/simulatedCystDataset/raw_0.0Att/'
    att_dir = '/mnt/nfs/isalazar/datasets/simulatedCystDataset/raw_0.5Att/'

    # Get the list of all files and directories
    path = '/mnt/nfs/efernandez/datasets/dataRF/RF_test/'
    # path = '/TESIS/DATOS_1/rf_test/'
    dir_test = sorted(os.listdir(path))

    # for simu in range(1,num_samples+1):
    #     simu_name = "simu" + str(simu).zfill(5)
    #     filename=simu_name+".npy"

    for simu in dir_test:
        simu_name = simu[:-4]
        filename=simu_name+".npy"

        write_to_file('File')
        write_to_file(filename)
        write_to_file('---')

        sub_row = []

        P = LoadData_nair2020(h5_dir=sim_dir,
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
        number_columns = 30
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

        # write_to_file(region)

        #testing model DAS
        test_DAS = '/mnt/nfs/efernandez/generated_samples/DAS/gen_test/'
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
        # write_to_file('DAS: ' + str(contrast_att))

        #testing model udiff
        dir_model_udiff = '/mnt/nfs/efernandez/generated_samples/DDPM_model/v6_TT_100steps/300epoch/gen_test/'
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
        # write_to_file('DIFF: ' + str(contrast_att))

        #testing model diff2
        dir_model_diff2 = '/mnt/nfs/efernandez/generated_samples/DDPM_model/final_model_FT/gen_test/'
        bmode_output = np.load(dir_model_diff2+filename).squeeze()
        # bmode_output = (bmode_output + 1) * 30 - 60
        contrast, cnr, gcnr, snr, decay_param, contrast_att = compute_metrics(cx, cz, r, bmode_output, grid, region)
        sub_row.append(contrast)
        sub_row.append(cnr)
        sub_row.append(gcnr)
        sub_row.append(snr)
        sub_row.append(decay_param)
        sub_row.append(contrast_att)
        diff2_contrast.append(contrast)
        diff2_cnr.append(cnr)
        diff2_gcnr.append(gcnr)
        diff2_snr.append(snr)
        diff2_decay.append(decay_param)
        diff2_contrast_att.append(contrast_att)

        rows.append(sub_row)
        n_sample = n_sample +1

    das_met.append(das_contrast)
    das_met.append(das_cnr)
    das_met.append(das_gcnr)
    das_met.append(das_snr)
    das_met.append(das_decay)
    das_met.append(das_contrast_att)

    diff_met.append(diff_contrast)
    diff_met.append(diff_cnr)
    diff_met.append(diff_gcnr)
    diff_met.append(diff_snr)
    diff_met.append(diff_decay)
    diff_met.append(diff_contrast_att)

    diff2_met.append(diff2_contrast)
    diff2_met.append(diff2_cnr)
    diff2_met.append(diff2_gcnr)
    diff2_met.append(diff2_snr)
    diff2_met.append(diff2_decay)
    diff2_met.append(diff2_contrast_att)

    save_dir='/mnt/nfs/efernandez/generated_samples/mets/test_TESIS_doc3'

    np.save(save_dir+"/met_das_test_TESIS_v1.npy", np.array(das_met))

    np.save(save_dir+"/met_diff_test_TESIS_v1.npy", np.array(diff_met))

    np.save(save_dir+"/met_diff2_test_TESIS_v1.npy", np.array(diff2_met))
    
    # name of csv file
    filename = '/mnt/nfs/efernandez/generated_samples/mets/test_TESIS_doc3.csv'
 
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