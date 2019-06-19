#!/usr/bin/env python
import os
import sys
import h5py
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from datetime import datetime
data_dir='ATL06/Byrd_glacier_rel001/'

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


# make sure we're dealing with the most recent version of any code we're using
#%load_ext autoreload
#%autoreload 2


def ATL06_to_dict(filename, dataset_dict):
    """
        Read selected datasets from an ATL06 file

        Input arguments:
            filename: ATl06 file to read
            dataset_dict: A dictinary describing the fields to be read
                    keys give the group names to be read, 
                    entries are lists of datasets within the groups
        Output argument:
            D6: dictionary containing ATL06 data.  Each dataset in 
                dataset_dict has its own entry in D6.  Each dataset 
                in D6 contains a list of numpy arrays containing the 
                data
    """
    
    D6=[]
    pairs=[1, 2, 3]
    beams=['l','r']
    gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
    gps_epoch_ts = gps_epoch.timestamp()
    # open the HDF5 file
    with h5py.File(filename) as h5f:
        # loop over beam pairs
        file_epoch_time = np.array(h5f['ancillary_data']['atlas_sdp_gps_epoch']) + gps_epoch_ts
        for pair in pairs:
            # loop over beams
            for beam_ind, beam in enumerate(beams):
                # check if a beam exists, if not, skip it
                if '/gt%d%s/land_ice_segments' % (pair, beam) not in h5f:
                    continue
                # loop over the groups in the dataset dictionary
                temp={}
                for group in dataset_dict.keys():
                    for dataset in dataset_dict[group]:
                        DS='/gt%d%s/%s/%s' % (pair, beam, group, dataset)
                        # since a dataset may not exist in a file, we're going to try to read it, and if it doesn't work, we'll move on to the next:
                        try:
                            temp[dataset]=np.array(h5f[DS])
                            if dataset == "delta_time":
                                temp[dataset] = temp[dataset] + file_epoch_time
                            # some parameters have a _FillValue attribute.  If it exists, use it to identify bad values, and set them to np.NaN
                            if '_FillValue' in h5f[DS].attrs:
                                fill_value=h5f[DS].attrs['_FillValue']
                                bad = temp[dataset] == fill_value
                                temp[dataset] = np.float64(temp[dataset])
                                temp[dataset][temp[dataset]==fill_value]=np.NaN
                        except KeyError as e:
                            pass
                if len(temp) > 0:
                    # it's sometimes convenient to have the beam and the pair as part of the output data structure: This is how we put them there.
                    temp['pair']=np.zeros_like(temp['h_li'])+pair
                    temp['beam']=np.zeros_like(temp['h_li'])+beam_ind
                    temp['filename']=filename
                    D6.append(temp)
    return D6

def get_velocity(d6):
    pass

def get_rema_elev(d6):
    pass
    
if __name__ == "__main__":
    lineno=898
    fn = "ben-data.h5" # file name for the line
    dataset_dict={'land_ice_segments':['h_li', 'delta_time','longitude','latitude'], 'land_ice_segments/ground_track':['x_atc']}
    # read ATL06 into a dictionary (the ATL06 file has the same name as the ATL03 file, except for the product name)
    
    D6_list=ATL06_to_dict(fn, dataset_dict)

    # pick out gt1r:
    D6 = D6_list[1]
    print(datetime.utcfromtimestamp(D6['delta_time'][0]))

    f1,ax = plt.subplots(num=1,figsize=(10,6))
    ax.plot(D6['x_atc'], D6['h_li'],'r.', markersize=2, label='ATL06')
    lgd = ax.legend(loc=3,frameon=False)

    ax.set_xlabel('x_atc, m')
    ax.set_ylabel('h, m')
    plt.savefig('thw0.png')
    
    vels = get_velocity(D6)
    rema_elev = get_rema_elev(D6)
    
    