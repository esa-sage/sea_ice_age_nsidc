import glob
from datetime import datetime
import os

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from scipy.ndimage import zoom

from zarr_file import ZarrFile

class Collocator:
    def __init__(self, rrdp_template_file, lmsiage_file, lmsiage_dir, newdc_dir, output_dir):
        self.rrdp_template_file = rrdp_template_file
        self.lmsiage_file = lmsiage_file
        self.lmsiage_dir = lmsiage_dir
        self.newdc_dir = newdc_dir
        self.output_dir = output_dir

    def get_lmsiage_dates(self):
        """ Get available LM-SIAge dates from files in the LM-SIAge directory """
        lmsiage_files = sorted(glob.glob(f'{self.lmsiage_dir}/*/grid_*.zip'))
        lmsiage_dates = [datetime.strptime(f.split('_')[-1], '%Y%m%d.zip') for f in lmsiage_files]
        return lmsiage_dates

    def read_template_file(self):
        """ Read RRDP template file to get destination grid """
        with xr.open_dataset(self.rrdp_template_file) as ds:
            self.dst_x = ds['xc'].values
            self.dst_y = ds['yc'].values
            self.NIC_myi_conc = ds['NIC_myi_conc'][0].values
        self.dst_grids = np.array(np.meshgrid(self.dst_y[::-1], self.dst_x)).T.reshape(-1, 2)

    def resample_lmsiage_masks(self):
        """ Read LM-SIAge grids and resample LM-SIAge masks to destination grid """
        with np.load(self.lmsiage_file) as data:
            self.lmsiage_xc = data['xc'] * 1000
            self.lmsiage_yc = data['yc'] * 1000
            mask = data['mask']
            landmask = data['landmask']
        self.lmsiage_xc2 = zoom(self.lmsiage_xc, 2, order=1)
        self.lmsiage_yc2 = zoom(self.lmsiage_yc, 2, order=1)
        self.masks_dst = []
        for a in [mask, landmask]:
            i = RectBivariateSpline(self.lmsiage_yc, self.lmsiage_xc, a, kx=1, ky=1)
            a_dst = i(self.dst_y[::-1], self.dst_x, grid=True) > 0.5
            self.masks_dst.append(np.ma.masked_array(a_dst, mask=a_dst == 0))
        self.mask_dst, self.landmask_ldst = self.masks_dst

    def get_lmsiage_data(self, lmsiage_file):
        """ Get LM-SIAge data and resample to destination grid """
        mf = ZarrFile(lmsiage_file)
        names = mf.read_names()
        sic_names = sorted([n for n in names if 'sic' in n])
        data = mf.load(sic_names)
        fyi_conc = data['sic_1yi']
        fyi_conc[np.isnan(fyi_conc)] = 0
        syi_conc = data['sic_2yi']
        syi_conc[np.isnan(syi_conc)] = 0
        myi_conc = np.nansum([data[n] for n in sic_names if n > 'sic_1yi'], axis=0)
        ice_type = np.zeros_like(fyi_conc)
        ice_type[fyi_conc > 15] = 2
        ice_type[myi_conc > 15] = 4
        ice_type[syi_conc > 15] = 3
        dst = []
        for a in [fyi_conc, syi_conc, myi_conc, ice_type]:
            rgi = RegularGridInterpolator((self.lmsiage_yc, self.lmsiage_xc), a, method='nearest', bounds_error=False, fill_value=0)
            a_dst = rgi(self.dst_grids).reshape(self.dst_y.size, self.dst_x.size)
            a_dst[(self.landmask_ldst == 1) | (self.mask_dst == 0)] = -1
            dst.append(a_dst)
        return dst

    def get_newdc_data(self, newdc_file):
        """ Get NewDC data and resample to destination grid """
        age, = ZarrFile(newdc_file).load(['age'], as_dict=False)
        ice_type = np.zeros_like(age)
        ice_type[age == 1] = 2
        ice_type[age == 2] = 3
        ice_type[age >= 3] = 4
        rgi = RegularGridInterpolator((self.lmsiage_yc2, self.lmsiage_xc2), ice_type, method='nearest', bounds_error=False, fill_value=0)
        ice_type_dst = rgi(self.dst_grids).reshape(self.dst_y.size, self.dst_x.size)
        ice_type_dst[self.landmask_ldst == 1] = -1
        return ice_type_dst

    def __call__(self, date):
        """ Collocate LM-SIAge and NewDC data for a given date and save to netCDF file """
        lmsiage_file = date.strftime(f'{self.lmsiage_dir}/%Y/grid_%Y%m%d.zip')
        newdc_file = date.strftime(f'{self.newdc_dir}/%Y/%m/newdc_age_%Y%m%d.zip')
        if not os.path.exists(lmsiage_file) or not os.path.exists(newdc_file):
            print(f'Files for date {date.strftime("%Y-%m-%d")} not found, skipping.')
            return

        lm_fyi_conc, lm_syi_conc, lm_myi_conc, lm_ice_type = self.get_lmsiage_data(lmsiage_file)
        newdc_ice_type = self.get_newdc_data(newdc_file)

        lm_fyi_conc_out = np.round(lm_fyi_conc).astype(np.int8)[None]
        lm_syi_conc_out = np.round(lm_syi_conc).astype(np.int8)[None]
        lm_myi_conc_out = np.round(lm_myi_conc).astype(np.int8)[None]
        lm_ice_type_out = lm_ice_type.astype(np.int8)[None]
        newdc_ice_type_out = newdc_ice_type.astype(np.int8)[None]

        # Create xarray dataset with the data variables
        common_ice_type_attrs = dict(
            flag_values = [np.byte(1), np.byte(2), np.byte(3), np.byte(4)],
            flag_meanings = "first_year_ice second_year_ice multi_year_ice ice_type"
        )
        ds = xr.Dataset(
            {
                'lm_fyi': (['time', 'y', 'x'], lm_fyi_conc_out, {'_FillValue': np.int8(-1), 'long_name': 'LM-SIAge First Year Ice Concentration', 'units': '%'}),
                'lm_syi': (['time', 'y', 'x'], lm_syi_conc_out, {'_FillValue': np.int8(-1), 'long_name': 'LM-SIAge Second Year Ice Concentration', 'units': '%'}),
                'lm_myi': (['time', 'y', 'x'], lm_myi_conc_out, {'_FillValue': np.int8(-1), 'long_name': 'LM-SIAge Multi Year Ice Concentration', 'units': '%'}),
                'lm_ice_type': (['time', 'y', 'x'], lm_ice_type_out, {'_FillValue': np.int8(-1), 'long_name': 'LM-SIAge dominant ice type', **common_ice_type_attrs}),
                'newdc_ice_type': (['time', 'y', 'x'], newdc_ice_type_out, {'_FillValue': np.int8(-1), 'long_name': 'NewDC dominant ice type', **common_ice_type_attrs})
            },
            coords={
                'y': self.dst_y,
                'x': self.dst_x,
                'time': np.array([np.datetime64(date)])
            }
        )
        out_dir = date.strftime(f'{self.output_dir}/%Y')
        os.makedirs(out_dir, exist_ok=True)
        ds.to_netcdf(f'{out_dir}/lagrangian_{date.strftime("%Y%m%d")}.nc')


if __name__ == '__main__':
    rrdp_template_file = '../sage/20240223_N.nc'
    lmsiage_file = '../sea_ice_age/mesh_arctic_ease_25km_max7.npz'
    lmsiage_dir = 'grid'
    newdc_dir = 'outputs'
    output_dir = './collocated/'

    collocator = Collocator(rrdp_template_file=rrdp_template_file, lmsiage_file=lmsiage_file, lmsiage_dir=lmsiage_dir, newdc_dir=newdc_dir, output_dir=output_dir)
    collocator.read_template_file()
    collocator.resample_lmsiage_masks()
    lmsiage_dates = collocator.get_lmsiage_dates()
    for date in lmsiage_dates:
        collocator(date)
