from datetime import datetime, timedelta
import glob
import os

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import RectBivariateSpline
from tqdm import tqdm
from scipy.ndimage import zoom

from zarr_file import ZarrFile

def fill_gaps(array, mask, distance=15):
    """ Fill gaps in input raster

    Parameters
    ----------
    array : 2D numpy.array
        Ratser with deformation field
    mask : 2D numpy.array
        Where are gaps
    distance : int
        Minimum size of gap to fill

    Returns
    -------
    arra : 2D numpy.array
        Ratser with gaps filled

    """
    dist, indi = distance_transform_edt(
        mask,
        return_distances=True,
        return_indices=True)
    gpi = dist <= distance
    r,c = indi[:,gpi]
    new_array = np.array(array)
    new_array[gpi] = array[r,c]
    return new_array

def load_data(sid_file, mask, xc, yc):
    """ Load ice drift data and create interpolators

    Parameters
    ----------
    sid_file : str
        Path to ice drift file
    mask : 2D numpy.array
        Land/sea mask
    xc : 1D numpy.array
        X coordinates
    yc : 1D numpy.array
        Y coordinates
    
    Returns
    -------
    u_int : RectBivariateSpline
        Interpolator for u ice drift component
    v_int : RectBivariateSpline
        Interpolator for v ice drift component
    c_int : RectBivariateSpline
        Interpolator for ice concentration

    """
    with np.load(sid_file) as data:
        u = data['u']
        v = - data['v']
        c = data['c']
    u = fill_gaps(u, np.isnan(u), distance=30)
    v = fill_gaps(v, np.isnan(v), distance=30)
    c = fill_gaps(c, np.isnan(c), distance=30)
    u[mask[1:-1:3, 1:-1:3] == 0] = 0
    v[mask[1:-1:3, 1:-1:3] == 0] = 0
    c[mask == 0] = 0

    u_int = RectBivariateSpline(yc[-1:1:-3], xc[1:-1:3], u[::-1], kx=1, ky=1)
    v_int = RectBivariateSpline(yc[-1:1:-3], xc[1:-1:3], v[::-1], kx=1, ky=1)
    c_int = RectBivariateSpline(yc[-1:1:-3], xc[1:-1:3], c[-1:1:-3, 1:-1:3], kx=1, ky=1)
    return u_int, v_int, c_int

def load_init_data(mesh_init_file, zoom_factor, resolution_factor):
    """ Load initial mesh data

    Parameters
    ----------
    mesh_init_file : str
        Path to initial mesh file
    zoom_factor : float
        Zoom factor for mesh grid

    Returns
    -------
    xc : 1D numpy.array
        X coordinates
    yc : 1D numpy.array
        Y coordinates
    mask : 2D numpy.array
        Land/sea mask
    x_grid : 2D numpy.array
        X mesh grid
    y_grid : 2D numpy.array
        Y mesh grid
    y_size : int
        Size of Y mesh grid
    x_size : int
        Size of X mesh grid
    m_int : RectBivariateSpline
        Interpolator for land/sea mask
    """
    with np.load(mesh_init_file) as data:
        xc = data['xc']
        yc = data['yc'][::-1]
        mask = data['mask']
    x_grid, y_grid = np.meshgrid(xc, yc)
    x_grid = zoom(x_grid, zoom_factor, order=1)
    y_grid = zoom(y_grid, zoom_factor, order=1)
    x_grid_age = zoom(x_grid, resolution_factor, order=1)
    y_grid_age = zoom(y_grid, resolution_factor, order=1)
    m_int = RectBivariateSpline(yc[::-1], xc, mask[::-1], kx=1, ky=1)
    return xc, yc, mask, x_grid, y_grid, x_grid_age, y_grid_age, m_int

def get_input_files(sid_dir, out_dir, restart):
    """ Get input files and initial data
    
    Parameters
    ----------
    sid_dir : str
        Path to ice drift files
    out_dir : str
        Path to output files
    restart : bool
        Whether to restart from scratch or continue from last output

    Returns
    -------
    sid_dates_files : list of tuples
        List of (date, ice drift file) tuples to process
    data0 : dict
        Initial data for advection

    """
    sid_files = sorted(glob.glob(f'{sid_dir}/*/ice_drift_nh_ease2*.nc.npz'))
    sid_dates = [datetime.strptime(f.split('-')[-1], '%Y%m%d1200.nc.npz') for f in sid_files]
    out_files = sorted(glob.glob(f'{out_dir}/*/*/newdc_age_*.zip'))
    out_dates = [datetime.strptime(f.split('_')[-1], '%Y%m%d.zip') for f in out_files]

    if restart:
        data0 = {}
        sid_dates_files = [(d,f) for (d,f) in zip(sid_dates, sid_files) if d >= datetime(1991,9,15)]
    else:
        data0 = ZarrFile(out_files[-1]).load()
        sid_dates_files = [(d,f) for (d,f) in zip(sid_dates, sid_files) if d >= max(out_dates)]
        
    return sid_dates_files, data0

def move_points(sid_dates_files, data0, mask, xc, yc, x_grid, y_grid, x_grid_age, y_grid_age, m_int, max_conc, max_fractions, file_to_process):
    """ Move points according to ice drift data and save points and rasterized ice age into Zarr files """
    for sid_date, sid_file in tqdm(sid_dates_files[:file_to_process]):
        # Load data
        u_int, v_int, c_int = load_data(sid_file, mask, xc, yc)

        # Init MYI
        if sid_date.month == 9 and sid_date.day == 15:
            # init new X, Y if it 15th Sept
            c_grid = c_int(y_grid, x_grid, grid=False)
            x0 = x_grid[c_grid > max_conc]
            y0 = y_grid[c_grid > max_conc]
            data0[str(sid_date.year)] = np.vstack([x0, y0])

        # Advect
        data1 = {}
        # select only max_fractions youngest fractions
        years = sorted([k for k in data0.keys() if k != 'age'], reverse=True)[:max_fractions]
        for year in years:
            x0, y0 = data0[str(year)]
            if x0 is None or y0 is None:
                continue
            # loop over input Xs,Ys
            u = u_int(y0, x0, grid=False)
            v = v_int(y0, x0, grid=False)

            x1 = x0 + u
            y1 = y0 + v
            m1 = m_int(y1, x1, grid=False)
            c1 = c_int(y1, x1, grid=False)
            x1 = x1[(m1 > 0) & (c1 > max_conc)]
            y1 = y1[(m1 > 0) & (c1 > max_conc)]
            if len(x1) > 10:
                data1[str(year)] = np.vstack([x1, y1]).astype(np.float32)

        # Calculate age
        age = np.zeros((np.array(x_grid_age.shape)).astype(int))
        c_grid = c_int(y_grid_age, x_grid_age, grid=False)
        age[c_grid > max_conc] = 1 # first year ice
        for i, year in enumerate(years):
            x1, y1 = data1[year]
            cols = np.round(x_grid_age.shape[1] * (x1 - xc.min()) / (xc.max() - xc.min())).astype(int)
            rows = np.round(x_grid_age.shape[0] * (y1 - yc.max()) / (yc.min() - yc.max())).astype(int)
            age[rows, cols] = i + 2 #second YI, third YI, etc...
        
        # Save advected points and age
        if data1:
            dst_date = sid_date + timedelta(1)
            dst_dir = dst_date.strftime('outputs/%Y/%m')
            os.makedirs(dst_dir, exist_ok=True)
            dst_file = dst_date.strftime(f'{dst_dir}/newdc_age_%Y%m%d.zip')
            data1['age'] = age.astype(np.int8)
            mf = ZarrFile(dst_file)
            mf.save(data1)
            if file_to_process:
                print(f'Save: {dst_file}')

        # Prepare for next day
        data0 = data1


if __name__ == '__main__':
    mesh_init_file = '../sea_ice_age/mesh_arctic_ease_25km_max7.npz'
    #zoom_factor = 3
    #resolution_factor = 0.666666667
    zoom_factor = 2
    resolution_factor = 1.0

    max_conc = 15
    restart = False
    sid_dir = './OSISAF_ice_drift_CDR_postproc'
    out_dir = './outputs'
    max_fractions = 10
    file_to_process = None  # set to an integer to process only a limited number of files

    xc, yc, mask, x_grid, y_grid, x_grid_age, y_grid_age, m_int = load_init_data(mesh_init_file, zoom_factor, resolution_factor)

    sid_dates_files, data0 = get_input_files(sid_dir, out_dir, restart)

    #original resolution, seeding resolution, output resolution
    print((xc.max() - xc.min()) / xc.size, (xc.max() - xc.min()) / x_grid.shape[1], (xc.max() - xc.min()) / x_grid_age.shape[1])

    move_points(sid_dates_files, data0, mask, xc, yc, x_grid, y_grid, x_grid_age, y_grid_age, m_int, max_conc, max_fractions, file_to_process)  