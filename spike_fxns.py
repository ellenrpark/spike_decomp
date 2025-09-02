#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 13:37:54 2025

@author: epark
"""


import numpy as np
import pandas as pd
import glob
import numpy.matlib
import gsw
from scipy import ndimage
import math

def QC_BBP700_data(pick):
    """
    Parameters
    -------
    pick : output from EuroArgo RTQC bbp toolbox

    Returns
    ------
    bbp_qcd: quality-controlled backscatter data (N_PROF x N_LEVELS)
    pres: pressure measurements (dbar) for bbp measurements (N_PROF x N_LEVELS)
    depth: depth measurements (m) for bbp measurements (N_PROF x N_LEVELS)
    juld: date (N_PROF, )
    lat: latitude (N_PROF, )
    lon: longitude (N_PROF, )
    """
    # Quality control data
    # Get largest dimension
    lat = np.zeros(len(pick)-1)*np.nan
    lon = np.zeros(len(pick)-1)*np.nan
    juld = (np.zeros(len(pick)-1)*np.nan).astype('datetime64[s]')

    # Get lat, lon, date values from pickle
    maxd = 0
    for i in np.arange(len(pick)-1):
        lat[i] = pick[i]['LAT']
        lon[i] = pick[i]['LON']
        juld[i] = pick[i]['JULD']

        if pick[i]['BBP700'].shape[0]>maxd:
            maxd = pick[i]['BBP700'].shape[0]

    # Format bbp and depth data into arrays of same size
    pres = np.zeros((len(pick)-1, maxd))*np.nan
    bbp = np.zeros((len(pick)-1, maxd))*np.nan
    bbp_qc_flags = np.zeros((len(pick)-1, maxd))*np.nan

    for i in np.arange(len(pick)-1):
        eind = pick[i]['BBP700'].shape[0]

        pres[i,:eind] = pick[i]['PRES']
        bbp[i,:eind] = pick[i]['BBP700']
        bbp_qc_flags[i,:eind] = pick[i]['BBP700_QC_flag']
        
    # Quality control data
    bbp_qcd = np.where((bbp_qc_flags == 1) | (bbp_qc_flags == 2) | (bbp_qc_flags == 5) | (bbp_qc_flags == 8), bbp, np.nan)
    lat_n = numpy.matlib.repmat(lat.reshape(lat.shape[0],1),
                              1, pres.shape[1])

    # Calculate depth from pressure
    depth = gsw.z_from_p(pres, lat_n)*-1

    return bbp_qcd, pres, depth, juld, lat, lon

def BriggsDecomposition(bbp_qcd, depth):
    """
    Decomposition of particulate backscatter signal from:
    Major role of particle fragmentation in regulating biological sequestration of CO2 by the oceans
    Nathan Briggs, Giorgio Dall'Olmo, Henrvé Claustre, Science, 2020

    Parameters
    -----
    bbp_qcd: quality-controlled backscatter data (N_PROF x N_LEVELS)
    depth: depth measurements for bbp data
    

    Returns 
    -----
    briggs_values: a dictionary containing values from spike decomp.
        "bbl": optical backscattering coefficient due to "large" particles
        "bbs": optical backscattering coefficient due to small, labile particles
        "bbr": optical backscattering coefficient due to refractory particles, includes instrument noise (referred to as blank)
        "noise": instrument noise component, here the deep blank

    References
    -----
    Briggs, N., Dall’Olmo, G., & Claustre, H. (2020). 
    Major role of particle fragmentation in regulating biologicalsequestration of CO 2 by the oceans.
    Science,367(6479), 791–793. https://doi.org/10.1126/science.aay1790 
    """

    # First apply, 11-point minimum filter
    window = 11
    bbp_min = ndimage.minimum_filter(bbp_qcd, size=[1,window])
    # Followed by 11-point maximum filter
    bbp_filtered = ndimage.maximum_filter(bbp_min, size=[1,window])

    # residual "spikes": difference between unfiltered and filtered data (bbl+instrument noise)
    residual_spikes = bbp_qcd-bbp_filtered
    residual_spikes = np.where(residual_spikes==0, np.nan, residual_spikes)

    # Calculate instrument noise component
    # For each profile, bin data below 300m into 50m bins
    dmin = 300; dmax = 2000
    d_bins = np.arange(300,2050, 50)
    d_inds = np.where((depth>=dmin) & (depth<dmax))
    
    nprof = numpy.matlib.repmat(np.arange(depth.shape[0]).reshape(depth.shape[0],1),
                              1, depth.shape[1])
    noise_df = pd.DataFrame({'DBIN': np.digitize(depth[d_inds], d_bins), 
                             'NPROF':nprof[d_inds], 
                             'RES_SPIKES':residual_spikes[d_inds]})
    
    noise_df = noise_df.dropna()
    # Get mean value of each 50m depth bin by profiles
    mean_df =noise_df.groupby(['NPROF','DBIN']).mean()
    noise_mean = np.zeros((depth.shape[0],d_bins.shape[0]))*np.nan
    for i in mean_df.index:
        noise_mean[i]=mean_df.loc[i]

    noise_med = np.nanmedian(noise_mean)
    
    # Remove bins that are greater than twice the median
    noise_med_clean = np.where(noise_mean>2*noise_med,np.nan,noise_mean)
    
    # Calculate median: instrument blank
    blank_briggs = np.nanmedian(noise_med_clean)

    # Add blank to filtered profiles to get bbsr
    # bbsr: bbs + bbr
    bbsr_briggs = bbp_filtered + blank_briggs
    bbl_briggs = bbp_qcd - bbsr_briggs
    bbl_briggs = np.where(bbl_briggs==0, np.nan, bbl_briggs)

    # Estimate bbr as the 25th percentile of bbsr from 850 to 900m
    bbr_briggs = np.nanpercentile(np.where(np.logical_and(depth>=850, depth<=900),bbsr_briggs, np.nan),25)
    bbs_briggs = bbsr_briggs-bbr_briggs # Assuming bbr is constant across both depth and time

    ## Make blank and bbr an array
    blank_briggs = np.ones(bbl_briggs.shape)*blank_briggs
    bbr_briggs = np.ones(bbl_briggs.shape)*bbr_briggs

    briggs_values = {'bbl': bbl_briggs,
                     'bbs': bbs_briggs,
                     'bbr': bbr_briggs,
                     'noise': blank_briggs}

    return briggs_values

def BriggsDecomposition_Modified(bbp_qcd, depth, window =7, 
                                 method = 'quantile',quantile = 0.7, threshold = 1e-5):
    """
    A modified versioin of the decomposition of particulate backscatter signal from:
    Major role of particle fragmentation in regulating biological sequestration of CO2 by the oceans
    Nathan Briggs, Giorgio Dall'Olmo, Henrvé Claustre, Science, 2020

    Parameters
    -----
    bbp_qcd: quality-controlled backscatter data (N_PROF x N_LEVELS)
    depth: depth measurements for bbp data
    window: window size for min/max filtering
    method: method used for isolating large particles from instrument noise
        - 'quantile': use a quantile, where any residual spikes greater than this quantile are large particles,
                        while those that are smaller can be considered noise
        - 'threshold': using a fixed backscatter values instead of a quantile

    Returns 
    -----
    new_values: a dictionary containing values from spike decomp.
        "bbl": optical backscattering coefficient due to "large" particles
        "bbs": optical backscattering coefficient due to small, labile particles
        "bbr": optical backscattering coefficient due to refractory particles, includes instrument noise (referred to as blank)
        "noise": instrument noise component, here a profile that is different from the deep blank

    References
    -----
    Briggs, N., Dall’Olmo, G., & Claustre, H. (2020). 
    Major role of particle fragmentation in regulating biologicalsequestration of CO 2 by the oceans.
    Science,367(6479), 791–793. https://doi.org/10.1126/science.aay1790 
    """
    
    # First apply, 11-point minimum filter
    bbp_min = ndimage.minimum_filter(bbp_qcd, size=[1,window])
    # Followed by 11-point maximum filter
    bbp_filtered = ndimage.maximum_filter(bbp_min, size=[1,window])

    # residual "spikes": difference between unfiltered and filtered data (bbl+instrument noise)
    residual_spikes = bbp_qcd-bbp_filtered
    
    residual_spikes = np.where(residual_spikes==0, np.nan, residual_spikes)


    # Profile noise
    noise_med = np.nanmedian(residual_spikes, axis=1)

    noise = np.zeros(residual_spikes.shape)*np.nan
    for i in np.arange(residual_spikes.shape[0]):

        # Remove spikes that are greater than 2 times the median noise for a given profile
        inds = np.where(residual_spikes[i,:]>2*noise_med[i])

        # Use remaining smaller spikes to determine threshold for
        # large particles vs. instrument noise profile
        noise_slice = residual_spikes[i,:].copy()
        noise_slice[inds]=np.nan

        # Determine threshold to separate instrument noise
        if method == 'quantile':
            thresh = np.nanquantile(noise_slice, quantile)
        elif method == 'threshold':
            thresh = threshold

        # Assign instrument noise profile
        noise[i,:]=np.where(residual_spikes[i,:]>thresh,np.nan, residual_spikes[i,:])

    # Replace NaN with zeros
    # This value is the combination of the deep blank from Briggs et al., 2020
    # plus the instrument noise component
    blank_noise_new = np.where(np.isnan(noise), 0, noise)
       
    
    bbsr_new = bbp_filtered+blank_noise_new
    bbl_new = bbp_qcd - bbsr_new
    bbl_new = np.where(bbl_new==0, np.nan, bbl_new)
    
    # Decompose blank and noise
    blank_noise_new = np.where(blank_noise_new==0, np.nan, blank_noise_new)
    blank_new = np.matlib.repmat(np.nanmean(blank_noise_new, axis = 1).reshape(-1,1),1,depth.shape[1])
    noise_new = np.where(np.isnan(blank_noise_new-blank_new), 0, blank_noise_new-blank_new)
    
    bbsr_new = bbsr_new-noise_new

    # Calculate bbr in the same way as Briggs 2020 but on a profile
    # by profile basis
    if depth.shape[0] == 0:
        max_depth = np.inf
    else:
        max_depth = np.nanmax(depth)
    bbr_new = np.nanpercentile(np.where(np.logical_and(depth>=850, depth<=max_depth),bbsr_new, np.nan),
                               25,
                               axis = 1)
    # Reformat to an array
    bbr_new = np.matlib.repmat(bbr_new.reshape(-1,1), 1, bbsr_new.shape[-1])
    bbs_new = bbsr_new - bbr_new
    
    new_values = {'bbl': bbl_new,
                  'bbs': bbs_new,
                  'bbr': bbr_new,
                  'noise': noise_new}
    
    return new_values

def bbp_decomposition(bbp_qcd, depth):
    
    briggs_values = BriggsDecomposition(bbp_qcd, depth)
    new_values = BriggsDecomposition_Modified(bbp_qcd, depth)
    
    return briggs_values, new_values
    