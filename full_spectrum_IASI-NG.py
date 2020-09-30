#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Source code of the article
M. Colom, J.M. Morel (2019). Full-Spectrum Denoising of High-SNR Hyperspectral Images.
Journal of the Optical Society of America A. 36, 450-463.
doi:10.1364/JOSAA.36.000450

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.


Please cite the article is you use this code. Bibtex:
    @article{Colom:19,
    author = {Miguel Colom and Jean-Michel Morel},
    journal = {J. Opt. Soc. Am. A},
    keywords = {Blackbodies; Hyperspectral imaging; Image processing algorithms; Material characterization; Multispectral imaging; Wavelet transforms},
    number = {3},
    pages = {450--463},
    publisher = {OSA},
    title = {Full-spectrum denoising of high-SNR hyperspectral images},
    volume = {36},
    month = {Mar},
    year = {2019},
    url = {http://josaa.osa.org/abstract.cfm?URI=josaa-36-3-450},
    doi = {10.1364/JOSAA.36.000450},
    abstract = {The high spectral redundancy of hyper/ultraspectral Earth-observation satellite imaging raises three challenges: (a)\&\#x00A0;to design accurate noise estimation methods, (b)\&\#x00A0;to denoise images with very high signal-to-noise ratio (SNR), and (c)\&\#x00A0;to secure unbiased denoising. We solve (a)\&\#x00A0;by a new noise estimation, (b)\&\#x00A0;by a novel Bayesian algorithm exploiting spectral redundancy and spectral clustering, and (c)\&\#x00A0;by accurate measurements of the interchannel correlation after denoising. We demonstrate the effectiveness of our method on two ultraspectral Earth imagers, IASI and IASI-NG, one flying and the other in project, and sketch the major resolution gain of future instruments entailed by such unbiased denoising.},
    }

(c) 2019 Miguel Colom
http://mcolom.info
"""

from netCDF4 import Dataset
import argparse
import numpy as np
from functions import *
import numpy.linalg as linalg
from scipy import interpolate
from scipy.cluster.vq import kmeans, vq, whiten
from sklearn.decomposition import PCA
import scipy.fftpack as fftpack
import dtcwt
import csv
import os
import sys


def tam(M, name):
    '''
    DEBUG: print out the size of a vector or matrix
    '''
    S = M.shape
    if len(S) == 1:
        print "Vector {} has {} elements".format(name, S[0])
    else:
        print "Matrix {} is {}x{}: {} rows and {} columns".format(name, S[0],S[1], S[0], S[1])

def MSE(I, K):
    '''
    Compute the Mean Squared Error (MSE)
    '''
    return np.mean((I - K)**2.0)

def RMSE(I, K):
    '''
    Compute the square Root of the Mean Squared Error (RMSE)
    '''
    return np.sqrt(MSE(I, K))

def PSNR(I, K):
    '''
    Compute the Peak Signal-to-Noise Ration (PSNR)
    '''
    max_I = np.max(I)
    return 10.0 * np.log10(max_I**2.0 / MSE(I, K))
    
def MSNR_freq(I, K, f):
    '''
    Compute Median Signal-to-Noise Ratio (MSNR)
    '''
    median_I = np.median(I[f,:])
    return 10.0 * np.log10(median_I**2.0 / MSE(I[f,:], K[f,:]))
    
def print_metadata(dataset):
    '''
    Print out metadata information associated to the granule
    '''
    print "- Source: %s" % dataset.source
    print "- Title: %s" % dataset.title
    print "- Data model: %s" % dataset.data_model
    
    print "- Dimensions:"
    for key in dataset.dimensions.keys():
        d = dataset.dimensions[key]
        print "\t%s;\tsize=%d" % ((d.name, len(d)))
        
    print "- Institution: %s" % dataset.institution
    print "- Orbit: %s" % dataset.orbit


def read_noise_model(filename):
    '''
    Read the CSV file containing the noise model
    '''
    f = open(filename, "r")
    r = csv.reader(f, delimiter=";")
    
    # NEdL, sigma
    data = []
    
    for row in r:
        entry = map(np.float, row)
        data.append(entry)
    
    f.close()

    return data

def get_interpolation_function(csv_filename):
    """
    Obtain a 1D interpolation function for the noise
    """
    noise_model = read_noise_model(csv_filename) # NEdL, sigma
    
    nedl = np.zeros(len(noise_model))
    sigma_nedl = np.zeros(len(noise_model))

    for i in range(len(noise_model)):
        nedl[i] = noise_model[i][0]
        sigma_nedl[i] = noise_model[i][1]
    
    return interpolate.interp1d(nedl, sigma_nedl)
    

def put_same_mean(target, ref):
    """
    Make 'target' have the same mean as 'ref'
    """
    for i in range(ref.shape[1]):
        mean_ref = np.mean(ref[:, i])
        mean_target = np.mean(target[:, i])
        #
        target[:, i] *= (mean_ref / mean_target)
    return target

    
def print_evaluation(S_name, N, denoised, band, band_noisy):
    """
    Print denoising evaluation
    """
    print "N={}, using {}".format(N, S_name)

    MSE_noisy_ref = MSE(band_noisy, band[:,:])
    MSE_denoised_ref = MSE(denoised, band[:,:])
    PSNR_ref_noisy = PSNR(band[:,:], band_noisy)
    PSNR_ref_denoised = PSNR(band[:,:], denoised)
    
    print "MSE(band_noisy, band)={}".format(MSE_noisy_ref)
    print "MSE(denoised, band)={}".format(MSE_denoised_ref)
    print "PSNR(band, band_noisy)={}".format(PSNR_ref_noisy)
    print "PSNR(band, denoised)={}".format(PSNR_ref_denoised)
    print "Ratio MSE: {}".format(MSE_noisy_ref / MSE_denoised_ref)
    print "Gain PSNR: {}".format(PSNR_ref_denoised - PSNR_ref_noisy)

def read_granule_all_bands(dataset):
    """
    Read IASI-NG hyperspectral image
    """
    num_bands = 0
    while 'spectrum_band%d' % (num_bands+1) in dataset.variables.keys():
        num_bands += 1
            
    # Get the number of frequencies
    num_freqs = 0
    for band_num in range(1,num_bands+1):
        band = dataset.variables["spectrum_band%d" % band_num][:-1, :] # spectrum band, without last freq (duplicated)
        num_freqs += band.shape[0]

    # Read all bands
    granule = np.zeros((num_freqs, band.shape[1]))
    freqs = np.zeros(num_freqs)
    
    f = 0
    for band_num in range(1,num_bands+1):
        band = dataset.variables["spectrum_band%d" % band_num][:-1, :] # spectrum band, without last freq (duplicated)
        freqs_band = dataset.variables['wavenumber_band%d' % band_num][:-1] # without last (duplicated)
        num_freqs_band = band.shape[0]
        granule[f:f+num_freqs_band, :] = band[:,:]
        freqs[f:f+num_freqs_band] = freqs_band

        f += num_freqs_band
    
    return granule, freqs

def store_indexed(local_from, global_to, indices_loc2global_dict):
    """
    Decode and unscramble the IASI-NG data
    """
    for lf in range(local_from.shape[0]):
        gf = indices_loc2global_dict[lf]
        global_to[gf, :] += local_from[lf, :]

    
def write_plot(filename, granule_denoised, granule):
    """
    Write a log10 plot of two vectors
    """
    f = open(filename, "w")
    for i in range(granule.shape[0]):
        string = "{} {}\n".format(i, np.log10(MSE(granule_denoised[i,:], granule[i,:])))
        #string = "{} {}\n".format(i, MSE(granule_dimred_denoised[i,:], granule[i,:]))

        f.write(string)
    f.close()
    
def write_plot_div(filename, granule_denoised, granule_dimred_denoised, granule):
    """
    Write a log10 plot of the difference between two vectors
    """
    f = open(filename, "w")
    for i in range(granule.shape[0]):
        string = "{} {}\n".format( \
          i, MSE(granule_dimred_denoised[i,:], granule[i,:]) / MSE(granule_denoised[i,:],granule[i,:]))
        f.write(string)
    f.close()


def write_plot_MSNR(filename, granule_denoised, granule):
    """
    Write a MSNR plot between a denoised and a reference hyperspectral images
    """
    f = open(filename, "w")
    for i in range(granule.shape[0]):
        string = "{} {}\n".format( \
          i, MSNR_freq(granule, granule_denoised, i))
        f.write(string)
    f.close()
    
def write_plot_diff_MSNR(filename, granule_dimred_denoised, granule_denoised, granule):
    """
    Write a MSNR plot between a denoised and a reference hyperspectral images (difference)
    """
    f = open(filename, "w")
    for i in range(granule.shape[0]):
        
        if MSNR_freq(granule, granule_denoised, i) - MSNR_freq(granule, granule_dimred_denoised, i) > 8:
            extra = "  #*****"
        else:
            extra = ""
        
        string = "{} {} {}\n".format( \
          i, MSNR_freq(granule, granule_denoised, i) - MSNR_freq(granule, granule_dimred_denoised, i), extra)
        f.write(string)
    f.close()
    
def write_plot_freqs(filename, freqs):
    """
    Write the frequencies plot
    """
    f = open(filename, "w")
    for i in range(len(freqs)):
        string = "{} {}\n".format(i, freqs[i])
        f.write(string)
    f.close()

def get_correlation(Rn, Rd):
    """
    Get the correlation between the noisy and denoised hyperspectral images
    """
    D = Rn - Rd

    Rcorr = np.corrcoef(D)
    
    # Don't consider the diagonal
    for i in range(Rcorr.shape[0]):
        Rcorr[i,i] = 999
    
    Rcorr = Rcorr.flatten()
    idx = np.where(Rcorr != 999)
    
    return np.std(Rcorr[idx]), np.mean(Rcorr[idx]), Rcorr


def plot_autocorrelation_histo(RF, title, filename):
    """
    Plot the histogram of autocorrelations
    """
    L = int(np.sqrt(len(RF)))
    R = np.reshape(RF, (L, L))

    plt.ioff()
    
    fig = plt.figure()
    fig.suptitle(title)
    
    #plt.imshow(R, cmap='gray')

    hrange=[-1.0, 1.0]
    plt.hist(RF, 300000, normed=False, color='green', histtype='step')
    plt.xlim(hrange[0], hrange[1])
    plt.ylim(0, 4.5e6)
    plt.grid(True)
    
    fig.savefig(filename)
    plt.close()


def get_SDV(dataset, granule_noisy, freqs, map_1d_to_2d):
    """
    Read the hyperspectral bands of the input image
    """
    num_bands = 0
    while 'spectrum_band%d' % (num_bands+1) in dataset.variables.keys():
        num_bands += 1
            
    # Get the number of frequencies
    num_freqs = 0
    for band_num in range(1,num_bands+1):
        band = dataset.variables["spectrum_band%d" % band_num][:-1, :] # spectrum band, without last freq (duplicated)
        num_freqs += band.shape[0]

    # Read all bands
    granule = np.zeros((num_freqs, band.shape[1]))
    freqs = np.zeros(num_freqs)
    
    f = 0
    for band_num in range(1,num_bands+1):
        band = dataset.variables["spectrum_band%d" % band_num][:-1, :] # spectrum band, without last freq (duplicated)
        freqs_band = dataset.variables['wavenumber_band%d' % band_num][:-1] # without last (duplicated)
        num_freqs_band = band.shape[0]
        granule[f:f+num_freqs_band, :] = band[:,:]
        freqs[f:f+num_freqs_band] = freqs_band

        f += num_freqs_band
    
    # IASI-NG: (40, 56, 16920)
    Nz = 16920
    assert(granule_noisy.shape[0] == Nz)
    
    SDV = np.zeros((56, 40, Nz))   # SDV pines: (145, 145, 200)
    for z in range(Nz):
        for i in range(56*40):
            x, y = map_1d_to_2d[i]
            SDV[x, y, z] = granule_noisy[z, i]
    
    return SDV
    

#################################################

# Read arguments
parser = argparse.ArgumentParser()
parser.add_argument("granule_num")
parser.add_argument("Q", default=3)
parser.add_argument("N", default=20)
parser.add_argument("K", default=400)
parser.parse_args()
args = parser.parse_args()

granule_num = int(args.granule_num)
Q = int(args.Q) # number of frequential clusters
N = int(args.N)
K = int(args.K)

print "granule #{}, Q={}, N={}, K={}".format(granule_num, Q, N, K)

# Obtain the SDV: (120, 23, 8461)
# IASI-NG: (40, 56, 16920)
# It needs to be spatially decoded
filename = "../gi_precomp/decoded_granule{}.npy".format(granule_num)
filename_nc = '../granule{}.nc'.format(granule_num)
dataset = Dataset(filename_nc, 'r')
_, freqs = read_granule_all_bands(dataset)
SDV = np.load(filename)

assert(SDV.shape == (56, 40, 16920))

num_bands = 4

granule = SDV.reshape(56*40, 16920) # (2240, 16920)
granule = granule.T
assert(granule.shape == (16920, 2240))

# Get number of freqs per band
num_freqs_band = []
for i in range(num_bands):
    num_freqs_band.append(len(dataset.variables["wavenumber_band%d" % (i+1)])-1) # remove last (duplicated)
    
# Add noise according to the model
granule_noisy = np.zeros(granule.shape)

function_inter_nedl = get_interpolation_function("../Bruit.csv")

np.random.seed(1234) # Deterministic noise, for the comparisons
for i in range(granule.shape[0]):
    sigma = function_inter_nedl(freqs[i])
    granule_noisy[i,:] = granule[i,:] + np.random.normal(loc=0.0, scale=sigma, size=granule.shape[1])

np.random.seed(12345)

# N: number of PCs per band
# K: we considerer the K most simlar pixeles, excluding Pn itself

# The result is written here
granule_denoised = np.zeros(granule.shape)


# Process each band
for band_num in range(4):
    print "Band: {}".format(band_num)
    
    # Indices de inicio y fin para la banda    
    start = int(np.sum(num_freqs_band[0:band_num]))
    end = int(np.sum(num_freqs_band[0:band_num+1]))
    
    band_noisy = granule_noisy[start:end]
    band = granule[start:end]
    freqs_band = freqs[start:end]

    # Clustering
    data_whiten = whiten(band_noisy)
    codebook, distortion = kmeans(data_whiten, Q)
    code, _ = vq(data_whiten, codebook)

    # Process each frequency cluster
    for label in range(Q):
        print "Label: {}".format(label)

        F = len(code[code == label]) # number of frequencies

        # Noise band, noise-free band, and frequencies,
        # for the freqs. in the cluster with that label
        cluster_noisy = np.zeros((F, band_noisy.shape[1]))
        cluster = np.zeros((F, band.shape[1]))
        freqs_cluster = np.zeros(F)

        count = 0
        indices_loc2global_dict = {}
        for f in range(band_noisy.shape[0]):
            if code[f] == label:
                cluster_noisy[count,:] = band_noisy[f,:]
                cluster[count,:] = band[f,:]
                freqs_cluster[count] = freqs_band[f]
                indices_loc2global_dict[count] = f
                count += 1
        assert(count == F)

        # Obtain the sigmas of band_noisy
        sigmas = np.zeros(F)
        for i in range(F):
            sigmas[i] = function_inter_nedl(freqs_cluster[i])

        # Divide by the STD of the noise, so it has STD=1
        Nz = cluster_noisy.shape[0]
        for z in range(Nz):
            if sigmas[z] > 0:
                cluster_noisy[z,:] /= sigmas[z]
                cluster[z,:] /= sigmas[z]

        # PCA of the noisy band
        pca = PCA(n_components=cluster_noisy.shape[0], whiten=False)
        S = pca.fit_transform(cluster_noisy[:,:].T).T
        W = pca.components_


        # Compute the covariance matrix of the noise
        noise = cluster_noisy - cluster
        
        Cn = np.zeros(2*[min(cluster_noisy.shape)])
        #
        Sn = pca.transform(noise.T).T # (100, 21025)
        Wn = pca.components_ # (100, 200)
        #
        for i in range(Cn.shape[0]):
            Cn[i,i] = np.var(Sn[i,:])
        #
        Cn_firstN = Cn[0:N, 0:N]
        
        sigmas_pca = np.zeros(Cn.shape[0])
        for z in range(Cn.shape[0]):
            sigmas_pca[z] = np.sqrt(np.abs(Cn[z,z]))


        ### Denoising
        S2 = S[0:N, :]
        S2_denoised = np.zeros(S2.shape)

        # Correlation matrix, to choose the most similar pixels
        Rp = np.corrcoef(S.T)

        S_denoised = np.zeros(S.shape)
            
        for pixel_index in range(S.shape[1]):
            Pn = S2[:, pixel_index] # Pixel ruidoso
            
            ## Denoise Pn
            sim_idx = np.argsort(Rp[pixel_index])[::-1] # Indices of the most similar pixels
            
            # Mean of the similar pixles, including Pn
            Pn_mean = np.mean(S2[:, sim_idx[0:K+1]], axis=1)
            
            # Covariance matrix of the similar pixels
            Cpn = np.cov(S2[:, sim_idx[1:K+1]]) # Los K Pn mas similares, excluyendo el propio Pn
            Cpn_inv = np.linalg.inv(Cpn)

            # Denoising
            P = Pn_mean + np.dot(Cpn - Cn_firstN, Cpn_inv).dot(Pn - Pn_mean)                
            S2_denoised[:, pixel_index] = P



        # Copy S2_denoised into S_denoised
        S_denoised[0:N, :] = S2_denoised[:,:]
        
        
        
        ############################################################
        # DTCWT denoising of the less important PCs
        ############################################################
        transform = dtcwt.Transform2d()

        # 3D-DCT of the less-important PC's, from index N
        S2 = S[N:, :]

        denoise_low = True

        # SDV: (56, 40, 16920)
        S2_denoised = np.zeros((S2.shape[0], SDV.shape[0]*SDV.shape[1]))
        LEVELS = 4

        for z in range(S2.shape[0]):
            image = S2[z, :].reshape((SDV.shape[0:2]))

            pyramid = transform.forward(image, nlevels=LEVELS)

            # Compute LEVELS levels of dtcwt with the defaul wavelet family
            for level in range(LEVELS):
                D = pyramid.highpasses[level][:,:,:]
                D2 =  np.abs(D**2.0)
                M2 =  (np.roll(D2[:,:,:], 0, axis=1) + np.roll(D2[:,:,:], 1, axis=1) + np.roll(D2[:,:,:], -1, axis=1)) / 3.0

                thr = sigmas_pca[z]*np.sqrt(2*np.log(np.prod(pyramid.highpasses[level][:,:,:].shape))) # * (2**(level+1 - LEVELS/float(2)))
                thr2 = thr**2.0

                # Soft thresholding
                A =  1.0 - thr2 / M2
                A = A.clip(min=0)
                #
                pyramid.highpasses[level][:,:,:] *= A        
                
                # Low frequencies
                if denoise_low:
                    thr = sigmas_pca[z]*np.sqrt(2*np.log(np.prod(pyramid.lowpass[:,:].shape)))
                    thr2 = thr**2.0

                    D2 = np.abs(pyramid.lowpass[:,:])**2.0 + 1e-70
                    M2 =  (np.roll(D2[:,:], 0, axis=1) + np.roll(D2[:,:], 1, axis=1) + np.roll(D2[:,:], -1, axis=1)) / 3.0
                    
                    A =  1.0 - thr2 / M2
                    A = A.clip(min=0)

                    pyramid.lowpass[:,:] *= A
                    
                image_denoised = transform.inverse(pyramid, gain_mask=None)[0:SDV.shape[0], 0:SDV.shape[1]]
                
                S2_denoised[z, :] = image_denoised.flatten()
                
        # Copy S2_denoised into S
        S_denoised[N:, :] = S2_denoised[:,:]


        ### Un-project
        cluster_denoised = pca.inverse_transform(S_denoised.T).T

        # Undo division by STD
        for z in range(Nz):
            if sigmas[z] > 0:
                cluster_denoised[z,:] *= sigmas[z]

        store_indexed(cluster_denoised, granule_denoised[start:end], indices_loc2global_dict)


# ==============================================================
# ==============================================================


# Weighted average so the variance of the noise matches the model's
# D --> alpha*D + (1-alpha) * S
#   D: denoised
#   S: noisy
noise   = granule_noisy - granule
removed = granule_noisy - granule_denoised

for i in range(granule.shape[0]):
    alpha2 = np.var(noise[i,:], ddof=1) / (np.var(granule_noisy[i,:], ddof=1)  + np.var(granule_denoised[i,:], ddof=1) - 2*np.cov(granule_noisy[i,:], granule_denoised[i,:])[0,1])
    assert(alpha2 >= 0)

    alpha = np.sqrt(alpha2)
    granule_denoised[i,:] = alpha*granule_denoised[i,:] + (1.0-alpha)*granule_noisy[i,:]

# Save denoised
filename = "gi_precomp/granule{}_denoised_Q{}_N{}_K{}.npy".format(granule_num, Q, N, K)
np.save(filename, granule_denoised)

# Save MSNR plot
write_plot_MSNR("granule{}_Q{}_N{}_K{}_MSNR-BBD.txt".format(granule_num, Q, N, K), granule_denoised, granule)
print "RMSE={}, noisy={}".format(np.sqrt(np.mean((granule - granule_denoised)**2.0)),  np.sqrt(np.mean((granule - granule_noisy)**2.0)))
