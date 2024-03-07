import numpy as np 
import fitsio as fio 
import os
import treecorr

band = 'H'
if os.path.exists('/hpc/group/cosmology/phy-lsst/my137/imcom/out/summary_statistics/StarCat_'+band+'_sample.fits'):
    d = fio.read('/hpc/group/cosmology/phy-lsst/my137/imcom/out/summary_statistics/StarCat_'+band+'_sample.fits')
else:
    d_type = [('ra', 'f8'), ('dec', 'f8'), ('bind_x', 'f8'), ('bind_y', 'f8'), ('x_out', 'f8'), ('y_out', 'f8'), ('xI_out', 'f8'), ('yI_out', 'f8'), ('dx', 'f8'), ('dy', 'f8'), ('amp_hsm_galsim', 'f8'), ('dx_hsm_galsim', 'f8'), ('dy_hsm_galsim', 'f8'), ('sig_hsm_galsim', 'f8'), ('g1_hsm_galsim', 'f8'), ('g2_hsm_galsim', 'f8'), ('amp_hsm_crout', 'f8'), ('dx_hsm_crout', 'f8'), ('dy_hsm_crout', 'f8'), ('sig_hsm_crout', 'f8'), ('g1_hsm_crout', 'f8'), ('g2_hsm_crout', 'f8'), ('mean_fid', 'f8'), ('nepoch', 'f8'), ('g1_noise_white', 'f8'), ('g2_noise_white', 'f8'), ('s2n_white', 'f8'), ('g1_noise_1f', 'f8'), ('g2_noise_1f', 'f8'), ('s2n_1f', 'f8')]
    d = np.loadtxt('/hpc/group/cosmology/phy-lsst/my137/imcom/out/summary_statistics/StarCat_'+band+'.txt', dtype=d_type)
    fio.write('/hpc/group/cosmology/phy-lsst/my137/imcom/out/summary_statistics/StarCat_'+band+'_sample.fits', d)
print('number of objects', len(d))

def _find_psi(ra, dec, ra_ctr, dec_ctr):
    
    ra = np.radians(ra)
    dec = np.radians(dec)
    ra_ctr = np.radians(ra_ctr)
    dec_ctr = np.radians(dec_ctr)

    zeta = np.arctan(np.cos(dec)*np.sin(ra-ra_ctr) / (-np.sin(dec)*np.cos(dec_ctr) + np.sin(dec_ctr)*np.cos(dec)*np.cos(ra-ra_ctr)))
    eta = np.arctan(np.cos(dec_ctr)*np.sin(ra_ctr-ra) / (-np.sin(dec_ctr)*np.cos(dec) + np.sin(dec)*np.cos(dec_ctr)*np.cos(ra-ra_ctr)))

    psi = eta - zeta
    psi_r = psi - np.pi*np.round(psi/np.pi)
    return psi_r

def _compute_treecorr_angle(g1, g2, psi):

    e1 = np.cos(2*psi)*g1 + np.sin(2*psi)*g2
    e2 = -np.sin(2*psi)*g1 + np.cos(2*psi)*g2

    return e1, e2


def _compute_GG_corr(ra, dec, g1, g2, out_f, xy=False):

    bin_config = dict(
        sep_units = 'arcmin',
        bin_slop = 0.1,

        min_sep = 0.5,
        max_sep = 80,
        nbins = 15,

        var_method = 'jackknife'
    )

    # shear-shear
    f_pc = '/hpc/group/cosmology/masaya/imcom_phase1/patch_center.txt'
    gg = treecorr.GGCorrelation(bin_config)
    if xy:
        cat = treecorr.Catalog(x=xy[0], y=xy[1], x_units='arcsec', y_units='arcsec', g1=g1, g2=g2, patch_centers=f_pc)
    else:
        cat = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', g1=g1, g2=g2, patch_centers=f_pc)

    gg.process(cat)
    gg.write(out_f)

def _compute_NG_corr(ra, dec, g1, g2, out_f):

    bin_config = dict(
        sep_units = 'arcmin',
        bin_slop = 0.1,

        min_sep = 0.5,
        max_sep = 80,
        nbins = 15,

        var_method = 'jackknife'
    )

    # count-shear
    f_pc = '/hpc/group/cosmology/masaya/imcom_phase1/patch_center.txt'
    ng = treecorr.NGCorrelation(bin_config)
    cat1 = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', patch_centers=f_pc)
    cat2 = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', g1=g1, g2=g2, patch_centers=f_pc)
    ng.process(cat1, cat2)
    ng.write(out_f)

def _compute_NK_corr(ra, dec, kappa, out_f):

    bin_config = dict(
        sep_units = 'arcmin',
        bin_slop = 0.1,

        min_sep = 0.5,
        max_sep = 80,
        nbins = 15,

        var_method = 'jackknife'
    )

    # count-kappa
    f_pc = '/hpc/group/cosmology/masaya/imcom_phase1/patch_center.txt'
    nk = treecorr.NKCorrelation(bin_config)
    cat1 = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', patch_centers=f_pc)
    cat2 = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', k=kappa, patch_centers=f_pc)
    nk.process(cat1, cat2)
    nk.write(out_f)


sph_correction = False
if sph_correction:
    ra_ctr = 53.000000
    dec_ctr = -40.000000
    psi = _find_psi(d['ra'], d['dec'], ra_ctr, dec_ctr)
    g1, g2 = _compute_treecorr_angle(d['g1_hsm_galsim'], d['g2_hsm_galsim'], psi)

# 2pcf
_compute_GG_corr(d['ra'], d['dec'], d['g1_hsm_galsim'], d['g2_hsm_galsim'], '/hpc/group/cosmology/phy-lsst/my137/imcom/out/summary_statistics/star_'+band+'_shear-shear_galsim_sample.fits')
_compute_NG_corr(d['ra'], d['dec'], d['g1_hsm_galsim'], d['g2_hsm_galsim'], '/hpc/group/cosmology/phy-lsst/my137/imcom/out/summary_statistics/star_'+band+'_sky-shear_galsim_sample.fits')

for fid_cut in [40, 45, 50]:
    print('fidelity cut: ', fid_cut)
    d = d[d['mean_fid'] > fid_cut]

    _compute_GG_corr(d['ra'], d['dec'], d['g1_hsm_galsim'], d['g2_hsm_galsim'], '/hpc/group/cosmology/phy-lsst/my137/imcom/out/summary_statistics/star_'+band+'_shear-shear_galsim_sample_fid'+str(fid_cut)+'.fits') #, xy=[d['x_out'], d['y_out']])
    _compute_NG_corr(d['ra'], d['dec'], d['g1_hsm_galsim'], d['g2_hsm_galsim'], '/hpc/group/cosmology/phy-lsst/my137/imcom/out/summary_statistics/star_'+band+'_sky-shear_galsim_sample_fid'+str(fid_cut)+'.fits')
    _compute_NK_corr(d['ra'], d['dec'], d['sig_hsm_galsim'], '/hpc/group/cosmology/phy-lsst/my137/imcom/out/summary_statistics/star_'+band+'_sky-sigma_galsim_sample_fid'+str(fid_cut)+'.fits')


# noise bias (white noise)
mean_noise_g1 = np.mean(d['g1_noise_white']/d['s2n_white']**2)
mean_noise_g2 = np.mean(d['g2_noise_white']/d['s2n_white']**2)
err_noise_g1 = np.std(d['g1_noise_white']/d['s2n_white']**2)/np.sqrt(len(d['g1_noise_white']))
err_noise_g2 = np.std(d['g2_noise_white']/d['s2n_white']**2)/np.sqrt(len(d['g2_noise_white']))
print('noise bias g1: ', '{:.7f}'.format(mean_noise_g1), '+/-',  '{:.7f}'.format(err_noise_g1))
print('noise bias g2: ', '{:.7f}'.format(mean_noise_g2), '+/-',  '{:.7f}'.format(err_noise_g2))

_compute_GG_corr(d['ra'], d['dec'], d['g1_noise_white']/d['s2n_white']**2, d['g2_noise_white']/d['s2n_white']**2, '/hpc/group/cosmology/phy-lsst/my137/imcom/out/summary_statistics/star_'+band+'_noise-noise_galsim_sample_whitenoise.fits')
_compute_NK_corr(d['ra'], d['dec'], d['g1_noise_white']/d['s2n_white']**2, '/hpc/group/cosmology/phy-lsst/my137/imcom/out/summary_statistics/star_'+band+'_sky-noise_galsim_sample_g1_whitenoise.fits')
_compute_NK_corr(d['ra'], d['dec'], d['g2_noise_white']/d['s2n_white']**2, '/hpc/group/cosmology/phy-lsst/my137/imcom/out/summary_statistics/star_'+band+'_sky-noise_galsim_sample_g2_whitenoise.fits')

# noise bias (1/f noise)
mean_noise_g1 = np.mean(d['g1_noise_1f']/d['s2n_1f']**2)
mean_noise_g2 = np.mean(d['g2_noise_1f']/d['s2n_1f']**2)
err_noise_g1 = np.std(d['g1_noise_1f']/d['s2n_1f']**2)/np.sqrt(len(d['g1_noise_1f']))
err_noise_g2 = np.std(d['g2_noise_1f']/d['s2n_1f']**2)/np.sqrt(len(d['g2_noise_1f']))
print('noise bias g1: ', '{:.7f}'.format(mean_noise_g1), '+/-',  '{:.7f}'.format(err_noise_g1))
print('noise bias g2: ', '{:.7f}'.format(mean_noise_g2), '+/-',  '{:.7f}'.format(err_noise_g2))

_compute_GG_corr(d['ra'], d['dec'], d['g1_noise_1f']/d['s2n_1f']**2, d['g2_noise_1f']/d['s2n_1f']**2, '/hpc/group/cosmology/phy-lsst/my137/imcom/out/summary_statistics/star_'+band+'_noise-noise_galsim_sample_1fnoise.fits')
_compute_NK_corr(d['ra'], d['dec'], d['g1_noise_1f']/d['s2n_1f']**2, '/hpc/group/cosmology/phy-lsst/my137/imcom/out/summary_statistics/star_'+band+'_sky-noise_galsim_sample_g1_1fnoise.fits')
_compute_NK_corr(d['ra'], d['dec'], d['g2_noise_1f']/d['s2n_1f']**2, '/hpc/group/cosmology/phy-lsst/my137/imcom/out/summary_statistics/star_'+band+'_sky-noise_galsim_sample_g2_1fnoise.fits')