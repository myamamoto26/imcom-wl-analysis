import ngmix
import galsim
import fitsio as fio
import numpy as np
from astropy.io import fits
from astropy import wcs
import healpy as hp
from tqdm import tqdm
import os,sys
import pickle

"""NOTES
- Injest correct weight map for galaxy and PSF image. -> Chris prefers using constant weight during the object fit measurement (non-constant weight could result in biases at different profiles of galaxies)
- Make sure the jacobians are correct. Be careful about the centroid and offset. 
- Make sure the noise applied to the images is the right noise. "fixnoise" operation (add noise that is rotated by 90 deg) assumes original noise image is uncorrelated. However, in imcom the noise between pixels is correlated. We need to think whether "fixnoise" procedure is the correct approach here. 
"""


# IMCOM params
TESTNAME = 'C'
NBLOCKS = 36
TOTAL_BLOCKS = NBLOCKS * NBLOCKS
SCALE = 0.0390625 # 0.025
SCA_NSIDE = 4096
N = 2560
BD_GAL = 140
BD_STAR = 50
HPRES = 14
RS = 1./np.sqrt(2.)/60.*np.pi/180*1.08333
SNR = 10.
AREA = {'Y': 7.06, 'J': 8.60, 'H': 10.96, 'F': 15.28}#, 'K': , 'W':}

def get_truth_coord(row, ibx, inpath, filtername):

    """
    Grab truth coordinates of where injected stars and galaxies are drawn.
    """

    iby = row[-2:]
    # infile = os.path.join(inpath, row+'/full_inpad1.24_smfwhm2.00-Y_{:02d}_{:02d}_map.fits'.format(int(ibx),int(iby)))
    infile = os.path.join(inpath, row+'/prod_'+filtername+'_{:02d}_{:02d}_map.fits'.format(int(ibx),int(iby)))
    fhist = np.zeros((61,),dtype=np.uint32)
    with fits.open(infile) as f:
        mywcs = wcs.WCS(f[0].header)

    # Get galaxy centroid from healpy.
    ra_cent, dec_cent = mywcs.all_pix2world([(N-1)/2], [(N-1)/2], [0.], [0.], 0, ra_dec_order=True)
    ra_cent = ra_cent[0]; dec_cent = dec_cent[0]
    vec = hp.ang2vec(ra_cent, dec_cent, lonlat=True)
    qp = hp.query_disc(2**HPRES, vec, RS, nest=False)
    ra_hpix, dec_hpix = hp.pix2ang(2**HPRES, qp, nest=False, lonlat=True)
    npix = len(ra_hpix)

    x, y, z1, z2 = mywcs.all_world2pix(ra_hpix, dec_hpix, np.zeros((npix,)), np.zeros((npix,)), 0)
    xi = np.rint(x).astype(np.int16); yi = np.rint(y).astype(np.int16)
    grp = np.where(np.logical_and(np.logical_and(xi>=BD_GAL,xi<N-BD_GAL),np.logical_and(yi>=BD_GAL,yi<N-BD_GAL)))
    ra_hpix = ra_hpix[grp]
    dec_hpix = dec_hpix[grp]
    x = x[grp]
    y = y[grp]
    npix = len(x)

    xi = np.rint(x).astype(np.int16)
    yi = np.rint(y).astype(np.int16)

    return x, y, xi, yi, npix


def get_layers(row, ibx, inpath,filtername):

    """
    Grab injected source layers and white noise layer from IMCOM output. 
    """
    
    iby = row[-2:]
    # infile = os.path.join(inpath, row+'/full_inpad1.24_smfwhm2.00-Y_{:02d}_{:02d}_map.fits'.format(int(ibx),int(iby)))
    infile = os.path.join(inpath, row+'/prod_'+filtername+'_{:02d}_{:02d}_map.fits'.format(int(ibx),int(iby)))
    fhist = np.zeros((61,),dtype=np.uint32)
    with fits.open(infile) as f:
        mywcs = wcs.WCS(f[0].header)
        wt = np.rint(1./np.amax(f['INWEIGHT'].data[0,:,:,:]+1e-6, axis=0))
        fmap = f['FIDELITY'].data[0,:,:].astype(np.float32)
        for fy in range(61): fhist[fy] += np.count_nonzero(f['FIDELITY'].data[0,100:-100,100:-100]==fy)
        map_simple = f[0].data[0,0,:,:]
        # map_truth = f[0].data[0,1,:,:]
        map_injstar = f[0].data[0,2,:,:]
        map_injext = f[0].data[0,6,:,:]; map_injext_g0 = f[0].data[0,7,:,:]; map_injext_g60 = f[0].data[0,8,:,:]; map_injext_g120 = f[0].data[0,9,:,:]; 
        WNmap = f[0].data[0,11,:,:]
    
    return map_injstar, map_injext, map_injext_g0, map_injext_g60, map_injext_g120, WNmap


def make_obs(rng, noise, im, psf_im, dx, dy):

    """
    Make Observation objects for galaxy and PSF to be input in ngmix. 
    """
    
    # psf_noise = 1e-6
    psf_im_ = psf_im # + psf_noise # rng.normal(scale=psf_noise, size=psf_im.shape)
    im_ = im + noise # rng.normal(scale=noise, size=im.shape)
    
    cen = (np.array(im_.shape)-1.0)/2.0 # depending on the convention, +1.0 could be -1.0. 
    psf_cen = (np.array(psf_im_.shape)-1.0)/2.0

    jacobian = ngmix.DiagonalJacobian(
        row=cen[0] + dy, col=cen[1] + dx, scale=SCALE,
    )
    psf_jacobian = ngmix.DiagonalJacobian(
        row=psf_cen[0] + dy, col=psf_cen[1] + dx, scale=SCALE,
    )

    wt = np.ones_like(im_) # im_*0 + 1.0/noise**2
    # psf_wt = np.ones_like(psf_im_)

    psf_obs = ngmix.Observation(
        psf_im_,
        # weight = psf_wt,
        jacobian=psf_jacobian,
    )

    obs = ngmix.Observation(
        im_,
        weight=wt,
        jacobian=jacobian,
        psf=psf_obs,
    )
    
    return obs


def make_data(rng, noise, shear):

    psf_noise = 1.0e-6

    scale = SCALE

    psf_fwhm = 0.9
    gal_hlr = 0.5
    dy, dx = rng.uniform(low=-scale/2, high=scale/2, size=2)

    psf = galsim.Moffat(
        beta=2.5, fwhm=psf_fwhm,
    ).shear(
        g1=0.02,
        g2=-0.01,
    )

    obj0 = galsim.Exponential(
        half_light_radius=gal_hlr,
    ).shear(
        g1=shear[0],
        g2=shear[1],
    ).shift(
        dx=dx,
        dy=dy,
    )

    obj = galsim.Convolve(psf, obj0)

    psf_im = psf.drawImage(scale=scale).array
    im = obj.drawImage(scale=scale).array

    psf_im += rng.normal(scale=psf_noise, size=psf_im.shape)
    im += rng.normal(scale=noise, size=im.shape)

    cen = (np.array(im.shape)-1.0)/2.0
    psf_cen = (np.array(psf_im.shape)-1.0)/2.0

    jacobian = ngmix.DiagonalJacobian(
        row=cen[0] + dy/scale, col=cen[1] + dx/scale, scale=scale,
    )
    psf_jacobian = ngmix.DiagonalJacobian(
        row=psf_cen[0], col=psf_cen[1], scale=scale,
    )

    wt = im*0 + 1.0/noise**2
    psf_wt = psf_im*0 + 1.0/psf_noise**2

    psf_obs = ngmix.Observation(
        psf_im,
        weight=psf_wt,
        jacobian=psf_jacobian,
    )

    obs = ngmix.Observation(
        im,
        weight=wt,
        jacobian=jacobian,
        psf=psf_obs,
    )

    return obs


def make_struct(res, obs, shear_type):

    """
    Make structured output from the shape measurement on metacal output.
    """

    dt = [
        ('flags', 'i4'),
        ('shear_type', 'U7'),
        ('s2n', 'f8'),
        ('g', 'f8', 2),
        ('T', 'f8'),
        ('Tpsf', 'f8'),
    ]
    data = np.zeros(1, dtype=dt)
    data['shear_type'] = shear_type
    data['flags'] = res['flags']
    if res['flags'] == 0:
        data['s2n'] = res['s2n']
        # for moments we are actually measureing e, the elliptity
        data['g'] = res['e']
        data['T'] = res['T']
    else:
        data['s2n'] = np.nan
        data['g'] = np.nan
        data['T'] = np.nan
        data['Tpsf'] = np.nan

        # we only have one epoch and band, so we can get the psf T from the
        # observation rather than averaging over epochs/bands
        data['Tpsf'] = obs.psf.meta['result']['T']

    return data


def run_mcal(ngal, rng, noise, shear):

    mcal_keys = ['noshear', '1p', '1m', '2p', '2m']
        
    weight_fwhm = 1.2
    fitter = ngmix.gaussmom.GaussMom(fwhm=weight_fwhm)
    psf_fitter = ngmix.gaussmom.GaussMom(fwhm=weight_fwhm)

    psf_runner = ngmix.runners.PSFRunner(fitter=psf_fitter)
    runner = ngmix.runners.Runner(fitter=fitter)

    boot = ngmix.metacal.MetacalBootstrapper(
            runner=runner, psf_runner=psf_runner,
            rng=rng,
            psf='gauss', # re-convolution PSF
            types=mcal_keys,
        )

    outlist = []
    for i in tqdm(range(ngal)):
        obs = make_data(rng, noise, shear)
        resdict, obsdict = boot.go(obs)
    
        if resdict['noshear']['flags']!=0:
                print('non-zero flags objects: ' , k, ', flags=', resdict['noshear']['flags'])
        for k in mcal_keys:
            st = make_struct(res=resdict[k], obs=obsdict[k], shear_type=k)
            outlist.append(st)
            
    data = np.hstack(outlist)

    return data


def run_mcal_grid(npix, map_ext, map_star, map_noise, x, xi, y, yi, rng, filtername):

    """
    Run metacalibration and shape measurement on all objects in the extended source layer.
    """

    # set up for ngmix. Measure with Gaussian weighted moments. 
    mcal_keys = ['noshear', '1p', '1m', '2p', '2m']
        
    weight_fwhm = 1.2
    fitter = ngmix.gaussmom.GaussMom(fwhm=weight_fwhm)
    psf_fitter = ngmix.gaussmom.GaussMom(fwhm=weight_fwhm)

    psf_runner = ngmix.runners.PSFRunner(fitter=psf_fitter)
    runner = ngmix.runners.Runner(fitter=fitter)

    boot = ngmix.metacal.MetacalBootstrapper(
            runner=runner, psf_runner=psf_runner,
            rng=rng,
            psf='gauss', # re-convolution PSF
            types=mcal_keys,
        )

    
    outlist = []
    for k in range(npix):
        im = map_ext[yi[k]+1-BD_GAL:yi[k]+BD_GAL,xi[k]+1-BD_GAL:xi[k]+BD_GAL] * 10**6 # scale the flux. 
        psf_im = map_star[yi[k]+1-BD_STAR:yi[k]+BD_STAR,xi[k]+1-BD_STAR:xi[k]+BD_STAR]
        noiseimage = map_noise[yi[k]+1-BD_GAL:yi[k]+BD_GAL,xi[k]+1-BD_GAL:xi[k]+BD_GAL]/SNR/np.sqrt(AREA[filtername[0]])
        # psfnoiseimage = WNmap[yi[k]+1-BD_STAR:yi[k]+BD_STAR,xi[k]+1-BD_STAR:xi[k]+BD_STAR]/SNR/np.sqrt(AREA[filtername[0]])
        dx = x[k] - xi[k]; dy = y[k] - yi[k]
        # weight = wt[yi[k]//BD_GAL,xi[k]//BD_GAL] TO-DO: input correct weight. 
        obs = make_obs(rng, noiseimage, im, psf_im, dx, dy)
        resdict, obsdict = boot.go(obs)
    
        if resdict['noshear']['flags']!=0:
            print('non-zero flags objects: ' , k, ', flags=', resdict['noshear']['flags'])
        for k in mcal_keys:
            st = make_struct(res=resdict[k], obs=obsdict[k], shear_type=k)
            outlist.append(st)
            
    data = np.hstack(outlist)
    
    return data


def get_shear_info(data):
    
    shear_pair_data = {}
    for k in data.keys():
        w = ((data[k]['shear_type'] == 'noshear') & (data[k]['flags'] == 0))
        w_1p = ((data[k]['shear_type'] == '1p') & (data[k]['flags'] == 0))
        w_1m = ((data[k]['shear_type'] == '1m') & (data[k]['flags'] == 0))
        w_2p = ((data[k]['shear_type'] == '2p') & (data[k]['flags'] == 0))
        w_2m = ((data[k]['shear_type'] == '2m') & (data[k]['flags'] == 0))
        d = np.zeros(len(data[k]['g'][w]), dtype=[('g1_noshear', 'f8'), ('g1_1p', 'f8'), ('g1_1m', 'f8'), ('g1_2p', 'f8'), ('g1_2m', 'f8'), ('g2_noshear', 'f8'), ('g2_1p', 'f8'), ('g2_1m', 'f8'), ('g2_2p', 'f8'), ('g2_2m', 'f8')])
        
        d['g1_noshear'] = data[k]['g'][w, 0]
        d['g1_1p'] = data[k]['g'][w_1p, 0]
        d['g1_1m'] = data[k]['g'][w_1m, 0]
        d['g1_2p'] = data[k]['g'][w_2p, 0]
        d['g1_2m'] = data[k]['g'][w_2m, 0]
        
        d['g2_noshear'] = data[k]['g'][w, 1]
        d['g2_1p'] = data[k]['g'][w_1p, 1]
        d['g2_1m'] = data[k]['g'][w_1m, 1]
        d['g2_2p'] = data[k]['g'][w_2p, 1]
        d['g2_2m'] = data[k]['g'][w_2m, 1]
        
        shear_pair_data[k] = [d]
        
    return shear_pair_data


def main():

    args = get_args()
    rng = np.random.RandomState(args.seed)

    if args.mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        print(rank, size)
    else:
        print('not running MPI')

    outpath = args.outpath
    filtername = args.filter; filtername_ = filtername[0]
    test = args.test

    if not test:
        inpath = args.inpath + filtername
        rows_total = ['Row'+str(i).zfill(2) for i in range(36)]
        rows = []
        for row in rows_total:
            if int(row[-2:]) % size != rank:
                continue
            rows.append(row)
            
        ibxs = [str(i).zfill(2) for i in range(36)]; ibys = [str(i).zfill(2) for i in range(36)]

        # Z: g1=0.00, g2=0.00
        # A: g1=0.02, g2=0.00
        # B: g1=-0.02/2, g2=0.02*sqrt(3)/2
        # C: g1=-0.02/2, g2=-0.02*sqrt(3)/2

        # Run metacal for each row and save the measurement. 
        SKIP = []
        mcal_data = {i+'_'+j:[] for i in ibxs for j in ibys}
        for row in rows:
            iby = row[-2:]
            for ibx in ibxs:
                if ibx+'_'+iby in SKIP:
                    continue
                map_injstar, map_injext, map_injext_g0, map_injext_g60, map_injext_g120, WNmap = get_layers(row, ibx, inpath, filtername_)
                x,y,xi,yi,npix = get_truth_coord(row, ibx, inpath, filtername_)
                imcom_out  = {'Z':{'map':map_injext, 'shear_true':[0.00, 0.00]}, 
                            'A':{'map':map_injext_g0, 'shear_true':[0.02, 0.00]}, 
                            'B':{'map':map_injext_g60, 'shear_true':[-0.02/2, 0.02*np.sqrt(3)/2]},
                            'C':{'map':map_injext_g120, 'shear_true':[-0.02/2, -0.02*np.sqrt(3)/2]}}
                res = {}
                for k in imcom_out.keys():
                    d = run_mcal_grid(npix, imcom_out[k]['map'], map_injstar, WNmap, x, xi, y, yi, rng, filtername)
                    res[k] = d
                
                shear_pair_data = get_shear_info(res)
                mcal_data[ibx+'_'+iby] = shear_pair_data
                print('done with row %s, col %s' % (iby, ibx))
        print('done with metacal rank %s' % rank)

        # IF DATA CANNOT BE SENT TO RANK 0, WRITE OUTPUT HERE.
        with open(outpath + 'mcal_data_%s.pkl' % rank, 'wb') as handle:
            pickle.dump(mcal_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('wrote output rank %s' % rank)
    else:
        print('Running test mode...')
        # let's first test objects drawn on a single stamp.
        ngal = args.ngal
        noise = 1.0e-6
        input_shear = {'Z':{'shear_true':[0.00, 0.00]}, 
                       'A':{'shear_true':[0.02, 0.00]}, 
                       'B':{'shear_true':[-0.02/2, 0.02*np.sqrt(3)/2]},
                       'C':{'shear_true':[-0.02/2, -0.02*np.sqrt(3)/2]}}
        res = {}; mcal_data = {}
        for key in input_shear.keys():
            d = run_mcal(ngal, rng, noise, input_shear[key]['shear_true'])
            res[key] = d 

        shear_pair_data = get_shear_info(res)
        mcal_data['00_00'] = shear_pair_data

        with open(outpath + 'mcal_data_0.pkl', 'wb') as handle:
            pickle.dump(mcal_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # we can test objects drawn on SCA on a hexagonal grid.


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=314,
                        help='seed for rng')
    parser.add_argument('--ngal', type=int, default=1000,
                        help='number of trials')
    parser.add_argument('--filter', type=str, default='H158',
                        help='filter name')
    parser.add_argument('--inpath', type=str, default='/cwork/mat90/RomanDESC_sims_2024/RomanWAS/images/coadds/',
                        help='input path')
    parser.add_argument('--outpath', type=str, default='/hpc/group/cosmology/masaya/imcom_phase2/mcal/',
                        help='output path')
    parser.add_argument('--test', default=False, action='store_const', const=True, help='whether or not to run test')
    parser.add_argument('--mpi', default=False, action='store_const', const=True, help='whether or not to run using mpi')
    
    return parser.parse_args()

if __name__ == "__main__":
    main()
