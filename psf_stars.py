import numpy as np
import fitsio as fio
import galsim 
from galsim import CelestialCoord
from astropy.io import fits
from astropy import wcs
import sys
from tqdm import tqdm
import os

def get_coadd_psf_stamp(coadd_file,coadd_psf_file,x,y,stamp_size,oversample_factor=1):

    """
    Returns the sum of high resolution image for each contributing image for each pixel in the coadd.
    """

    final_nxy = 8825
    psf_cache = {}
    xy = galsim.PositionD(int(final_nxy/2),int(final_nxy/2))
    wcs = galsim.AstropyWCS(file_name=coadd_file,hdu=1).local(xy)
    psf_wcs = galsim.JacobianWCS(dudx=wcs.dudx/oversample_factor,
                                dudy=wcs.dudy/oversample_factor,
                                dvdx=wcs.dvdx/oversample_factor,
                                dvdy=wcs.dvdy/oversample_factor)

    hdr = fio.FITS(coadd_file)['CTX'].read_header()
    if hdr['NAXIS']==3:
        ctx = np.left_shift(fio.FITS(coadd_file)['CTX'][1,int(x),int(y)].astype('uint64'),32)+fio.FITS(coadd_file)['CTX'][0,int(x),int(y)].astype('uint32')
    elif hdr['NAXIS']==2:
        ctx = fio.FITS(coadd_file)['CTX'][int(x),int(y)].astype('uint32')
    else:
        print('Not designed to work with more than 64 images.')

    ctx = ctx.item()
    if ctx==0:
        return None
    if ctx not in psf_cache:
        psf_coadd = galsim.InterpolatedImage(coadd_psf_file,hdu=fio.FITS(coadd_psf_file)[str(ctx)].get_extnum(),x_interpolant='lanczos5')
        b_psf = galsim.BoundsI( xmin=1,
                                ymin=1,
                                xmax=stamp_size*oversample_factor,
                                ymax=stamp_size*oversample_factor)
        psf_stamp = galsim.Image(b_psf, wcs=psf_wcs)
        # psf_coadd.drawImage(image=psf_stamp,offset=xy-psf_stamp.true_center)
        psf_coadd.drawImage(image=psf_stamp)
        psf_cache[ctx] = psf_stamp

    return psf_cache[ctx]

def _run_2pcf(ra, dec, g1, g2, out_f, gamt=False):

    import treecorr

    bin_config = dict(
        sep_units = 'arcmin',
        bin_slop = 0.1,

        min_sep = 0.5,
        max_sep = 80,
        nbins = 15,

        var_method = 'jackknife'
    )

    if os.path.exists('/hpc/group/cosmology/masaya/imcom_phase1/patch_center.txt'):
        f_pc = '/hpc/group/cosmology/masaya/imcom_phase1/patch_center.txt'
    else:
        cat_patch = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', npatch=5)
        cat_patch.write_patch_centers('/hpc/group/cosmology/masaya/imcom_phase1/patch_center.txt')
        f_pc = '/hpc/group/cosmology/masaya/imcom_phase1/patch_center.txt'

    if gamt:
        ng = treecorr.NGCorrelation(bin_config)
        cat1 = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', patch_centers=f_pc)
        cat2 = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', g1=g1, g2=g2, patch_centers=f_pc)
        ng.process(cat1, cat2)
        ng.write(out_f)
    else:
        gg = treecorr.GGCorrelation(bin_config)
        cat = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', g1=g1, g2=g2, patch_centers=f_pc)
        gg.process(cat)
        gg.write(out_f)

def _find_block_from_star_coord(ra, dec, ra_ctr, dec_ctr):
    """
    Returns which imcom block the input star is in from ra, dec of the star. 
    e.g.,) test3F_26_30_map.fits —> block at x=26, y=30 (0-index; 0-47)

    Uses the project function in galsim (https://galsim-developers.github.io/GalSim/_build/html/wcs.html#celestial-coordinates)
    """

    ra = ra*galsim.degrees # np.radians(ra)
    dec = dec*galsim.degrees # np.radians(dec)
    ra_ctr = ra_ctr*galsim.degrees #np.radians(ra_ctr)
    dec_ctr = dec_ctr*galsim.degrees #np.radians(dec_ctr)

    """
    # The steps are, 
    # (Ra,dec) on sphere -> x,y,z in Cartesian coordinates. 
    x = np.cos(ra) * np.sin(dec)
    y = np.sin(ra) * np.cos(dec)
    z = np.sin(dec)

    # rotate to (x,y,z) in frame where z is out of the plane at the projection center, x is east, y is north. 
    zeta = np.arctan(np.cos(dec)*np.sin(ra-ra_ctr) / (-np.sin(dec)*np.cos(dec_ctr) + np.sin(dec_ctr)*np.cos(dec)*np.cos(ra-ra_ctr)))
    eta = np.arctan(np.cos(dec_ctr)*np.sin(ra_ctr-ra) / (-np.sin(dec_ctr)*np.cos(dec) + np.sin(dec)*np.cos(dec_ctr)*np.cos(ra-ra_ctr)))

    psi = eta - zeta
    psi_r = psi - np.pi*np.round(psi/np.pi)
    """

    # do stereographic projection onto (x,y) plane
    center = CelestialCoord(ra=ra_ctr, dec=dec_ctr)
    x = []
    y = []
    for coord1,coord2 in zip(ra, dec):
        coords = CelestialCoord(ra=coord1, dec=coord2)
        u, v = center.project(coords, projection='stereographic')
        x.append(u.deg) # galsim convention: +u is west
        y.append(v.deg) # galsim convention: +v is north

    # And then express coordinates in units of arc minutes, add 24 to x and y to put (24,24) at center, and take the floor of the coordinates to get the integer block numbers. 
    x = np.array(x)*60 + 24; y = np.array(y)*60 + 24
    xblock = np.floor(x)
    yblock = np.floor(y)

    msk = ((xblock >= 0) & (xblock <= 47) & (yblock >= 0) & (yblock <= 47))

    return msk, xblock[msk], yblock[msk]

def main(argv):
    stars = fio.read('/hpc/group/cosmology/masaya/imcom_phase1/star_truth_summary.fits.gz')
    ra_ctr = 53.000000; dec_ctr = -40.000000

    msk_mag = ((stars['stamp']>0) & (stars['mag_H158']>0) & (stars['mag_Y106']>0) & (stars['mag_J129']>0) & (stars['mag_F184']>0))
    stars = stars[msk_mag]
    msk_block, xblock, yblock = _find_block_from_star_coord(stars['ra'], stars['dec'], ra_ctr, dec_ctr)
    stars = stars[msk_block]
    print('number of stars considered, ', len(stars))
    assert len(stars) == len(xblock)

    star_index = stars['ind']
    star_dict = {}
    drizzle_tilename = np.load('/hpc/group/cosmology/masaya/imcom_phase1/star_cat_coadd_tilename.npy')
    print('setting up drizzle coadd...', len(drizzle_tilename))
    for tname in tqdm(drizzle_tilename):
        fname = f"""/hpc/group/cosmology/phy-lsst/public/dc2_sim_output/truth/coadd/dc2_index_{tname}.fits.gz"""
        d = fio.read(fname)
        d = d[((d['x'] > 500) & (d['x'] < 8325) & (d['y'] > 500) & (d['y'] < 8325) & (d['gal_star'] == 1) & (d['stamp'] > 0) & (d['mag_H158']>0) & (d['mag_Y106']>0) & (d['mag_J129']>0) & (d['mag_F184']>0))] # only unique stars
        d_ = d[np.in1d(d['ind'], star_index)]
        for i,st in enumerate(d_['ind']):
            if st not in star_dict.keys():
                star_dict[st] = tname
            else:
                print('non-unique stars are in the truth file. ')
    assert len(star_index) == len(star_dict.keys())


    # starcube process
    # get x,y coordinates of stars in a block.
    # cutout stamps from x, y, stamp size
    # measure with adaptive moments
    filtername = sys.argv[1]
    filter = sys.argv[2]
    amp_scaling = (0.025/0.11)**2
    amp_scaling_drizzle = (0.0575/0.11)**2
    n = 2600
    bd = 50
    bd_drizzle = 22
    bd2 = 10
    rs = 1./np.sqrt(2.)/60.*np.pi/180*1.08333
    sigma = 20.
    # WNuse_slice = 4
    # PNuse_slice = 5
    use_slice = 0
    use_sliceB = 1

    # ncol = 30
    # pos = np.zeros((1,ncol)) # ra, dec, ibx, iby, x, y, xi, yi, dx, dy, [Axysg1g2]G, [Axysg1g2]C, fid, ct
    # image = np.zeros((1,bd*2-1,bd*2-1))
    measure_drizzle = False
    save_res = False
    run_2pcf = False
    outfile_g = '/hpc/group/cosmology/masaya/imcom_phase1/PSFStarCat_{:s}.fits'.format(filter)
    fhist = np.zeros((61,),dtype=np.uint32)
    print('measuring...')
    count = 0
    radec = np.zeros(10, dtype=[('point', object), ('ra', 'f8'), ('dec', 'f8'), ('custom', object)])
    count_stars = {}
    for j,(ibx,iby) in tqdm(enumerate(zip(xblock, yblock))):
        # if ((int(ibx) == 23) & (int(iby) == 22)):
        #     radec['point'][count] = 'point'
        #     radec['ra'][count] = stars['ra'][j]
        #     radec['dec'][count] = stars['dec'][j]
        #     radec['custom'][count] = '# point=x'
        #     count += 1
        #     continue
        # else:
        #     continue

        if not str(int(ibx))+'_'+str(int(iby)) in list(count_stars.keys()):
            count_stars[str(int(ibx))+'_'+str(int(iby))] = 1
        else:
            count_stars[str(int(ibx))+'_'+str(int(iby))] += 1
        
        if filtername == 'wide_band':
            infile = r'/hpc/group/cosmology/phy-lsst/public/imcom_output/YJHF/imageW{:02d}_{:02d}.fits'.format(int(ibx),int(iby))
        else:
            infile = r'/hpc/group/cosmology/phy-lsst/public/imcom_output/{:s}/test3{:s}_{:02d}_{:02d}_map.fits'.format(filtername,filter,int(ibx),int(iby))
            
        with fits.open(infile) as f:
            mywcs = wcs.WCS(f[0].header)
            # wcs = galsim.FitsWCS(file_name=infile)
            # x,y = wcs.toImage(ra[j], dec[j], units='degrees')

            x,y,z1,z2 = mywcs.all_world2pix(stars['ra'][j], stars['dec'][j], 0, 0, 0)
            xi = np.rint(x).astype(np.int64); yi = np.rint(y).astype(np.int64)
            # Only use stars that are in the inner region defined by 50 pix boundary around the image. 
            if not ((x >= 50) & (x <= 2550) & (y >= 50) & (y <= 2550)):
                continue

            if filtername != 'wide_band':
                wt = np.rint(1./np.amax(f['INWEIGHT'].data[0,:,:,:]+1e-6, axis=0))
                fmap = f['FIDELITY'].data[0,:,:].astype(np.float32)
                for fy in range(61): fhist[fy] += np.count_nonzero(f['FIDELITY'].data[0,100:-100,100:-100]==fy)
                map = f[0].data[0,use_slice,:,:]
                mapB = f[0].data[0,use_sliceB,:,:]
            else:
                map = f[0].data[use_slice,:,:]
                mapB = f[0].data[use_sliceB,:,:]
            

        newpos = np.zeros(1, dtype=[('ra', 'f8'), ('dec', 'f8'), ('x', 'f8'), ('y', 'f8'), ('xi', 'i8'), ('yi', 'i8'), ('dx', 'f8'), ('dy', 'f8'), ('amp_simple_imcom', 'f8'), ('dx_simple_imcom', 'f8'), ('dy_simple_imcom', 'f8'), ('sigma_simple_imcom', 'f8'), ('g1_simple_imcom', 'f8'), ('g2_simple_imcom', 'f8'), ('amp_truth_imcom', 'f8'), ('dx_truth_imcom', 'f8'), ('dy_truth_imcom', 'f8'), ('sigma_truth_imcom', 'f8'), ('g1_truth_imcom', 'f8'), ('g2_truth_imcom', 'f8'), ('amp_simple_drizzle', 'f8'), ('dx_simple_drizzle', 'f8'), ('dy_simple_drizzle', 'f8'), ('sigma_simple_drizzle', 'f8'), ('g1_simple_drizzle', 'f8'), ('g2_simple_drizzle', 'f8'), ('amp_truth_drizzle', 'f8'), ('dx_truth_drizzle', 'f8'), ('dy_truth_drizzle', 'f8'), ('sigma_truth_drizzle', 'f8'), ('g1_truth_drizzle', 'f8'), ('g2_truth_drizzle', 'f8'), ('amp_highres_drizzle', 'f8'), ('dx_highres_drizzle', 'f8'), ('dy_highres_drizzle', 'f8'), ('sigma_highres_drizzle', 'f8'), ('g1_highres_drizzle', 'f8'), ('g2_highres_drizzle', 'f8'), ('amp_highres_imcom', 'f8'), ('dx_highres_imcom', 'f8'), ('dy_highres_imcom', 'f8'), ('sigma_highres_imcom', 'f8'), ('g1_highres_imcom', 'f8'), ('g2_highres_imcom', 'f8'), ('fidelity', 'f8'), ('coverage', 'f8'), ('flags', int), ('mag_Y106', 'f8'), ('mag_J129', 'f8'), ('mag_H158', 'f8'), ('mag_F184', 'f8')])
        newpos['ra'] = stars['ra'][j]
        newpos['dec'] = stars['dec'][j]
        newpos['x'] = x
        newpos['y'] = y
        newpos['xi'] = xi
        newpos['yi'] = yi
        newpos['dx'] = x-xi
        newpos['dy'] = y-yi
        newpos['mag_Y106'] = stars['mag_Y106'][j]
        newpos['mag_J129'] = stars['mag_J129'][j]
        newpos['mag_H158'] = stars['mag_H158'][j]
        newpos['mag_F184'] = stars['mag_F184'][j]

        # stamp_size = stars['stamp'][j]
        # print(stamp_size)
        # newimage = map[yi-stamp_size//2:yi+stamp_size//2,xi-stamp_size//2:xi+stamp_size//2]
        newimage = map[yi+1-bd:yi+bd,xi+1-bd:xi+bd]
        # PSF shape (simple model)
        try:
            moms = galsim.Image(newimage).FindAdaptiveMom()
            newpos['amp_simple_imcom'] = moms.moments_amp 
            newpos['dx_simple_imcom'] = moms.moments_centroid.x + (xi-bd) - x # centroid offset
            newpos['dy_simple_imcom'] = moms.moments_centroid.y + (yi-bd) - y
            newpos['sigma_simple_imcom'] = moms.moments_sigma
            newpos['g1_simple_imcom'] = moms.observed_shape.g1
            newpos['g2_simple_imcom'] = moms.observed_shape.g2
            if stars['mag_Y106'][j]>18 and 2*((0.025*moms.moments_sigma)**2) > 0.032:
                fio.write('./outliers/imcom_sci_'+str(j)+'_Y.fits', map[yi+1-70:yi+70,xi+1-70:xi+70])
        except galsim.errors.GalSimHSMError:
            newpos['flags'] = 1
            print('HSM failed on simple model image')
        # if stars['mag_Y106'][j] < 17:
        #     # fio.write('./stars/bright_simple_'+str(j)+'.fits', newimage)
        #     print(j, stars['ra'][j], stars['dec'][j], np.mean(fmap[yi+1-bd2:yi+bd2,xi+1-bd2:xi+bd2]), moms.observed_shape.g1, moms.observed_shape.g2)
        #     sys.exit()

        # PSF shape (truth)
        newimageB = mapB[yi+1-bd:yi+bd,xi+1-bd:xi+bd]
        try:
            moms = galsim.Image(newimageB).FindAdaptiveMom()
            newpos['amp_truth_imcom'] = moms.moments_amp
            newpos['dx_truth_imcom'] = moms.moments_centroid.x + (xi-bd) - x # centroid offset
            newpos['dy_truth_imcom'] = moms.moments_centroid.y + (yi-bd) - y
            newpos['sigma_truth_imcom'] = moms.moments_sigma
            newpos['g1_truth_imcom'] = moms.observed_shape.g1
            newpos['g2_truth_imcom'] = moms.observed_shape.g2
            # print('amp, imcom, truth, ', moms.moments_amp*amp_scaling)
            # fio.write('./imcom_truth_1.fits', newimageB)
            # if stars['mag_F184'][j] > 22 and np.sqrt(moms.observed_shape.g1**2 + moms.observed_shape.g2**2) > 0.008:
            #     fio.write('./stars/largeshear_truth_'+str(j)+'.fits', newimageB)
            #     print(moms.observed_shape.g1, moms.observed_shape.g2, np.mean(fmap[yi+1-bd2:yi+bd2,xi+1-bd2:xi+bd2]))
        except galsim.errors.GalSimHSMError:
            newpos['flags'] = 2
            print('HSM failed on truth image')

        if filtername != 'wide_band':
            # fidelity
            newpos['fidelity'] = np.mean(fmap[yi+1-bd2:yi+bd2,xi+1-bd2:xi+bd2])
            # coverage
            newpos['coverage'] = wt[yi//bd,xi//bd]

        ## Measure on truth stamp cutouts
        """
        dither = int(stars['dither'][j])
        sca = int(stars['sca'][j])
        start_row = int(stars['start_row'][j])
        stamp = 256 # int(stars['stamp'][j])
        f_stamp = /hpc/group/cosmology/phy-lsst/public/dc2_sim_output/stamps/dc2_F184_{dither}_{sca}_0_star.fits.gz
        truth_stamp = fio.FITS(f_stamp)['image_cutouts'].read()
        im = truth_stamp[start_row:start_row+stamp**2].reshape(stamp, stamp)
        print(start_row, im)
        try:
            moms = galsim.Image(im).FindAdaptiveMom()
            newpos['amp_stamp'] = moms.moments_amp
            # newpos['dx_stamp'] = moms.moments_centroid.x + (xi-bd) - x # centroid offset
            # newpos['dy_stamp'] = moms.moments_centroid.y + (yi-bd) - y
            newpos['sigma_stamp'] = moms.moments_sigma
            newpos['g1_stamp'] = moms.observed_shape.g1
            newpos['g2_stamp'] = moms.observed_shape.g2
        except galsim.errors.GalSimHSMError:
            newpos['flags'] = 4
            print('HSM failed on truth stamp')
        """

        if measure_drizzle:
            # Measure on drizzle images (simple and coadd psf)
            tilename = star_dict[stars['ind'][j]]
            f_coadd = f"""/hpc/group/cosmology/phy-lsst/public/dc2_sim_output/images/simple_model/coadd/dc2_{filtername}_{tilename}.fits.gz"""
            coadd = fio.read(f_coadd)
            xi_coadd = np.rint(stars['x'][j]).astype(np.int64); yi_coadd = np.rint(stars['y'][j]).astype(np.int64)
            newimage_drizzle = coadd[yi_coadd+1-bd_drizzle:yi_coadd+bd_drizzle,xi_coadd+1-bd_drizzle:xi_coadd+bd_drizzle]
            try:
                moms = galsim.Image(newimage_drizzle).FindAdaptiveMom()
                newpos['amp_simple_drizzle'] = moms.moments_amp
                newpos['dx_simple_drizzle'] = moms.moments_centroid.x + (xi_coadd-bd_drizzle) - stars['x'][j] # centroid offset
                newpos['dy_simple_drizzle'] = moms.moments_centroid.y + (yi_coadd-bd_drizzle) - stars['y'][j]
                newpos['sigma_simple_drizzle'] = moms.moments_sigma
                newpos['g1_simple_drizzle'] = moms.observed_shape.g1
                newpos['g2_simple_drizzle'] = moms.observed_shape.g2
                # print('amp, drizzle, sci, ', moms.moments_amp*amp_scaling_drizzle)
                # fio.write('./drizzle_sci_1.fits', newimage_drizzle)
            except galsim.errors.GalSimHSMError:
                newpos['flags'] = 4
                print('HSM failed on simple model drizzle image')

            # print(j, stars[j]['mag_F184'])
            # fio.write('./stars/paper/mag22_simple_'+str(j)+'_F.fits', newimage)
            # fio.write('./stars/paper/mag22_truth_'+str(j)+'_F.fits', newimageB)
            # fio.write('./stars/paper/mag22_simple_drizzle_'+str(j)+'_F.fits', newimage_drizzle)
            # sys.exit()

            f_psf = f"""/hpc/group/cosmology/phy-lsst/public/dc2_sim_output/psf/coadd/dc2_{filtername}_{tilename}_psf.fits"""
            try:
                coadd_psf = get_coadd_psf_stamp(f_coadd, f_psf, stars['x'][j], stars['y'][j], bd_drizzle*2, oversample_factor=1)
            except:
                print(j, f_coadd, f_psf)
                continue
            psf_image = coadd_psf.array
            try:
                moms = galsim.Image(psf_image).FindAdaptiveMom()
                newpos['amp_truth_drizzle'] = moms.moments_amp
                newpos['dx_truth_drizzle'] = moms.moments_centroid.x + (xi_coadd-bd_drizzle) - stars['x'][j] # centroid offset
                newpos['dy_truth_drizzle'] = moms.moments_centroid.y + (yi_coadd-bd_drizzle) - stars['y'][j]
                newpos['sigma_truth_drizzle'] = moms.moments_sigma
                newpos['g1_truth_drizzle'] = moms.observed_shape.g1
                newpos['g2_truth_drizzle'] = moms.observed_shape.g2
            except galsim.errors.GalSimHSMError:
                newpos['flags'] = 8
                print('HSM failed on truth drizzle psf')

            # Measure measurement of high res PSF
            # high res PSF (imcom)
            target_psf = fio.read(f"""/hpc/group/cosmology/masaya/imcom_phase1/test3{filter}_00_00_outputpsfs.fits""")
            try:
                moms = galsim.Image(target_psf).FindAdaptiveMom()
                newpos['amp_highres_imcom'] = moms.moments_amp
                newpos['dx_highres_imcom'] = moms.moments_centroid.x + (xi-bd) - stars['x'][j] # centroid offset
                newpos['dy_highres_imcom'] = moms.moments_centroid.y + (yi-bd) - stars['y'][j]
                newpos['sigma_highres_imcom'] = moms.moments_sigma
                newpos['g1_highres_imcom'] = moms.observed_shape.g1
                newpos['g2_highres_imcom'] = moms.observed_shape.g2
            except galsim.errors.GalSimHSMError:
                newpos['flags'] = 16
                print('HSM failed on high-res drizzle psf')
            
            # high res PSF (drizzle)
            highres_coadd_psf = get_coadd_psf_stamp(f_coadd, f_psf, stars['x'][j], stars['y'][j], bd_drizzle*2, oversample_factor=16)
            highres_psf_image = highres_coadd_psf.array
            try:
                moms = galsim.Image(highres_psf_image).FindAdaptiveMom()
                newpos['amp_highres_drizzle'] = moms.moments_amp
                newpos['dx_highres_drizzle'] = moms.moments_centroid.x + (xi_coadd-bd_drizzle) - stars['x'][j] # centroid offset
                newpos['dy_highres_drizzle'] = moms.moments_centroid.y + (yi_coadd-bd_drizzle) - stars['y'][j]
                newpos['sigma_highres_drizzle'] = moms.moments_sigma
                newpos['g1_highres_drizzle'] = moms.observed_shape.g1
                newpos['g2_highres_drizzle'] = moms.observed_shape.g2
            except galsim.errors.GalSimHSMError:
                newpos['flags'] = 32
                print('HSM failed on high-res drizzle psf')

        if j == 0:
            pos = newpos
        else:
            pos = np.concatenate((pos, newpos), axis=0)
        # image = np.concatenate((image, newimage), axis=0)


    # np.savetxt('./radec_23_22.reg', radec, fmt='%s %.18e %.18e %s')
    # print(count_stars)
    # sys.exit()

    if save_res:
        fio.write(outfile_g, pos)

    if run_2pcf:
        print('running 2pcf...')
        pos = fio.read(outfile_g)
        if filtername == 'wide_band':
            pos = pos[((pos['flags'] == 0) & (pos['mag_H158'] > 18))]
        else:
            pos = pos[((pos['flags'] == 0) & (pos['mag_'+filtername] > 18))]
        _run_2pcf(pos['ra'], pos['dec'], pos['g1_simple_imcom'], pos['g2_simple_imcom'], '/hpc/group/cosmology/masaya/imcom_phase1/psf_'+filtername+'_shape_simple_imcom_2pcf.fits')
        _run_2pcf(pos['ra'], pos['dec'], pos['g1_simple_drizzle'], pos['g1_simple_drizzle'], '/hpc/group/cosmology/masaya/imcom_phase1/psf_'+filtername+'_shape_simple_drizzle_2pcf.fits')
        _run_2pcf(pos['ra'], pos['dec'], pos['g1_simple_imcom'], pos['g2_simple_imcom'], '/hpc/group/cosmology/masaya/imcom_phase1/psf_'+filtername+'_gamt_simple_imcom_2pcf.fits', gamt=True)
        _run_2pcf(pos['ra'], pos['dec'], pos['g1_simple_drizzle'], pos['g1_simple_drizzle'], '/hpc/group/cosmology/masaya/imcom_phase1/psf_'+filtername+'_gamt_simple_drizzle_2pcf.fits', gamt=True)

if __name__ == "__main__":
    main(sys.argv)