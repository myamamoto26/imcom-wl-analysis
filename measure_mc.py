import numpy as np 
import fitsio as fio
import pickle
import os, sys
import glob
from tqdm import tqdm

NBLOCKS = 36

def measure_m_c_original(data):
    
    # TO-DO: rewrite this function. 
    
    w = ((data['shear_type'] == 'noshear') & (data['flags'] == 0))
    w_1p = ((data['shear_type'] == '1p') & (data['flags'] == 0))
    w_1m = ((data['shear_type'] == '1m') & (data['flags'] == 0))

    g = data['g'][w].mean(axis=0)
    gerr = data['g'][w].std(axis=0) / np.sqrt(w.size)
    g1_1p = data['g'][w_1p, 0].mean()
    g1_1m = data['g'][w_1m, 0].mean()
    R11 = (g1_1p - g1_1m)/0.02

    shear = g / R11
    shear_err = gerr / R11

    m = shear[0]/shear_true[0]-1
    merr = shear_err[0]/shear_true[0]

    s2n = data['s2n'][w].mean()

    print('S/N: %g' % s2n)
    print('R11: %g' % R11)
    print('m: %g +/- %g (99.7%% conf)' % (m, merr*3))
    print('c: %g +/- %g (99.7%% conf)' % (shear[1], shear_err[1]*3))

def measure_m_c(cat):
    
    """
    Z: g1=0.00, g2=0.00
    A: g1=0.02, g2=0.00
    B: g1=-0.02/2, g2=0.02*sqrt(3)/2
    C: g1=-0.02/2, g2=-0.02*sqrt(3)/2
    """
    
    # Shear response for g1=0.00, g2=0.00 run.
    R11_Z = (np.mean(cat['Z']['g1_1p']) - np.mean(cat['Z']['g1_1m']))/(2*0.01)
    R22_Z = (np.mean(cat['Z']['g2_2p']) - np.mean(cat['Z']['g2_2m']))/(2*0.01)
    
    # Shear response for g1=0.02, g2=0.00 run.
    R11_A = (np.mean(cat['A']['g1_1p']) - np.mean(cat['A']['g1_1m']))/(2*0.01)
    R22_A = (np.mean(cat['A']['g2_2p']) - np.mean(cat['A']['g2_2m']))/(2*0.01)
    
    # Shear response for g1=-0.02/2, g2=0.02*sqrt(3)/2 run.
    R11_B = (np.mean(cat['B']['g1_1p']) - np.mean(cat['B']['g1_1m']))/(2*0.01)
    R22_B = (np.mean(cat['B']['g2_2p']) - np.mean(cat['B']['g2_2m']))/(2*0.01)
    
    # Shear response for g1=-0.02/2, g2=-0.02*sqrt(3)/2 run.
    R11_C = (np.mean(cat['C']['g1_1p']) - np.mean(cat['C']['g1_1m']))/(2*0.01)
    R22_C = (np.mean(cat['C']['g2_2p']) - np.mean(cat['C']['g2_2m']))/(2*0.01)
    
    R11_BC = (R11_B + R11_C)/2.; R22_BC = (R22_B + R22_C)/2.
    
    # Now compute m1 and c2.
    dg = 0.02
    m1 = (4*np.mean(cat['A']['g1_noshear']) - 2*np.mean(cat['B']['g1_noshear']) - 2*np.mean(cat['C']['g1_noshear']))/(3 * dg * (R11_A + R11_BC)) - 1
    c2 = (np.mean(cat['Z']['g2_noshear']) + np.mean(cat['A']['g2_noshear']))/(R22_Z + R22_A)
    
    # Now compute m2 and c1. 
    m2 = (2*np.mean(cat['B']['g2_noshear']) - 2*np.mean(cat['C']['g2_noshear']))/(np.sqrt(3) * dg * (R22_B + R22_C)) - 1
    c1 = (np.mean(cat['B']['g1_noshear']) + np.mean(cat['C']['g1_noshear']))/(R11_B + R11_C)
    
    return m1,m2,c1,c2


def compute_jk_errors(res_mc, N):
    
    jk_cov = np.zeros(1, dtype=[('m1','f8'), ('m2','f8'), ('c1','f8'), ('c2','f8')])

    for k in res_mc.keys():
        # compute jackknife average. 
        jk_all_ave = np.mean(res_mc[k])
        jk_ave = np.array(res_mc[k])

        cov = np.sqrt((N-1)/N)*np.sqrt(np.sum((jk_ave - jk_all_ave)**2))

        jk_cov[k] = cov

    return jk_cov


def main(argv):

    inpath = sys.argv[1]
    rows = ['Row'+str(i).zfill(2) for i in range(36)]
    ibxs = [str(i).zfill(2) for i in range(36)]; ibys = [str(i).zfill(2) for i in range(36)]

    # accumulate individual outputs. 
    print('accumulating outputs')
    fs_mcal = sorted(glob.glob(os.path.join(inpath, 'mcal_data_*.pkl')))
    with open(fs_mcal[0], 'rb') as handle:
        mcal_data = pickle.load(handle)
    for f in fs_mcal[1:]:
        with open(f, 'rb') as handle:
            tmp_mcal_data = pickle.load(handle)
        for row in rows:
            iby = row[-2:]
            for ibx in ibxs:
                if len(tmp_mcal_data[ibx+'_'+iby]) != 0:
                    mcal_data[ibx+'_'+iby] = tmp_mcal_data[ibx+'_'+iby]
    
    # concatenate results from all the blocks to compute mean m,c.
    print('concatenating')
    mcal_res = {'Z':[], 'A':[], 'B':[], 'C':[]}
    for row in rows:
        iby = row[-2:]
        for ibx in ibxs:
            for k in mcal_res.keys():
                mcal_res[k].append(mcal_data[ibx+'_'+iby][k][0])

    mcal_concat_res = {}
    for k in mcal_res.keys():
        mcal_concat_res[k] = np.concatenate(mcal_res[k])

    # Get mean m, c for all blocks.
    m1,m2,c1,c2 = measure_m_c(mcal_concat_res) # shear_pair_data

    # Let's get jackknife errors. 
    jk_mc = {'m1':[], 'm2':[], 'c1':[], 'c2':[]}
    for row in tqdm(rows):
        iby = int(row[-2:])
        start = iby * NBLOCKS
        end = (iby+1) * NBLOCKS
            
        Z = np.concatenate(mcal_res['Z'][start:end])
        A = np.concatenate(mcal_res['A'][start:end])
        B = np.concatenate(mcal_res['B'][start:end])
        C = np.concatenate(mcal_res['C'][start:end])
        
        jk_res = {'Z':Z, 'A':A, 'B':B, 'C':C}
        m1,m2,c1,c2 = measure_m_c(jk_res)
        jk_mc['m1'].append(m1)
        jk_mc['m2'].append(m2)
        jk_mc['c1'].append(c1)
        jk_mc['c2'].append(c2)

    jk_cov = compute_jk_errors(jk_mc, NBLOCKS)
    num_obj = len(mcal_concat_res['Z'])

    # print('S/N: %g' % s2n)
    # print('R11: %g' % R11)
    print('Total number of objects: %g' % num_obj)
    print('m1: %g +/- %g (99.7%% conf)' % (m1, jk_cov['m1']*3))
    print('c1: %g +/- %g (99.7%% conf)' % (c1, jk_cov['c1']*3))
    print('m2: %g +/- %g (99.7%% conf)' % (m2, jk_cov['m2']*3))
    print('c2: %g +/- %g (99.7%% conf)' % (c2, jk_cov['c2']*3))

if __name__ == "__main__":
    main(sys.argv)