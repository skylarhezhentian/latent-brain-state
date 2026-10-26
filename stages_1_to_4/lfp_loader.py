"""Tolerant public IBL LFP loader for small probe clock-rate deviations."""
from __future__ import annotations
from pathlib import Path
import numpy as np, scipy.signal
from brainbox.io.one import SpikeSortingLoader
from ibldsp.voltage import destripe_lfp
CACHE=Path('stage24_outputs/cache')

def read_stream(sr,first,last,piece_s=5.):
    step=max(1,int(round(piece_s*float(sr.fs)))); parts=[]
    for a in range(first,last,step):parts.append(np.asarray(sr[a:min(a+step,last),:-sr.nsync],dtype='float32'))
    return np.concatenate(parts,axis=0).T

def load_lfp(one,pid,t0_main,duration=100.,fs_out=500.):
    CACHE.mkdir(parents=True,exist_ok=True); f=CACHE/f'{pid}_{int(duration)}s_{int(fs_out)}hz.npz'
    if f.exists():
        with np.load(f) as z:return z['x'].astype('float32'),float(z['fs'])
    ssl=SpikeSortingLoader(pid=pid,one=one); sr=ssl.raw_electrophysiology(band='lf',stream=True); fs=float(sr.fs); down=max(1,int(round(fs/fs_out))); actual=fs/down
    try:t0=float(ssl.timesprobe2times(np.array([t0_main]),direction='reverse')[0])
    except:t0=float(t0_main)
    s0=int(round(t0*fs)); total=int(round(duration*fs)); chunk=int(round(20*fs)); pad=int(round(1.5*fs)); out=[]
    try:
        for r0 in range(0,total,chunk):
            r1=min(r0+chunk,total); d0=s0+r0; d1=s0+r1; a=max(0,d0-pad); b=d1+pad
            raw=read_stream(sr,a,b); x=destripe_lfp(raw,fs=fs,channel_labels=np.zeros(raw.shape[0])).astype('float32'); y=scipy.signal.resample_poly(x,1,down,axis=1).astype('float32')
            k0=int(round((d0-a)/down)); kn=int(round((d1-d0)/down)); out.append(y[:,k0:k0+kn])
    finally:
        try:sr.close()
        except:pass
    x=np.concatenate(out,axis=1)*1e6; n=int(round(duration*actual)); x=x[:,:n].astype('float32')
    if x.shape[0]!=384 or x.shape[1]<n-3:raise RuntimeError(f'unexpected LFP shape {x.shape}')
    np.savez_compressed(f,x=x,fs=actual); return x,actual
