"""Utilities for Stage 4 PCA component decomposition."""
from __future__ import annotations
import json
import numpy as np, pandas as pd, scipy.signal, scipy.stats
from iblatlas.regions import BrainRegions
BANDS={'1_4':(1,4),'4_8':(4,8),'8_13':(8,13),'13_30':(13,30),'30_100':(30,100),'100_200':(100,200)}

def pearson(x,y):
    g=np.isfinite(x)&np.isfinite(y)
    if g.sum()<20 or np.std(x[g])<1e-12 or np.std(y[g])<1e-12:return np.nan,np.nan,int(g.sum())
    r,p=scipy.stats.pearsonr(x[g],y[g]); return float(r),float(p),int(g.sum())

def bh(p):
    p=np.asarray(p,float); q=np.full_like(p,np.nan); g=np.isfinite(p)
    if not g.any():return q
    v=p[g]; o=np.argsort(v); z=v[o]*len(v)/np.arange(1,len(v)+1); z=np.minimum.accumulate(z[::-1])[::-1]; b=np.empty_like(z); b[o]=np.minimum(z,1); q[g]=b; return q

def regions(ch):
    try:return np.asarray(BrainRegions().id2acronym(np.asarray(ch.get('atlas_id',np.zeros(384)),int),mapping='Cosmos')).astype(str)
    except:return np.array(['unknown']*384)

def spectral(score,fs):
    f,p=scipy.signal.welch(score,fs=fs,nperseg=min(int(4*fs),len(score))); m=(f>=1)&(f<200); den=p[m].sum(); d={}
    for name,(lo,hi) in BANDS.items():d[f'power_fraction_{name}']=float(p[(f>=lo)&(f<hi)].sum()/den) if den else np.nan
    d['dominant_band']=max(BANDS,key=lambda z:d[f'power_fraction_{z}']); d['centroid_hz']=float((f[m]*p[m]).sum()/den) if den else np.nan; return d

def trial_col(df,names):
    c=next((x for x in names if x in df),None); return df[c].to_numpy(float) if c else None

def rmswin(s,fs,rel,a,b):
    i=max(0,int((rel+a)*fs)); j=min(len(s),int((rel+b)*fs)); return float(np.sqrt(np.mean(s[i:j]**2))) if j-i>=max(2,int(.05*fs)) else np.nan

def spatial_metrics(load,reg):
    w=load**2; valid=[z for z in np.unique(reg) if z not in ('void','root','unknown','nan','')]
    masses={z:float(w[reg==z].sum()/w.sum()) for z in valid}; dom=max(masses,key=masses.get) if masses else 'unknown'; pv=np.array(list(masses.values()))
    ent=float(-(pv[pv>0]*np.log(pv[pv>0])).sum()/np.log(max(2,len(pv)))) if len(pv) else np.nan
    return {'effective_channels':float(1/(load**4).sum()),'loading_roughness':float(np.mean(np.diff(load)**2)),'dominant_region':dom,'region_entropy':ent,'region_masses_json':json.dumps(masses)}
