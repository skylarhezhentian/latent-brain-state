"""Continuous IBL behavior loader used identically by Stages 2 and 4."""
from __future__ import annotations
import numpy as np, pandas as pd, scipy.ndimage
from brainbox.io.one import SessionLoader

def frame(x):
    if isinstance(x,pd.DataFrame):return x
    if isinstance(x,dict):
        try:return pd.DataFrame(x)
        except:return None
    return None

def first(df,names):return next((c for c in names if c in df),None)

def load_behavior(one,eid):
    sl=SessionLoader(one=one,eid=eid); sl.load_session_data(trials=False,wheel=True,pose=False,motion_energy=True,pupil=True,licks=True,pawstates=False); out={}
    wheel=frame(getattr(sl,'wheel',None))
    if wheel is not None and len(wheel):
        t=first(wheel,('times','timestamps')); v=first(wheel,('velocity','speed'))
        if t and v:out['wheel_speed']=(wheel[t].to_numpy(float),np.abs(wheel[v].to_numpy(float)))
    motion=getattr(sl,'motion_energy',None)
    if isinstance(motion,dict):
        for cam,cols,name in [('leftCamera',('whiskerMotionEnergy','motionEnergy'),'whisker_left'),('rightCamera',('whiskerMotionEnergy','motionEnergy'),'whisker_right'),('bodyCamera',('bodyMotionEnergy','motionEnergy'),'body_motion')]:
            df=frame(motion.get(cam))
            if df is not None and len(df):
                t=first(df,('times','timestamps')); v=first(df,cols)
                if t and v:out[name]=(df[t].to_numpy(float),df[v].to_numpy(float))
    pobj=getattr(sl,'pupil',None); pdf=frame(pobj)
    if pdf is None and isinstance(pobj,dict):
        for value in pobj.values():
            pdf=frame(value)
            if pdf is not None and len(pdf):break
    if pdf is not None and len(pdf):
        t=first(pdf,('times','timestamps')); v=first(pdf,('pupilDiameter_smooth','pupilDiameter_raw','pupilDiameter','diameter'))
        if t and v:out['pupil_diameter']=(pdf[t].to_numpy(float),pdf[v].to_numpy(float))
    licks=frame(getattr(sl,'licks',None))
    if licks is not None and len(licks):
        t=first(licks,('times','timestamps'))
        if t:
            lt=licks[t].to_numpy(float); lt=lt[np.isfinite(lt)]
            if len(lt):
                grid=np.arange(lt.min(),lt.max()+.1,.1); counts,_=np.histogram(lt,bins=np.append(grid,grid[-1]+.1)); out['lick_rate']=(grid+.05,scipy.ndimage.gaussian_filter1d(counts.astype(float)/.1,2))
    return out
