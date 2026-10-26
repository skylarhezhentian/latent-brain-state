"""Fair nonlinear Track-A compression: LFP-only CNN autoencoder versus PCA.

The autoencoder sees only destriped LFP. Behavior is used after training, with the
same fixed spectral-feature + ridge evaluation used for the PCA reconstruction.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, time, traceback
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from track_a_rate_distortion import run_track_a as base
from stages_1_to_4.behavior_loader import load_behavior

DURATION=100.0; FS=500.0; KS=(8,16,32)

class SpatialCNN(nn.Module):
    def __init__(self,k):
        super().__init__(); self.k=k
        self.enc=nn.Sequential(nn.Conv1d(1,8,5,2,2),nn.GELU(),nn.Conv1d(8,16,5,2,2),nn.GELU(),nn.Conv1d(16,16,5,2,2),nn.GELU())
        self.e=nn.Linear(16*48,k); self.d=nn.Linear(k,16*48)
        self.dec=nn.Sequential(nn.ConvTranspose1d(16,16,4,2,1),nn.GELU(),nn.ConvTranspose1d(16,8,4,2,1),nn.GELU(),nn.ConvTranspose1d(8,1,4,2,1))
    def forward(self,x):
        z=self.e(self.enc(x[:,None]).flatten(1)); return self.dec(self.d(z).view(-1,16,48))[:,0]

def loader(x,idx,batch,shuffle,seed):
    t=torch.from_numpy(np.ascontiguousarray(x[:,idx].T.astype('float32')))
    return DataLoader(TensorDataset(t),batch_size=batch,shuffle=shuffle,generator=torch.Generator().manual_seed(seed),num_workers=0)

def train_codec(x,k,seed):
    torch.manual_seed(seed); torch.set_num_threads(1); n=x.shape[1]; ntr=int(.64*n); ndev=int(.8*n)
    mu=x[:,:ntr].mean(1,keepdims=True).astype('float32'); scale=max(float(np.sqrt(np.mean((x[:,:ntr]-mu)**2))),1e-6); z=((x-mu)/scale).astype('float32')
    rng=np.random.default_rng(seed); tri=np.arange(ntr); vai=np.arange(ntr,ndev)
    if len(tri)>16000: tri=np.sort(rng.choice(tri,16000,False))
    if len(vai)>4000: vai=np.sort(rng.choice(vai,4000,False))
    model=SpatialCNN(k); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5); best=1e9; best_ep=0; wait=0; hist=[]
    for ep in range(15):
        model.train(); s=c=0
        for (b,) in loader(z,tri,512,True,seed+ep):
            loss=((model(b)-b)**2).mean(); opt.zero_grad(); loss.backward(); opt.step(); s+=float(loss)*len(b); c+=len(b)
        model.eval(); vs=vc=0
        with torch.no_grad():
            for (b,) in loader(z,vai,1024,False,seed):
                loss=((model(b)-b)**2).mean(); vs+=float(loss)*len(b); vc+=len(b)
        vl=vs/vc; hist.append([ep,s/c,vl])
        if vl<best-1e-5: best=vl; best_ep=ep; wait=0
        else:
            wait+=1
            if wait>=3: break
    mu=x[:,:ndev].mean(1,keepdims=True).astype('float32'); scale=max(float(np.sqrt(np.mean((x[:,:ndev]-mu)**2))),1e-6); z=((x-mu)/scale).astype('float32')
    di=np.arange(ndev)
    if len(di)>20000: di=np.sort(rng.choice(di,20000,False))
    torch.manual_seed(seed+999); model=SpatialCNN(k); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5)
    for ep in range(best_ep+1):
        model.train()
        for (b,) in loader(z,di,512,True,seed+100+ep):
            loss=((model(b)-b)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); rec=np.empty_like(z); cur=0
    with torch.no_grad():
        for (b,) in loader(z,np.arange(n),2048,False,seed):
            y=model(b).numpy(); rec[:,cur:cur+len(y)]=y.T; cur+=len(y)
    return rec*scale+mu, {"best_epoch":best_ep,"validation_loss":best,"parameters":sum(p.numel() for p in model.parameters()),"history":hist}

def metrics(x,rec):
    te=slice(int(.8*x.shape[1]),None); a=x[:,te]; b=rec[:,te]
    return base.pooled_reconstruction_metrics(a,b)

def decode(rec,fs,beh,t0):
    X,t,_=base.bandpower_features(rec,fs); Y,names=base.behavior_at_times(beh,t0+t); return base.decode_behaviors(X,t,Y,names,DURATION)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--shard',type=int,required=True); ap.add_argument('--n-shards',type=int,required=True); ap.add_argument('--limit',type=int); ap.add_argument('--out',default='stage24_outputs'); a=ap.parse_args()
    out=Path(a.out)/f'shard_{a.shard:02d}'; out.mkdir(parents=True,exist_ok=True); df=pd.read_csv('data/selected_pids.csv').sort_values('selected_rank');
    if a.limit: df=df.iloc[:a.limit]
    df=df.iloc[a.shard::a.n_shards]; one=base.ONE(base_url='https://openalyx.internationalbrainlab.org',username='intbrainlab',password='international',silent=True)
    base.FS_OUT=FS; base.DURATION_S=DURATION; base.TRAIN_FRACTION=.8; rows=[]; failures=[]
    for _,r in df.iterrows():
        pid=str(r.pid); t0=float(r.start_s); start=time.time(); details=one.get_details(str(r.eid)); subject=str(details.get('subject','unknown'))
        try:
            x,fs=base.load_destriped_250(one,pid,t0,DURATION); beh=load_behavior(one,str(r.eid)); rawbeh=decode(x,fs,beh,t0); rows.append({"pid":pid,"eid":str(r.eid),"subject":subject,"lab":str(r.get('lab','')),"k":384,"method":"raw","fs":fs,"values_per_second":384*fs,"reconstruction_r2":1.0,"rmse_uv":0.0,"behavior_json":json.dumps(rawbeh),"meta_json":"{}","runtime_s":time.time()-start,**{f'behavior_r2_{q}':v for q,v in rawbeh.items()}}); ndev=int(.8*x.shape[1]); p=PCA(n_components=max(KS),svd_solver='randomized',random_state=0).fit(x[:,:ndev].T); sc=p.transform(x.T)
            for k in KS:
                prec=(sc[:,:k]@p.components_[:k]+p.mean_).T.astype('float32'); pr2,prm=metrics(x,prec); pbeh=decode(prec,fs,beh,t0)
                seed=int(hashlib.sha256(f'{pid}-{k}'.encode()).hexdigest()[:8],16); crec,meta=train_codec(x,k,seed); cr2,crm=metrics(x,crec); cbeh=decode(crec,fs,beh,t0)
                for method,r2,rm,bs,m in [('pca',pr2,prm,pbeh,{}),('cnn',cr2,crm,cbeh,meta)]:
                    row={"pid":pid,"eid":str(r.eid),"subject":subject,"lab":str(r.get('lab','')),"k":k,"method":method,"fs":fs,"values_per_second":k*fs,"reconstruction_r2":r2,"rmse_uv":rm,"behavior_json":json.dumps(bs),"meta_json":json.dumps(m),"runtime_s":time.time()-start}; row.update({f'behavior_r2_{q}':v for q,v in bs.items()}); rows.append(row)
                del prec,crec; gc.collect()
            print(pid,'done',flush=True)
        except Exception as e:
            traceback.print_exc(); failures.append({"pid":pid,"error":f'{type(e).__name__}: {e}'})
    pd.DataFrame(rows).to_csv(out/'stage2_cnn.csv',index=False); (out/'stage2_failures.json').write_text(json.dumps(failures,indent=2));
    if not rows: raise RuntimeError('no Stage 2 result')
if __name__=='__main__': main()
