"""Stage 4 runner: PCA components by spectrum, anatomy, behavior, and trials."""
from __future__ import annotations
import argparse, json, traceback
from pathlib import Path
import numpy as np, pandas as pd, scipy.ndimage, scipy.stats
from sklearn.decomposition import PCA
from brainbox.io.one import SpikeSortingLoader
from track_a_rate_distortion import run_track_a as base
from stages_1_to_4.stage4_helpers import pearson,bh,regions,spectral,trial_col,rmswin,spatial_metrics
DURATION=100.; FS=500.; NPC=32; STEP=.04

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--shard',type=int,required=True); ap.add_argument('--n-shards',type=int,required=True); ap.add_argument('--limit',type=int); ap.add_argument('--out',default='stage24_outputs'); a=ap.parse_args(); out=Path(a.out)/f'shard_{a.shard:02d}'; out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv('data/selected_pids.csv').sort_values('selected_rank');
    if a.limit:df=df.iloc[:a.limit]
    df=df.iloc[a.shard::a.n_shards]; one=base.ONE(base_url='https://openalyx.internationalbrainlab.org',username='intbrainlab',password='international',silent=True); base.FS_OUT=FS; base.DURATION_S=DURATION; rows=[]; brows=[]; trows=[]; fails=[]
    for _,r in df.iterrows():
        pid=str(r.pid); t0=float(r.start_s); subject=str(one.get_details(str(r.eid)).get('subject','unknown'))
        try:
            x,fs=base.load_destriped_250(one,pid,t0,DURATION); ndev=int(.8*x.shape[1]); pca=PCA(n_components=NPC,svd_solver='randomized',random_state=0).fit(x[:,:ndev].T); scores=pca.transform(x.T); reg=regions(SpikeSortingLoader(pid=pid,one=one).load_channels()); beh=base.load_behavior_series(one,str(r.eid)); times=np.arange(.5,DURATION-.5+1e-9,STEP); Y,names=base.behavior_at_times(beh,t0+times); idx=np.clip(np.round(times*fs).astype(int),0,len(scores)-1); test=times>=80.5
            try:trials=pd.DataFrame(one.load_object(str(r.eid),'trials',collection='alf'))
            except:trials=pd.DataFrame()
            stim=trial_col(trials,('stimOn_times','stimOn')); move=trial_col(trials,('firstMovement_times','firstMovement')); feedback=trial_col(trials,('feedback_times','feedbackTime')); choice=trial_col(trials,('choice',)); ftype=trial_col(trials,('feedbackType','feedback_type')); rt=move-stim if move is not None and stim is not None else None
            for pc in range(NPC):
                load=pca.components_[pc]; row={'pid':pid,'eid':str(r.eid),'subject':subject,'lab':str(r.get('lab','')),'pc':pc+1,'explained_variance_ratio':float(pca.explained_variance_ratio_[pc]),'cumulative_variance':float(pca.explained_variance_ratio_[:pc+1].sum()),**spatial_metrics(load,reg),**spectral(scores[:ndev,pc],fs)}; rows.append(row)
                signed=scipy.ndimage.uniform_filter1d(scores[:,pc],size=max(1,int(.04*fs)),mode='nearest')[idx]; logr=np.log10(np.sqrt(scipy.ndimage.uniform_filter1d(scores[:,pc]**2,size=max(1,int(.2*fs)),mode='nearest'))[idx]+1e-12)
                for j,name in enumerate(names):
                    for feature,v in [('signed_40ms',signed),('log_rms_200ms',logr)]:
                        rr,pp,nn=pearson(v[test],Y[test,j]); brows.append({'pid':pid,'eid':str(r.eid),'subject':subject,'pc':pc+1,'target':name,'feature':feature,'r':rr,'p':pp,'n':nn})
                if not trials.empty:
                    rec=[]
                    for i in range(len(trials)):
                        z={}
                        for en,ev in [('stim',stim),('move',move),('feedback',feedback)]:
                            if ev is None or i>=len(ev) or not np.isfinite(ev[i]) or ev[i]<t0+80 or ev[i]>t0+100:z[en+'pre']=z[en+'post']=np.nan
                            else:rel=ev[i]-t0; z[en+'pre']=rmswin(scores[:,pc],fs,rel,-.5,0); z[en+'post']=rmswin(scores[:,pc],fs,rel,0,.5)
                        z['rt']=rt[i] if rt is not None and i<len(rt) else np.nan; z['choice']=choice[i] if choice is not None and i<len(choice) else np.nan; z['ftype']=ftype[i] if ftype is not None and i<len(ftype) else np.nan; rec.append(z)
                    q=pd.DataFrame(rec)
                    for en in ('stim','move','feedback'):
                        d=q[en+'post']-q[en+'pre']; g=np.isfinite(d); trows.append({'pid':pid,'eid':str(r.eid),'subject':subject,'pc':pc+1,'association':en+'_post_minus_pre_rms','value':float(d[g].mean()) if g.any() else np.nan,'p':float(scipy.stats.ttest_1samp(d[g],0).pvalue) if g.sum()>=3 else np.nan,'n':int(g.sum())})
                    for nm,xn,yn in [('reaction_vs_pre_move','rt','movepre'),('choice_vs_pre_move','choice','movepre'),('feedback_vs_post_feedback','ftype','feedbackpost')]:
                        rr,pp,nn=pearson(q[xn].to_numpy(float),q[yn].to_numpy(float)); trows.append({'pid':pid,'eid':str(r.eid),'subject':subject,'pc':pc+1,'association':nm,'value':rr,'p':pp,'n':nn})
            print(pid,'done',flush=True)
        except Exception as e:traceback.print_exc(); fails.append({'pid':pid,'error':f'{type(e).__name__}: {e}'})
    for arr in (brows,trows):
        for z,v in zip(arr,bh([q['p'] for q in arr])):z['q_bh_global']=v
    pd.DataFrame(rows).to_csv(out/'stage4_components.csv',index=False); pd.DataFrame(brows).to_csv(out/'stage4_behavior.csv',index=False); pd.DataFrame(trows).to_csv(out/'stage4_trials.csv',index=False); (out/'stage4_failures.json').write_text(json.dumps(fails,indent=2));
    if not rows:raise RuntimeError('no Stage 4 result')
if __name__=='__main__':main()
