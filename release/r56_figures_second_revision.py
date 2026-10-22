#!/usr/bin/env python
"""R56 second formatting revision. Read existing R55 summary exports only.
The historical scatter actually used posterior_mean, not q50. This revision
preserves those coordinates and labels them honestly pending a median decision.
PP curves are marginal posterior-rank ECDFs (not central-interval coverage).
"""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.stats import kstest

ROOT = Path(__file__).resolve().parent
REMOTE = (ROOT/'workspace').is_dir()
DEFAULT_OUT = ROOT / ('workspace/outputs/r56' if REMOTE else 'outputs/r56')
DEFAULT_DATA = ROOT / ('workspace/outputs/r55/t2_main' if REMOTE else 'outputs/r56/data')
SEEDS = (20265921,11,22,33,44,55)
LABELS = [r'$\log_{10}r$',r'$n_\mathrm{t}$',r'$\log_{10}\kappa_{10}$',r'$\log_{10}T_{\rm re}$',r'$\Delta N_{\rm re}$']
PARAMS = ['log10 r','n_t','log10 kappa10','log10 T_re','DN_re']
BOX = np.array([[-40,0],[-1,6],[-7,3],[-3,7],[0,40]],float)
COL = ['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4']
MARKS = ['o','s','^','D','v']
STYLES = ['solid','dashed','dashdot','dotted',(0,(4,1,1,1))]
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
                     'legend.fontsize':7,'font.family':'DejaVu Sans','pdf.fonttype':42})


def filename(tier,seed):
    return f'U1c_s1_main_now_noise{seed}_Median5000_plus.npz' if tier=='now' else f'U1c_fut_s1_main_fut_noise{seed}_Median5000_plus.npz'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,default=DEFAULT_OUT)
    ap.add_argument('--data',type=Path,default=DEFAULT_DATA)
    a=ap.parse_args()
    out=a.output.resolve(); data=a.data.resolve();figdir=out/'figures';figdir.mkdir(parents=True,exist_ok=True)
    diagnostics={'scatter':{},'coverage':{},'pp':{}}

    def save(fig,stem):
        # All scatter artists are rasterized at 300 dpi in PDF; axes/text remain vector.
        fig.savefig(figdir/(stem+'.pdf'),dpi=300,bbox_inches='tight')
        fig.savefig(figdir/(stem+'.png'),dpi=180,bbox_inches='tight')
        plt.close(fig)

    # T2: preserve old mean-based scatter values; correct false median label.
    fig,axes=plt.subplots(2,5,figsize=(7.1,4.2))
    fig.subplots_adjust(left=.09,right=.88,bottom=.12,top=.88,hspace=.32,wspace=.34)
    cmap=LinearSegmentedColormap.from_list('snr_blue',['#cde2fb','#2a78d6','#0d366b'])
    norm=LogNorm(1,1000)
    scatter_mappable=None
    for row,tier in enumerate(('now','future')):
        with np.load(data/filename(tier,20265921),allow_pickle=True) as z:
            truth=z['true'][:,:5]; point=z['posterior_mean'][:,:5]; snr=z['snr_combined']
            low=snr<=1; high=snr>1; over10=snr>10
            order=np.argsort(snr[high],kind='stable')
            r2=1-np.sum((point[over10]-truth[over10])**2,axis=0)/np.sum((truth[over10]-truth[over10].mean(0))**2,axis=0)
            diagnostics['scatter'][tier]={'source_seed':20265921,'point_estimator':'posterior_mean (historical coordinates retained)',
                                           'n_total':len(snr),'n_gt1':int(high.sum()),'n_gt10':int(over10.sum()),'r2_gt10':r2.tolist(),
                                           'points_outside_plot_prior_box':((point<BOX[:,0])|(point>BOX[:,1])).sum(0).tolist()}
            for i,ax in enumerate(axes[row]):
                ax.scatter(truth[low,i],point[low,i],s=4,facecolors='none',edgecolors='#a8a69f',linewidths=.25,alpha=.45,rasterized=True)
                scatter_mappable=ax.scatter(truth[high,i][order],point[high,i][order],s=6,c=snr[high][order],cmap=cmap,norm=norm,edgecolors='none',alpha=.65,rasterized=True)
                ax.plot(BOX[i],BOX[i],'--',color='#777',lw=.65)
                ax.set_xlim(BOX[i]);ax.set_ylim(BOX[i]);ax.tick_params(pad=2)
                if row==1:ax.set_xlabel('true '+LABELS[i])
                if i==0:ax.set_ylabel(('Current' if tier=='now' else 'Future')+'\nposterior mean')
                ax.text(.035,.955,f'SNR>1: {int(high.sum()):,}\n'+r'$R^2_{>10}$'+f' = {r2[i]:.2f}',transform=ax.transAxes,va='top',fontsize=5.6)
    handles=[Line2D([],[],marker='o',linestyle='none',markerfacecolor='none',markeredgecolor='#a8a69f',markersize=4,label=r'SNR $\leq1$'),
             Line2D([],[],marker='o',linestyle='none',markerfacecolor='#2a78d6',markeredgecolor='none',markersize=4,label=r'SNR $>1$')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.48,1.0),ncol=2,frameon=False)
    cax=fig.add_axes([.9,.2,.018,.59])
    cb=fig.colorbar(scatter_mappable,cax=cax,extend='max');cb.set_label('combined SNR')
    # Matplotlib colorbar solids are rasterized by default; leave them vector.
    if cb.solids is not None:cb.solids.set_rasterized(False)
    save(fig,'r56_scatter_truth_vs_median')

    # T3: aggregate final point is not a fifth disjoint SNR bin.
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.0),sharey=True)
    fig.subplots_adjust(left=.09,right=.99,bottom=.18,top=.89,wspace=.13)
    for ax,tier in zip(axes,('now','future')):
        values=[]
        for seed in SEEDS:
            with np.load(data/filename(tier,seed),allow_pickle=True) as z:
                s=z['snr_combined'];t=z['true'][:,:5];q0=z['q05'][:,:5];q1=z['q95'][:,:5]
                masks=[s<=1,(s>1)&(s<=10),(s>10)&(s<=100),s>100,s>1]
                values.append([((t[m]>=q0[m])&(t[m]<=q1[m])).mean(0) for m in masks])
        v=np.asarray(values);mean=v.mean(0);lo=v.min(0);hi=v.max(0)
        diagnostics['coverage'][tier]={'mean':mean.tolist(),'min':lo.tolist(),'max':hi.tolist(),'noise_seeds':list(SEEDS)}
        for i in range(5):
            x=np.arange(5)+(i-2)*.045
            ax.errorbar(x[:4],mean[:4,i],yerr=np.vstack([mean[:4,i]-lo[:4,i],hi[:4,i]-mean[:4,i]]),color=COL[i],marker=MARKS[i],linestyle=STYLES[i],ms=3,lw=.8,label=LABELS[i])
            ax.errorbar(x[4:],mean[4:,i],yerr=np.vstack([mean[4:,i]-lo[4:,i],hi[4:,i]-mean[4:,i]]),color=COL[i],marker=MARKS[i],linestyle='none',ms=3,lw=.8)
        ax.axhline(.9,color='#777',ls='--',lw=.8);ax.axhline(.85,color='#bbb',ls=':',lw=.8)
        ax.axvline(3.5,color='#999',ls='--',lw=.7)
        ax.set_xticks(np.arange(5),[r'$<1$',r'$1$--$10$',r'$10$--$100$',r'$>100$',r'$>1$ (all)'])
        ax.set_ylim(.78,1.0);ax.set_xlabel('combined SNR');ax.set_title('Current' if tier=='now' else 'Future')
    axes[0].set_ylabel('90% empirical coverage');axes[1].legend(frameon=False,ncol=2,loc='lower left',fontsize=7)
    save(fig,'r56_coverage_vs_snr')

    # T4: marginal ranks, not central credible-interval containment.
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.4),sharey=True)
    fig.subplots_adjust(left=.09,right=.99,bottom=.16,top=.89,wspace=.15)
    for ax,tier in zip(axes,('now','future')):
        stats=np.zeros((6,5));n=None
        for si,seed in enumerate(SEEDS):
            with np.load(data/filename(tier,seed),allow_pickle=True) as z:
                m=z['snr_combined']>10;ranks=z['ranks'][m,:5];n=int(m.sum())
                for i in range(5):
                    x=np.sort(ranks[:,i]);y=(np.arange(1,len(x)+1)-.5)/len(x)
                    ax.plot(x,y,color=COL[i],linestyle=STYLES[i],lw=.5,alpha=.35)
                    stats[si,i]=kstest(ranks[:,i],'uniform').statistic
        diagnostics['pp'][tier]={'n_gt10':n,'noise_seeds':list(SEEDS),'ks_D_by_seed':stats.tolist(),'ks_D_mean':stats.mean(0).tolist(),
                                  'previous_label_D_last_seed55':stats[-1].tolist(),
                                  'definition':'ECDF of true-parameter marginal posterior ranks; not central interval coverage',
                                  'D_definition':'one two-sided one-sample uniform KS statistic per realization, then arithmetic mean'}
        for i in range(5):
            yy=.955-i*.055
            ax.plot([.03,.08],[yy,yy],color=COL[i],linestyle=STYLES[i],lw=1.2,transform=ax.transAxes)
            ax.text(.095,yy,LABELS[i]+f': D={stats[:,i].mean():.3f}',transform=ax.transAxes,va='center',fontsize=6,color='#222')
        ax.plot([0,1],[0,1],'--',color='#777',lw=.8);ax.set_xlim(0,1);ax.set_ylim(0,1)
        ax.set_xlabel('marginal posterior quantile of truth');ax.set_title('Current' if tier=='now' else 'Future')
    axes[0].set_ylabel('empirical cumulative fraction')
    save(fig,'r56_pp_snr_gt10')

    diagnostics['figure_files']={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in figdir.glob('*') if p.suffix in ('.pdf','.png')}
    (out/'r56_plot_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n',encoding='utf-8')
    print('FIGURES_DONE',diagnostics['figure_files']['r56_scatter_truth_vs_median.pdf']['bytes'])


if __name__=='__main__':
    main()
