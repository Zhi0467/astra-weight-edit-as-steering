import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser(description="Build the blog's pooled Astra chart with current arm names.")
parser.add_argument("--summary", required=True, help="Combined test_summary_with_clasd.json")
parser.add_argument("--out", required=True, type=Path)
parser.add_argument("--include-positive", action="store_true", help="Retain positive-only NTP for the supplement")
args = parser.parse_args()
S = json.loads(Path(args.summary).read_text())
O = args.out
O.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':'#243247','axes.labelcolor':'#243247','xtick.color':'#8190a5','ytick.color':'#243247','svg.fonttype':'path'})
fig,axs=plt.subplots(2,2,figsize=(11,7.4));fig.subplots_adjust(left=.21,right=.96,top=.90,bottom=.07,hspace=.55,wspace=.25)
labels=['Original model','Astra’s steering\n'+r'$s=\alpha v$','Unlikelihood','Corrective NTP'];colors=['#9aa3b1','#508dd3','#26a996','#a27aca']
if args.include_positive:
 labels.insert(3,"Positive-only NTP"); colors.insert(3,"#e9a23e")
for row,n in enumerate([4,32]):
 arms=['control',next(a for a in S['per_arm'] if a.startswith(f'sv_n{n}_')),f'ul_n{n}',f'clasd_n{n}']
 if args.include_positive: arms.insert(3,f'clas_n{n}')
 for col,metric in enumerate(['loop','accuracy']):
  ax=axs[row,col];v=[S['per_arm'][a]['pooled'][metric]*100 for a in arms]
  ax.barh(range(len(arms)),v,color=[matplotlib.colors.to_rgba(c,.38) for c in colors],edgecolor=colors,height=.52)
  for i,(val,c) in enumerate(zip(v,colors)): ax.text(val+2,i,f'{val:.1f}',va='center',color=c,fontweight='bold',fontsize=10)
  ax.set_xlim(0,105);ax.set_xticks([0,25,50,75,100]);ax.set_yticks(range(len(arms)),labels if col==0 else ['']*len(arms));ax.invert_yaxis();ax.tick_params(length=0,pad=8)
  ax.set_title(['Loop rate (%) ↓','Accuracy (%) ↑'][col],loc='left',pad=14,fontsize=12);ax.grid(axis='x',color='#edf0f5');ax.set_axisbelow(True)
  for sp in ax.spines.values():sp.set_visible(False)
 axs[row,0].text(-.50,1.19,f'{n} pairs',transform=axs[row,0].transAxes,fontsize=15,fontweight='bold')
fig.savefig(O/('astra_pooled_summary_with_positive_ntp.svg' if args.include_positive else 'astra_pooled_summary.svg'))
