"""Plot explicitly selected result rows; no implicit pooling across experiments."""
import argparse
import csv
import math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='results/summary.csv')
    p.add_argument('--metric',required=True,help='Exact numeric column name from summary.csv')
    p.add_argument('--filter',default='',help='Only include source paths containing this text')
    p.add_argument('--ylabel',default=None,help='Metric label including its unit')
    p.add_argument('--output',default='results/figures/comparison')
    args=p.parse_args()
    with (ROOT/args.input).open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle)
        if args.metric not in (reader.fieldnames or []): p.error('Metric column does not exist: '+args.metric)
        rows=list(reader)
    selected=[]
    for row in rows:
        if args.filter not in row['source']: continue
        try: value=float(row[args.metric])
        except (ValueError,TypeError): continue
        if math.isfinite(value): selected.append((row['source'],value))
    if not selected: p.error('No matching finite numeric results')
    if len(selected)>40: p.error('More than 40 records; use --filter to select one comparison')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels,values=zip(*selected)
    fig,ax=plt.subplots(figsize=(8,max(3,len(values)*0.3)))
    ax.barh(range(len(values)),values,color='#276B9C')
    ax.set_yticks(range(len(values)),labels=labels,fontsize=7)
    ax.invert_yaxis();ax.set_xlabel(args.ylabel or args.metric)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    output=ROOT/args.output;output.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'): fig.savefig(output.with_suffix('.'+ext),dpi=300,bbox_inches='tight')
    plt.close(fig)
    print(output)
if __name__=='__main__': main()
