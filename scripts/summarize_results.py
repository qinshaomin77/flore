"""Export recorded metrics with source provenance; never pool different experiments."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='results')
    p.add_argument('--output',default='results/summary.csv')
    args=p.parse_args()
    source=(ROOT/args.input).resolve()
    output=(ROOT/args.output).resolve()
    rows=[]
    for file in sorted(source.rglob('eval_summary.json')):
        payload=json.loads(file.read_text(encoding='utf-8-sig'))
        row={'source':file.relative_to(source).as_posix()}
        for key,value in payload.items():
            if isinstance(value,(str,int,float,bool)) or value is None: row[key]=value
        rows.append(row)
    for file in sorted(source.rglob('episode_summary.csv')):
        with file.open(encoding='utf-8-sig',newline='') as handle:
            for record in csv.DictReader(handle):
                rows.append({'source':file.relative_to(source).as_posix(),**record})
    # Windowed evaluator uses episode-level network CSVs; retain run identity and interval.
    for file in sorted(source.rglob('network_300s.csv')):
        if 'all_episodes' in file.parts: continue
        with file.open(encoding='utf-8-sig',newline='') as handle: records=list(csv.DictReader(handle))
        for record in records:
            row={'source':file.relative_to(source).as_posix()}
            row.update(record)
            rows.append(row)
    if not rows: raise SystemExit(f'No evaluation summaries or network_300s.csv under {source}')
    fields=list(dict.fromkeys(key for row in rows for key in row))
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    print(f'{len(rows)} records -> {output}')

if __name__=='__main__': main()
