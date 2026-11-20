"""Audit an archived 32-arm run and measure reader/restorer interactions on CPU."""

import argparse
import csv
import io
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from qlrg.common import atomic_write, object_hash, read_json, read_jsonl, sha256, write_json
from qlrg.evaluation import paired_bootstrap


def write_table(path, rows):
    contents = io.StringIO()
    writer = csv.DictWriter(contents, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, contents.getvalue().encode('utf-8'))


def analyze(source, output):
    complete, binding = read_json(source/'COMPLETE.json'), read_json(source/'run.json')
    if complete['binding'] != object_hash(binding):
        raise ValueError('Invalid run binding')
    for name, digest in complete['artifacts'].items():
        if sha256(source/name) != digest:
            raise ValueError(f'Artifact changed: {name}')
    by_arm = defaultdict(dict)
    for row in read_jsonl(source/'predictions.jsonl'):
        key = row['image_id'], row['question_id']
        if key in by_arm[row['arm']]:
            raise ValueError('Duplicate image/question/arm')
        by_arm[row['arm']][key] = row
    keys = sorted(next(iter(by_arm.values())))
    if len(by_arm) != 32 or len(keys) != 5000 or any(sorted(a) != keys for a in by_arm.values()):
        raise ValueError('Expected the same 5,000 questions in 32 arms')
    groups = [iid for iid, qid in keys]
    if len(set(groups)) != 3166:
        raise ValueError('Wrong number of images')
    summary = read_json(source/'summary.json')
    accuracy = {row['arm']: row['accuracy_pct'] for row in summary['accuracy']}
    scores = {name: np.asarray([index[k]['score'] for k in keys]) for name,index in by_arm.items()}
    for name, values in scores.items():
        if not math.isclose(100*float(values.mean()), accuracy[name], abs_tol=1e-8):
            raise ValueError(f'Summary does not reproduce from predictions: {name}')
    interactions, cases = [], []
    for budget in ('fair576','native'):
        for layout, flags in [('full','1'),('crop','1'),('fusion','10'),('fusion','01'),('fusion','11')]:
            off = '00' if layout == 'fusion' else '0'
            names = [f'{reader}__{budget}__{layout}__{state}'
                     for reader,state in [('A0',off),('A0',flags),('A2',off),('A2',flags)]]
            a0,s1b,a2,both = [scores[name] for name in names]
            interaction = (both-a2)-(s1b-a0)
            value = {'budget':budget,'layout':layout,'restoration_flags':flags,
                     's1b_effect_A0_pp':float(100*(s1b-a0).mean()),
                     's1b_effect_A2_pp':float(100*(both-a2).mean()),
                     **paired_bootstrap(np.zeros(len(keys)),interaction,groups,10000,42),
                     'S1b_helps_A0_hurts_A2':int(sum((s1b>a0)&(both<a2))),
                     'A0_0to1_and_A2_1to0':int(sum((a0==0)&(s1b==1)&(a2==1)&(both==0))),
                     'A0_gains_already_solved_by_A2':int(sum((s1b>a0)&(a2==1))),
                     'A0_gains_total':int(sum(s1b>a0))}
            interactions.append(value)
            for i,key in enumerate(keys):
                if s1b[i]>a0[i] and both[i]<a2[i]:
                    cases.append({'budget':budget,'layout':layout,'restoration_flags':flags,
                                  'image_id':key[0],'question_id':key[1],
                                  **{name:by_arm[name][key]['answer'] for name in names},
                                  'cause':'unreviewed'})
    output.mkdir(parents=True,exist_ok=True)
    write_table(output/'interactions.csv',interactions)
    # Different layouts have different arm-name columns; use consistent serialized names.
    if cases:
        import json
        flattened = [{k:v for k,v in c.items() if '__' not in k} |
                     {'answers_json':json.dumps({k:v for k,v in c.items() if '__' in k},ensure_ascii=False)} for c in cases]
        write_table(output/'interaction_error_cases.csv',flattened)
    result = {'stage':'crop_matrix_posthoc','questions':len(keys),'images':len(set(groups)),
              'predictions':len(keys)*len(by_arm),'source_binding':complete['binding'],
              'source_summary_sha256':sha256(source/'summary.json'),
              'all_source_artifacts_verified':True,'all_accuracy_reproduced':True,
              'interactions':interactions,
              'interpretation':'Difference of S1b score effects between readers, not proof of a feature-level cause.',
              'uncertainty':'Pointwise image-cluster CI conditional on these checkpoints; no training-seed variance or multiplicity correction.'}
    write_json(output/'summary.json',result)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.run,args.output)
    print(f"Verified {result['predictions']} predictions; wrote {len(result['interactions'])} interactions")
