"""Train matched full/crop visual modules through the fixed 11305 A2 reader."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from filelock import FileLock

from qlrg.common import dataset_paths, object_hash, read_config, read_json, read_jsonl, sha256, write_json, write_jsonl
from qlrg.evaluation import evaluator, paired_bootstrap
from qlrg.experiment import verified_complete
from qlrg.qwen_wrapper import Qwen
from qlrg.s1_manifests import read_s1_manifests
from qlrg.s1_model import text_training
from qlrg.s1_run import make_adapter, runtime_shared
from scripts.followup_common import bind, write_table
from scripts.s1_convergence import (Features, evaluate, merged_train, prepare_proposals,
                                    restoration_objective, train_stage, validate_config)


METHODS=('restoration','ce_only')


def validate_study(settings):
    validate_config(settings)
    if settings.get('reader')!='A2' or settings.get('freeze_a2') is not True:
        raise ValueError('This study requires the fixed A2 reader')
    if settings.get('methods')!=list(METHODS):
        raise ValueError('Run both matched feature-restoration and CE-only controls')


def method_config(base,method):
    if method not in METHODS:raise ValueError('Unknown visual-module method')
    return {**base,'feature_weight':base['feature_weight'] if method=='restoration' else 0.}


def module_digest(module):
    digest=hashlib.sha256()
    for name,tensor in sorted(module.state_dict().items()):
        digest.update(json.dumps([name,str(tensor.dtype),list(tensor.shape)]).encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def assert_fixed_reader(model,lora,expected):
    if any(p.requires_grad or p.grad is not None for p in lora.parameters()):
        raise RuntimeError('A2 must remain frozen without parameter gradients')
    if any(p.requires_grad or p.grad is not None for p in model.model.parameters()):
        raise RuntimeError('Base Qwen must remain frozen without parameter gradients')
    if module_digest(lora)!=expected:
        raise RuntimeError('Frozen A2 weights changed')


def load_a2(source,trained,expected_examples):
    path=source/'variants/A2/checkpoint.pt'
    digest=sha256(path)
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if (checkpoint['arm']!='A2' or checkpoint['binding']!=object_hash(trained) or
        checkpoint['examples']!=expected_examples or not checkpoint['history'] or
        checkpoint['history'][-1]['examples']!=expected_examples or sha256(path)!=digest):
        raise ValueError('A2 checkpoint does not match the locked training run')
    provenance={'path':str(path),'sha256':digest,'training_binding':checkpoint['binding'],
                'example_visits':expected_examples,'frozen':True}
    return checkpoint['adapter'],provenance


def baseline_once(model,adapter,cache,dev,path,binding):
    if (path/'COMPLETE.json').exists():
        done=read_json(path/'COMPLETE.json')
        if done['binding']!=object_hash(binding):raise ValueError('A2 baseline binding changed')
        for name,digest in done['artifacts'].items():
            if sha256(path/name)!=digest:raise ValueError('A2 baseline artifact changed')
        result=read_json(path/'summary.json'),read_jsonl(path/'predictions.jsonl')
    else:
        result=evaluate(model,adapter,cache,dev,evaluator(),baseline=True)
        write_json(path/'summary.json',result[0]);write_jsonl(path/'predictions.jsonl',result[1])
        write_json(path/'COMPLETE.json',{'binding':object_hash(binding),
                   'artifacts':{name:sha256(path/name) for name in ('summary.json','predictions.jsonl')}})
    expected={(im['image_id'],q['question_id'],c) for im in dev['images'] for q in im['questions'] for c in ('CLEAN','LR4')}
    actual=[(r['image_id'],r['question_id'],r['condition']) for r in result[1]]
    if len(actual)!=len(expected) or set(actual)!=expected:raise ValueError('A2 baseline cohort differs')
    return result


def gradient_probe(model,adapter,cache,config,question):
    prompt=cache.prompt('LR4',question['question'])
    with torch.no_grad():
        value=adapter(cache.features['LR4'],cache.grid)[0].to(model.dtype)
        if not torch.equal(model.prefill_logits(prompt),model.prefill_logits(prompt,value)):
            raise RuntimeError('S1b zero-initialization changed fixed-A2 logits')
    answer=next(a.strip() for a in question['answers'] if a.strip())
    with text_training(model.model,True):
        loss,match,task,residual=restoration_objective(model,adapter,cache,question,'LR4',answer,config)
        loss.backward()
        norm=float(torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.,error_if_nonfinite=True))
        if norm<=0:raise RuntimeError('No visual-module gradients through frozen A2')
        adapter.zero_grad(set_to_none=True)
    return {'passed':True,'grad_norm':norm,'loss':float(loss.detach()),
            'task_loss':float(task.detach()),'feature_mse':float(match.detach()),
            'feature_weight':config['feature_weight'],'reader':'A2','frozen_reader':True}


def compare_methods(destination,dev,baseline):
    """Exploratory validation contrasts, not a locked final evaluation."""
    indexes={'A2':{(r['image_id'],r['question_id'],r['condition']):r for r in baseline[1]}}
    selections={}
    for method in METHODS:
        directory=destination/method/'selection'
        done=read_json(directory/'COMPLETE.json');selections[method]=done
        predictions=directory/('best_predictions.jsonl' if done['selection']['best_examples'] else 'baseline_predictions.jsonl')
        indexes[method]={(r['image_id'],r['question_id'],r['condition']):r for r in read_jsonl(predictions)}
    rows=[]
    for condition in ('CLEAN','LR4'):
        cases=[(im['image_id'],q['question_id'],condition) for im in dev['images'] for q in im['questions']]
        groups=[key[0] for key in cases]
        for left,right in (('A2','restoration'),('A2','ce_only'),('ce_only','restoration')):
            rows.append({'condition':condition,'reference':left,'target':right,
                         'questions':len(cases),**paired_bootstrap([indexes[left][k]['score'] for k in cases],
                         [indexes[right][k]['score'] for k in cases],groups,10000,42)})
    write_table(destination/'validation_comparisons.csv',rows)
    write_json(destination/'validation_comparisons.json',{'role':'exploratory selected-checkpoint validation',
               'comparisons':rows,'selection_examples':{m:s['selection']['best_examples'] for m,s in selections.items()},
               'note':'Separate early stopping can select different durations; CIs condition on selected checkpoints.'})


def run_view(output,view):
    launch=read_json(output/'launch.json');settings=launch['train_config'];validate_study(settings)
    paths=dataset_paths(read_config(launch['paths']))
    model_config=read_config(Path(launch['repo'])/'configs/model_qwen.yaml')
    manifests,hashes=read_s1_manifests(Path(launch['data_dir']),paths['textvqa'])
    all_train=merged_train(manifests)
    source=Path(launch['s1_run'])/'s1';trained=verified_complete(source,'s1')
    if hashes!=trained['dataset_hashes'] or model_config!=trained['shared']['model']:
        raise ValueError('Model or datasets differ from the locked A2 source')
    expected_examples=sum(len(im['questions']) for im in manifests['train_inner']['images'])*trained['config']['epochs']
    a2_state,a2_provenance=load_a2(source,trained,expected_examples)
    records=manifests['train_inner']['images'];dev=manifests['val_inner']
    if launch.get('smoke_images'):
        records=records[:launch['smoke_images']];dev={**dev,'images':dev['images'][:launch['smoke_images']]}
        all_train={**all_train,'images':records+dev['images']}
        settings={**settings,'max_epochs':1,'min_epochs':1,'evaluate_every_epoch_fraction':1.,'refit_official_train':False}
    destination=output/view;destination.mkdir(exist_ok=True)
    with FileLock(str(destination/'.run.lock'),timeout=0):
        proposal_source=prepare_proposals(launch,paths,model_config,all_train['images'],output) if view=='crop' else None
        torch.manual_seed(settings['seed']);torch.backends.cuda.matmul.allow_tf32=False
        model=Qwen(model_config,paths['hf_hub'])
        lora=make_adapter(model,'A2',trained['config']).eval().requires_grad_(False)
        lora.load_state_dict(a2_state,strict=True);a2_digest=module_digest(lora)
        base_config={**trained['config'],'seed':settings['seed'],'learning_rate':settings['learning_rate'],
                     'accumulate':settings['accumulate']}
        common={'stage':'s1_a2_convergence','reader':'A2','view':view,'settings':settings,
                'base_config':base_config,'shared':runtime_shared(paths,model_config),'a2':a2_provenance,
                'a2_state_digest':a2_digest,'train_manifest_hash':object_hash(records),
                'dev_manifest_hash':object_hash(dev),'official_train_hash':object_hash(all_train),
                'proposal_source':proposal_source,'worker_sha256':sha256(Path(__file__)),
                'engine_sha256':sha256(Path(__file__).with_name('s1_convergence.py')),
                'policy_sha256':sha256(Path(__file__).with_name('convergence_policy.py'))}
        bind(destination,common)
        write_json(destination/'train_manifest.json',{'images':records})
        write_json(destination/'official_train_manifest.json',all_train)
        cache=Features(model,paths['textvqa'],view,output);cache.label=f'A2/{view}'
        baseline_adapter=make_adapter(model,'S1b',base_config)
        results={}
        with lora.attached():
            assert_fixed_reader(model,lora,a2_digest)
            cache.load(records[0],records[0]['questions'][0])
            initial_probe=gradient_probe(model,baseline_adapter,cache,base_config,records[0]['questions'][0])
            assert_fixed_reader(model,lora,a2_digest)
            write_json(destination/'reader_probe.json',initial_probe)
            write_json(output/'progress.json',{'view':view,'stage':'frozen_A2_baseline'})
            baseline=baseline_once(model,baseline_adapter,cache,dev,destination/'baseline',common)
            del baseline_adapter
            for method in METHODS:
                cfg=method_config(base_config,method)
                binding={**common,'method':method,'base_config':cfg}
                work=destination/method;work.mkdir(exist_ok=True)
                bind(work,binding)
                write_json(output/'progress.json',{'view':view,'method':method,'stage':'selection'})
                print(f'STAGE A2/{view}/{method}: selection, feature_weight={cfg["feature_weight"]}',flush=True)
                adapter=make_adapter(model,'S1b',cfg);cache.label=f'A2/{view}/{method}'
                cache.load(records[0],records[0]['questions'][0])
                probe=gradient_probe(model,adapter,cache,cfg,records[0]['questions'][0])
                assert_fixed_reader(model,lora,a2_digest)
                write_json(work/'gradient_probe.json',probe)
                selection=train_stage(model,cache,adapter,records,dev,cfg,settings,work/'selection',binding,
                                      baseline_result=baseline,reader_guard=lambda:assert_fixed_reader(model,lora,a2_digest))
                assert_fixed_reader(model,lora,a2_digest)
                result={'method':method,'selection':selection,'refit':None,'a2_unchanged':True,
                        'official_val_evaluated':False}
                if settings['refit_official_train']:
                    visits=selection['selection']['best_examples']
                    count=sum(len(im['questions']) for im in all_train['images'])
                    target=round(visits/selection['examples_per_epoch']*count/settings['accumulate'])*settings['accumulate']
                    if target:
                        write_json(output/'progress.json',{'view':view,'method':method,'stage':'official_train_refit'})
                        print(f'STAGE A2/{view}/{method}: refit, {target} visits',flush=True)
                        adapter=make_adapter(model,'S1b',cfg)
                        refit_binding={**binding,'selected_examples':visits,
                                       'selected_checkpoint_sha256':selection['checkpoint_sha256'],'refit_visits':target}
                        result['refit']=train_stage(model,cache,adapter,all_train['images'],None,cfg,settings,
                                                   work/'refit',refit_binding,refit_visits=target,
                                                   reader_guard=lambda:assert_fixed_reader(model,lora,a2_digest))
                        assert_fixed_reader(model,lora,a2_digest)
                    else:result['refit']={'skipped':'No trained checkpoint beat the eligible frozen-A2 baseline.'}
                result['a2_state_digest_after']=module_digest(lora)
                write_json(work/'summary.json',result)
                write_json(work/'COMPLETE.json',{'binding':object_hash(binding),'summary_sha256':sha256(work/'summary.json')})
                results[method]=result
                del adapter
            compare_methods(destination,dev,baseline)
            assert_fixed_reader(model,lora,a2_digest)
        assert_fixed_reader(model,lora,a2_digest)
        write_json(destination/'summary.json',results)
        write_json(destination/'COMPLETE.json',{'stage':'s1_a2_convergence','binding':object_hash(common),
                                             'summary_sha256':sha256(destination/'summary.json')})


def run(output):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('A2 restoration training requires a Slurm allocation')
    launch=read_json(output/'launch.json')
    if launch['study']!='s1_a2_convergence':raise ValueError('Wrong study')
    validate_study(launch['train_config'])
    views=('full','crop') if launch['train_view']=='both' else (launch['train_view'],)
    with FileLock(str(output/'.group.lock'),timeout=0):
        for view in views:
            subprocess.run([sys.executable,'-B','-m','scripts.s1_a2_convergence',str(output),'--worker-view',view],check=True)
        write_json(output/'progress.json',{'stage':'complete'})
        write_json(output/'COMPLETE.json',{'stage':'s1_a2_convergence','views':list(views),
                                         'summaries':{v:sha256(output/v/'summary.json') for v in views}})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--worker-view',choices=('full','crop'))
    args=parser.parse_args()
    if args.worker_view:run_view(args.output,args.worker_view)
    else:run(args.output)
