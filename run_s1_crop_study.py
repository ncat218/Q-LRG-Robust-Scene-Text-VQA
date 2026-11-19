"""Submit isolated old-checkpoint factorial or new S1b convergence training."""

import argparse
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

from run_all import REPO, WITH_SLURM, write_script


def prepare(args):
    if args.resume:
        output=args.resume.resolve()
        launch=json.loads((output/'launch.json').read_text())
        if launch['study'] != args.mode:
            raise ValueError('Resume mode differs from archived study')
    else:
        source=args.from_run.resolve()
        settings=json.loads((source/'launch.json').read_text())
        if not (source/'COMPLETE.json').is_file():
            raise ValueError('Source crop benchmark is incomplete')
        if not args.dry_run:
            (REPO/'runs').mkdir(exist_ok=True)
            output=Path(tempfile.mkdtemp(prefix=args.mode+'-',dir=REPO/'runs'))
        else:
            output=REPO/'runs'/('dry-run-'+args.mode)
        launch={**settings,'repo':str(REPO),'study':args.mode,'split':'official_val',
                'readers':['A0','A2'],'smoke_images':args.smoke_images,'flops_images':3,
                'selector_benchmark':settings['selector_benchmark'],'source':str(source)}
        if args.mode=='s1_convergence':
            if args.train_config is None:
                raise ValueError('Convergence training requires --train-config')
            launch['train_config']=json.loads(args.train_config.read_text(encoding='utf-8'))
            launch['selector_assets']=str(args.selector_assets.resolve()) if args.selector_assets else str(
                Path(settings['repo'])/'local/visionrl2')
            launch['train_view']=args.view
        if not args.dry_run:
            (output/'launch.json').write_text(json.dumps(launch,indent=2)+'\n',encoding='utf-8')
    required=[Path(launch['env'])/'bin/python',Path(launch['paths']),
              Path(launch['data_dir'])/'official_val.json',Path(launch['selector_benchmark'])/'COMPLETE.json']
    for arm in ('S1b','A2'):
        required.append(Path(launch['s1_run'])/'s1/variants'/arm/'checkpoint.pt')
    if not args.dry_run:
        for path in required:
            if not path.is_file():raise FileNotFoundError(path)
    return output,launch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('crop_matrix','s1_convergence'),required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--from-run',type=Path)
    group.add_argument('--resume',type=Path)
    parser.add_argument('--smoke-images',type=int)
    parser.add_argument('--train-config',type=Path)
    parser.add_argument('--selector-assets',type=Path)
    parser.add_argument('--view',choices=('full','crop','both'),default='both')
    parser.add_argument('--time')
    parser.add_argument('--dependency')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if os.environ.get('SLURM_JOB_ID'):parser.error('Submit from the login node')
    if args.smoke_images is not None and args.smoke_images<1:parser.error('Smoke size must be positive')
    output,launch=prepare(args)
    if not args.dry_run:
        (output/'logs').mkdir(exist_ok=True)
    resources=json.loads((REPO/'configs/slurm_hpc.json').read_text())
    python=Path(launch['env'])/'bin/python'
    commands=[
        'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"',
        'export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1',
        'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1',
        'export QLRG_ENV='+shlex.quote(launch['env']),
        'export QLRG_PATHS_CONFIG='+shlex.quote(launch['paths']),
        'export PYTHONPATH='+shlex.quote(str(REPO)),
        'nvidia-smi',
        'exec '+shlex.join([str(python),'-B','-m','scripts.'+args.mode,str(output)]),
    ]
    time=args.time or ('00:30:00' if launch.get('smoke_images') else
                       '2-00:00:00' if args.mode=='s1_convergence' else '16:00:00')
    script=output/(args.mode+'.sbatch')
    if not args.dry_run:script=write_script(output,args.mode,commands)
    command=['sbatch','--parsable',f'--chdir={REPO}',f'--job-name=qlrg-{args.mode.replace("_","-")}',
             f"--account={resources['account']}",f"--partition={resources['partition']}",
             f"--qos={resources['qos']}",f"--gres={resources['gres']}",
             '--cpus-per-task=8','--mem=48G',f'--time={time}',
             f'--output={output}/logs/%x-%j.out',f'--error={output}/logs/%x-%j.err']
    if args.dependency:command.append('--dependency=afterok:'+args.dependency)
    command.append(str(script))
    print('Run:',output,flush=True)
    print(shlex.join(command),flush=True)
    if args.dry_run:return
    # Avoid duplicate active submissions into one set of shards/checkpoints.
    jobfile=output/'job-id.txt'
    if jobfile.exists():
        old=jobfile.read_text().strip().split(';')[0]
        state=subprocess.check_output(['bash','-c',WITH_SLURM,'qlrg-slurm','squeue','-h','-j',old,'-o','%T'],text=True).strip()
        if state:raise RuntimeError(f'Run already has active job {old}: {state}')
    job=subprocess.check_output(['bash','-c',WITH_SLURM,'qlrg-slurm',*command],text=True).strip()
    jobfile.write_text(job+'\n')
    print('Job:',job,flush=True)


if __name__=='__main__':main()
