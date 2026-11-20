"""Resume an inactive training run without changing its original workers or checkpoints."""

import argparse
import getpass
import json
from pathlib import Path
import shlex
import subprocess

from run_all import REPO, WITH_SLURM, write_script


def slurm(*args):
    return subprocess.check_output(['bash','-c',WITH_SLURM,'qlrg-slurm',*args],text=True).strip()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--time',default='3-00:00:00')
    args=parser.parse_args();run=args.run.resolve()
    launch=json.loads((run/'launch.json').read_text())
    original=Path(launch['repo']).resolve()
    if launch['study'] not in ('s1_convergence','s1_a2_convergence'):raise ValueError('Wrong training run')
    # The original job and any previous recovery must be inactive before reusing paths.
    active=set(slurm('squeue','-h','-u',getpass.getuser(),'-o','%i').split())
    for f in [run/'job-id.txt',run/'recovery-job-id.txt']:
        if f.exists():
            job=f.read_text().strip().split(';')[0]
            if job in active:raise RuntimeError(f'Run still active: {job}')
    resources=json.loads((original/'configs/slurm_hpc.json').read_text())
    recovery=run/'recovery';recovery.mkdir(exist_ok=True)
    # This mount is independent from /datastore; only error records/emergency checkpoints go here.
    error_dir=Path.home()/'.qlrg_recovery'/run.name
    error_dir.mkdir(parents=True,exist_ok=True)
    stamp=recovery/'launch.json'
    stamp.write_text(json.dumps({'original_repo':str(original),'run':str(run),
                     'recovery_repo':str(REPO),'error_dir':str(error_dir),
                     'policy':'No training-setting or checkpoint-binding change'},indent=2)+'\n')
    python=Path(launch['env'])/'bin/python'
    commands=[
        'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"',
        'export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1',
        'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1',
        'export PYTHONPATH='+shlex.quote(str(original)),
        'exec '+shlex.join([str(python),'-B',str(REPO/'scripts/resume_recovery.py'),
                           '--repo',str(original),'--run',str(run),'--error-dir',str(error_dir)]),
    ]
    script=write_script(recovery,'resume',commands)
    command=['sbatch','--parsable',f'--chdir={original}',f'--job-name=qlrg-recover-{launch["study"]}',
             f"--account={resources['account']}",f"--partition={resources['partition']}",
             f"--qos={resources['qos']}",f"--gres={resources['gres']}",
             '--cpus-per-task=8','--mem=48G',f'--time={args.time}',
             f'--output={run}/logs/qlrg-recover-%j.out',f'--error={run}/logs/qlrg-recover-%j.err',str(script)]
    print(shlex.join(command),flush=True)
    job=slurm(*command)
    (run/'recovery-job-id.txt').write_text(job+'\n')
    print('Recovery job:',job,'error journal:',error_dir,flush=True)


if __name__=='__main__':main()
