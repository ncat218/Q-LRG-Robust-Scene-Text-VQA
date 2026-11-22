"""Submit one isolated, resumable old/new full/crop/fusion evaluation job."""

import argparse
import getpass
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

from run_all import REPO, WITH_SLURM, write_script


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--from-matrix', type=Path)
    group.add_argument('--resume', type=Path)
    parser.add_argument('--a0-training', type=Path)
    parser.add_argument('--a2-training', type=Path)
    parser.add_argument('--dependency')
    parser.add_argument('--time', default='2-00:00:00')
    parser.add_argument('--preflight', action='store_true', help='Verify all completed sources on CPU, no submission')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if os.environ.get('SLURM_JOB_ID'):
        parser.error('Submit from the login node')
    if args.resume:
        output = args.resume.resolve()
        launch = json.loads((output / 'launch.json').read_text())
        if launch['study'] != 'checkpoint_matrix' or launch['repo'] != str(REPO):
            raise ValueError('Resume the same archived worker snapshot')
    else:
        if not args.a0_training or not args.a2_training:
            parser.error('Both training run directories are required')
        source = args.from_matrix.resolve()
        if not (source / 'COMPLETE.json').is_file():
            raise ValueError('Original comparison matrix is incomplete')
        settings = json.loads((source / 'launch.json').read_text())
        for root, stage in [(args.a0_training, 's1_convergence'), (args.a2_training, 's1_a2_convergence')]:
            info = json.loads((root / 'launch.json').read_text())
            if info['study'] != stage or info.get('smoke_images'):
                raise ValueError('Expected full-scale convergence study')
        if args.dry_run:
            output = REPO / 'runs/dry-run-checkpoint-matrix'
        else:
            (REPO / 'runs').mkdir(exist_ok=True)
            output = Path(tempfile.mkdtemp(prefix='checkpoint-matrix-', dir=REPO / 'runs'))
        launch = {**settings, 'repo': str(REPO), 'study': 'checkpoint_matrix', 'split': 'official_val',
                  'source_matrix': str(source), 'a0_training': str(args.a0_training.resolve()),
                  'a2_training': str(args.a2_training.resolve()), 'smoke_images': None,
                  'flops_images': 3, 'conditions': ['CLEAN', 'LR4'], 'dependency_requested': args.dependency}
        if not args.dry_run:
            (output / 'launch.json').write_text(json.dumps(launch, indent=2) + '\n', encoding='utf-8')
    if args.preflight:
        if args.dry_run:
            parser.error('Preflight requires an actual launch directory')
        from scripts.checkpoint_matrix import preflight
        result = preflight(output, allow_pending=True)
        (output / 'preflight-login.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(result, indent=2), flush=True)
        print('Preflight only; no job submitted. Run:', output, flush=True)
        return
    python = Path(launch['env']) / 'bin/python'
    required = [python, Path(launch['paths']), Path(launch['data_dir']) / 'official_val.json']
    if not args.dry_run:
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(path)
    # Also reject a duplicate active job for the same source studies in this snapshot.
    active = subprocess.check_output(['bash', '-c', WITH_SLURM, 'qlrg-slurm', 'squeue',
                                     '-h', '-u', getpass.getuser(), '-o', '%i'], text=True).split()
    for directory in (REPO / 'runs').glob('checkpoint-matrix-*'):
        if directory == output:
            candidates = [directory]
        else:
            candidates = [directory] if (directory / 'launch.json').exists() else []
        for candidate in candidates:
            saved = json.loads((candidate / 'launch.json').read_text())
            jobfile = candidate / 'job-id.txt'
            if (saved.get('a0_training'), saved.get('a2_training')) == (launch['a0_training'], launch['a2_training']):
                if jobfile.exists() and jobfile.read_text().strip().split(';')[0] in active:
                    raise RuntimeError('A matching evaluation is already active: ' + str(candidate))
    resources = json.loads((REPO / 'configs/slurm_hpc.json').read_text())
    commands = ['export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"',
                'export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1',
                'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1',
                'export QLRG_PATHS_CONFIG=' + shlex.quote(launch['paths']),
                'export PYTHONPATH=' + shlex.quote(str(REPO)), 'nvidia-smi',
                'exec ' + shlex.join([str(python), '-B', '-m', 'scripts.checkpoint_matrix', str(output)])]
    if not args.dry_run:
        (output / 'logs').mkdir(exist_ok=True)
        script = write_script(output, 'checkpoint-matrix', commands)
    else:
        script = output / 'checkpoint-matrix.sbatch'
    command = ['sbatch', '--parsable', f'--chdir={REPO}', '--job-name=qlrg-checkpoint-matrix',
               f"--account={resources['account']}", f"--partition={resources['partition']}",
               f"--qos={resources['qos']}", f"--gres={resources['gres']}",
               '--cpus-per-task=8', '--mem=48G', f'--time={args.time}',
               f'--output={output}/logs/%x-%j.out', f'--error={output}/logs/%x-%j.err']
    if args.dependency:
        command.append('--dependency=afterok:' + args.dependency)
    command.append(str(script))
    print('Run:', output, flush=True)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    job = subprocess.check_output(['bash', '-c', WITH_SLURM, 'qlrg-slurm', *command], text=True).strip()
    (output / 'job-id.txt').write_text(job + '\n')
    print('Job:', job, flush=True)


if __name__ == '__main__':
    main()
