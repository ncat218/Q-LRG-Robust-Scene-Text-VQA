"""Submit one pinned, resumable full-data semantic-restoration study."""

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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--from-matrix', type=Path)
    source.add_argument('--resume', type=Path)
    parser.add_argument('--config', type=Path, default=REPO / 'configs/semantic_restoration.json')
    parser.add_argument('--time', default='3-00:00:00')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--preflight', action='store_true', help='CPU source checks only, no Slurm submission')
    args = parser.parse_args()
    if os.environ.get('SLURM_JOB_ID'):
        parser.error('Launch from the login node')
    if args.resume:
        output = args.resume.resolve()
        launch = json.loads((output / 'launch.json').read_text())
        if launch['study'] != 'semantic_restoration' or launch['repo'] != str(REPO):
            raise ValueError('Resume the same pinned snapshot and study')
    else:
        old = args.from_matrix.resolve()
        if not (old / 'COMPLETE.json').is_file():
            raise ValueError('Supply the completed old-checkpoint matrix')
        settings = json.loads((old / 'launch.json').read_text())
        config = json.loads(args.config.read_text(encoding='utf-8'))
        if config['methods'] != ['continued', 'answer_kd', 'answer_kd_patch'] or config['extra_epochs'] != 2:
            raise ValueError('Use the registered matched three-arm study')
        if args.dry_run:
            output = REPO / 'runs/dry-run-semantic-restoration'
        else:
            (REPO / 'runs').mkdir(exist_ok=True)
            output = Path(tempfile.mkdtemp(prefix='semantic-restoration-', dir=REPO / 'runs'))
        launch = {**settings, 'repo': str(REPO), 'study': 'semantic_restoration', 'source_matrix': str(old),
                  'split': 'official_val', 'semantic_config': config, 'smoke_images': None}
        if not args.dry_run:
            (output / 'launch.json').write_text(json.dumps(launch, indent=2) + '\n', encoding='utf-8')
    if args.preflight:
        if args.dry_run:
            parser.error('Preflight requires an actual run directory')
        from qlrg.common import dataset_paths, object_hash, read_config, read_json, sha256
        from qlrg.data import verify_pillow
        from qlrg.experiment import verified_complete
        from qlrg.s1_manifests import read_s1_manifests
        from qlrg.s1_model import VisualResidual
        import torch
        verify_pillow()
        old = verified_complete(Path(launch['s1_run']) / 's1', 's1')
        paths = dataset_paths(read_config(launch['paths']))
        manifests, hashes = read_s1_manifests(Path(launch['data_dir']), paths['textvqa'])
        if hashes != old['dataset_hashes'] or read_config(REPO / 'configs/model_qwen.yaml') != old['shared']['model']:
            raise ValueError('Pinned model/datasets changed')
        checked = {}
        for arm in ('S1b', 'A2'):
            path = Path(launch['s1_run']) / 's1/variants' / arm / 'checkpoint.pt'
            saved = torch.load(path, map_location='cpu', weights_only=True)
            if saved['binding'] != object_hash(old) or saved['arm'] != arm or saved['examples'] != 62258:
                raise ValueError('Expected completed matched 11305 checkpoint')
            if any(not torch.isfinite(v).all() for v in saved['adapter'].values()):
                raise ValueError('Nonfinite source checkpoint')
            if arm == 'S1b':
                cfg = old['config']
                module = VisualResidual(saved['adapter']['output.weight'].shape[0], cfg['hidden'], cfg['spatial_blocks'])
                module.load_state_dict(saved['adapter'], strict=True)
            checked[arm] = sha256(path)
        result = {'ready': True, 'source_checkpoints': checked, 'datasets': {
            k: {'images': len(v['images']), 'questions': sum(len(im['questions']) for im in v['images'])}
            for k, v in manifests.items()}, 'gpu_forward_verified': False}
        (output / 'preflight.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(result, indent=2), flush=True)
        print('Preflight only; no submission. Run:', output, flush=True)
        return
    active = subprocess.check_output(['bash', '-c', WITH_SLURM, 'qlrg-slurm', 'squeue', '-h',
                                     '-u', getpass.getuser(), '-o', '%i'], text=True).split()
    for root in (REPO / 'runs').glob('semantic-restoration-*'):
        if not (root / 'launch.json').exists():
            continue
        job = root / 'job-id.txt'
        saved = json.loads((root / 'launch.json').read_text())
        if saved['semantic_config'] == launch['semantic_config'] and job.exists() and job.read_text().strip().split(';')[0] in active:
            raise RuntimeError('Identical study already active: ' + str(root))
    resources = json.loads((REPO / 'configs/slurm_hpc.json').read_text())
    python = Path(launch['env']) / 'bin/python'
    if not args.dry_run:
        for path in (python, Path(launch['paths']), Path(launch['data_dir']) / 'official_val.json',
                     Path(launch['s1_run']) / 's1/variants/S1b/checkpoint.pt'):
            if not path.is_file():
                raise FileNotFoundError(path)
    commands = ['export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"',
                'export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1',
                'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1',
                'export PYTHONPATH=' + shlex.quote(str(REPO)),
                'export QLRG_PATHS_CONFIG=' + shlex.quote(launch['paths']), 'nvidia-smi',
                'exec ' + shlex.join([str(python), '-B', '-m', 'scripts.semantic_restoration', str(output)])]
    if not args.dry_run:
        (output / 'logs').mkdir(exist_ok=True)
        script = write_script(output, 'semantic', commands)
    else:
        script = output / 'semantic.sbatch'
    command = ['sbatch', '--parsable', f'--chdir={REPO}', '--job-name=qlrg-semantic-restoration',
               f"--account={resources['account']}", f"--partition={resources['partition']}",
               f"--qos={resources['qos']}", f"--gres={resources['gres']}", '--cpus-per-task=8', '--mem=48G',
               f'--time={args.time}', '--signal=TERM@120',
               f'--output={output}/logs/%x-%j.out', f'--error={output}/logs/%x-%j.err', str(script)]
    print('Run:', output, flush=True)
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        job = subprocess.check_output(['bash', '-c', WITH_SLURM, 'qlrg-slurm', *command], text=True).strip()
        (output / 'job-id.txt').write_text(job + '\n')
        print('Job:', job, flush=True)


if __name__ == '__main__':
    main()
