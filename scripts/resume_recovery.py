"""Resume a pinned worker with recorded I/O recovery, without changing its binding."""

import argparse
import errno
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone


RETRY_ERRNOS={getattr(errno,name) for name in ('EIO','EAGAIN','ENOSPC','ETIMEDOUT','ESTALE','EDQUOT') if hasattr(errno,name)}


def safe_write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(dir=path.parent,prefix='.recovery-')
    try:
        with os.fdopen(fd,'wb') as handle:
            handle.write(data);handle.flush();os.fsync(handle.fileno())
        os.replace(name,path)
    finally:
        try:os.unlink(name)
        except FileNotFoundError:pass


def record(directory,name,value):
    safe_write(directory/name,(json.dumps(value,ensure_ascii=False,default=str)+'\n').encode())


def io_wrapper(original,directory,attempts=3):
    def recovered(path,data):
        for index in range(attempts):
            try:
                result=original(path,data)
                if index:print(f'IO_RETRY_RECOVERED target={path} attempts={index+1}',flush=True)
                return result
            except OSError as error:
                event={'target':str(path),'errno':error.errno,'error':str(error),'attempt':index+1,
                       'time_utc':datetime.now(timezone.utc).isoformat()}
                record(directory,'last-io-error.json',event)
                print('IO_ERROR '+json.dumps(event),file=sys.stderr,flush=True)
                if error.errno not in RETRY_ERRNOS or index==attempts-1:
                    if Path(path).suffix=='.pt':
                        key=hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()[:16]
                        backup=directory/('emergency-'+key+'.pt')
                        safe_write(backup,data)
                        record(directory,'emergency-checkpoint.json',{**event,'backup':str(backup),
                               'sha256':hashlib.sha256(data).hexdigest()})
                        print(f'EMERGENCY_CHECKPOINT {backup}',file=sys.stderr,flush=True)
                    raise
                time.sleep(index+1)
    return recovered


def install_io_recovery(directory):
    from qlrg import common
    original=common.atomic_write
    replacement=io_wrapper(original,directory)
    # Only replace aliases of this exact atomic writer; optimization and model code stay pinned.
    for module in list(sys.modules.values()):
        namespace=getattr(module,'__dict__',{})
        if namespace.get('atomic_write') is original:namespace['atomic_write']=replacement


def failure_record(error):
    frames=[];trace=error.__traceback__
    while trace is not None:
        frames.append({'file':trace.tb_frame.f_code.co_filename,'line':trace.tb_lineno,
                       'function':trace.tb_frame.f_code.co_name})
        trace=trace.tb_next
    # Do not depend on reading source files from a potentially unavailable NFS mount.
    return {'error_type':type(error).__name__,'message':str(error),'frames':frames,
            'time_utc':datetime.now(timezone.utc).isoformat(),'job_id':os.environ.get('SLURM_JOB_ID'),
            'pid':os.getpid()}


def execute(args):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Recovery worker requires Slurm')
    sys.path.insert(0,str(args.repo.resolve()))
    os.chdir(args.repo)
    launch=json.loads((args.run/'launch.json').read_text())
    if launch['study'] not in ('s1_convergence','s1_a2_convergence'):raise ValueError('Wrong study')
    module=importlib.import_module('scripts.'+launch['study'])
    if not Path(module.__file__).resolve().is_relative_to(args.repo):
        raise RuntimeError('Recovery imported a worker outside the original pinned snapshot')
    from filelock import FileLock
    from qlrg.common import sha256,write_json
    install_io_recovery(args.error_dir)
    record(args.error_dir,f'worker-{os.getpid()}.json',{
        'original_repo':str(args.repo),'run':str(args.run),'view':args.view,
        'original_worker_sha256':sha256(module.__file__),
        'recovery_runner_sha256':sha256(__file__),
        'policy':'Same model/data/loss/optimizer/schedule/RNG. Atomic I/O retries and independent error journal only.'})
    if args.view:
        print(f'RECOVERY_RESUME {launch["study"]}/{args.view} from existing latest.pt',flush=True)
        module.run_view(args.run,args.view)
        return
    views=('full','crop') if launch['train_view']=='both' else (launch['train_view'],)
    with FileLock(str(args.run/'.recovery.lock'),timeout=0):
        for view in views:
            command=[sys.executable,'-B',str(Path(__file__).resolve()),'--repo',str(args.repo),
                     '--run',str(args.run),'--error-dir',str(args.error_dir),'--view',view]
            subprocess.run(command,check=True)
        write_json(args.run/'COMPLETE.json',{'stage':launch['study'],'views':list(views),
                   'summaries':{view:sha256(args.run/view/'summary.json') for view in views}})
        if launch['study']=='s1_a2_convergence':write_json(args.run/'progress.json',{'stage':'complete'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--error-dir',type=Path,required=True)
    parser.add_argument('--view',choices=('full','crop'))
    args=parser.parse_args()
    args.repo=args.repo.resolve();args.run=args.run.resolve()
    args.error_dir=args.error_dir.resolve()
    try:execute(args)
    except BaseException as error:
        if isinstance(error,SystemExit) and error.code in (None,0):raise
        details=failure_record(error)
        try:
            path=args.error_dir/f'failure-{os.getpid()}.json'
            record(args.error_dir,path.name,details)
            print(f'ERROR_JOURNAL {path}',file=sys.stderr,flush=True)
        except OSError as journal_error:print(f'ERROR_JOURNAL_FAILED {journal_error}',file=sys.stderr,flush=True)
        print(json.dumps(details,ensure_ascii=False),file=sys.stderr,flush=True)
        raise SystemExit(1)


if __name__=='__main__':main()
