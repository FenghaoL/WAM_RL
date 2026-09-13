"""Sequential three-GPU experiment queue. Stops at the first failed experiment."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RUN = 'runs/ovcr_adapter_padapter_v2'
HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--names', nargs='+', default=['p1_v32_balanced','p1_q32_balanced','p1_t4_balanced'])
    p.add_argument('--smoke', action='store_true', help='Two steps per selected variant; writes separate smoke outputs')
    p.add_argument('--run-root', default=DEFAULT_RUN, help='Directory holding configs/, logs/ and outputs of one task')
    args = p.parse_args()
    RUN = ROOT / args.run_root
    if not (RUN / 'configs').is_dir():
        raise FileNotFoundError(f'No generated configurations under {RUN}')
    RUN.mkdir(parents=True, exist_ok=True)
    lock = (RUN/'suite.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (RUN/'suite.pid').write_text(str(os.getpid())+'\n')
    (RUN/'logs').mkdir(exist_ok=True)
    child = None
    def stop(signum, frame):
        # Only signal here. Calling child.wait() inside the handler deadlocks: the
        # interrupted main-thread wait() still holds Popen's internal lock. The
        # finally block below reaps the child once that wait() has unwound.
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    status = dict(pid=os.getpid(), started=time.time(), smoke=args.smoke, experiments=[])
    def save():
        temporary = RUN/'suite_status.tmp'
        temporary.write_text(json.dumps(status, indent=2)+'\n')
        temporary.replace(RUN/'suite_status.json')
    try:
        for name in args.names:
            if Path(name).name != name or not (RUN/'configs'/f'{name}.yaml').is_file():
                raise ValueError(f'Unknown experiment: {name}')
            tag = name + ('_smoke' if args.smoke else '')
            output = RUN/tag/'model'
            if output.exists():
                raise FileExistsError(f'Refusing to overwrite {output}; choose a new run name/config')
            record = dict(name=name, tag=tag, state='running', started=time.time(), log=str(RUN/'logs'/f'{tag}.log'))
            status['experiments'].append(record)
            save()
            command = [str(HERE/'launch.sh'), '--config', str(RUN/'configs'/f'{name}.yaml')]
            if args.smoke: command += ['--smoke-steps','2']
            with Path(record['log']).open('x') as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                record['launcher_pid'] = child.pid
                save()
                code = child.wait()
            record.update(state='completed' if code == 0 else 'failed', exit_code=code, ended=time.time())
            save()
            if code: raise SystemExit(code)
        status['state'] = 'completed'
    finally:
        if child is not None and child.poll() is None:
            try:
                child.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        for record in status['experiments']:
            if record.get('state') == 'running':
                record.update(state='stopped', exit_code=child.returncode if child else None, ended=time.time())
        status.setdefault('state', 'stopped')
        save()
        (RUN/'suite.pid').unlink(missing_ok=True)


if __name__ == '__main__': main()
