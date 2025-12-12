import os
import subprocess
import sys

if __name__ == '__main__': 
    cmd = [
        sys.executable, '-m', 'chainlit', 'run', 'chainlit_app.py',
        '--debug'
    ]
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd)