from subprocess import PIPE, Popen
from pathlib import Path

from flask_socketio import emit

from .. import socketio

python_executable = Path('.venv', 'Scripts', 'python.exe')
test_script = Path('test_nn.py')
tester = None


@socketio.on('connect', namespace='/test-nn-ws')
def test_connect():
    emit('message', 'Connected successfully\n\n')


@socketio.on('start', namespace='/test-nn-ws')
def start():
    global tester
    if tester is not None:
        emit('message', 'Already started, wait for a result\n\n')
        return
    try:
        tester = Popen([python_executable, '-u', test_script], stdout=PIPE, stderr=PIPE)
        for output in tester.stdout:
            emit('message', output.decode('utf-8'))
        for output in tester.stderr:
            emit('message', output.decode('utf-8'))
        emit('message', '\n')
    finally:
        tester = None
