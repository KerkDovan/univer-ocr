from subprocess import PIPE, Popen
from pathlib import Path

from flask_socketio import emit

from .. import socketio

python_executable = Path('.venv', 'Scripts', 'python.exe')
tester_filepath = Path('test_nn.py')
test_scripts = {
    'test_gradients': 'Test gradients',
}
tester = None


@socketio.on('connect', namespace='/test-nn-ws')
def test_connect():
    emit('message', 'Connected successfully\n\n')


@socketio.on('start', namespace='/test-nn-ws')
def start(message):
    global tester
    if tester is not None:
        emit('message', 'Already started, wait for a result\n\n')
        return
    try:
        tester = Popen([python_executable, '-u', tester_filepath, message['test_name']],
                       stdout=PIPE, stderr=PIPE)
        for output in tester.stdout:
            emit('message', output.decode('utf-8'))
        for output in tester.stderr:
            emit('message', output.decode('utf-8'))
        emit('message', '\n')
    finally:
        tester = None


@socketio.on('stop', namespace='/test-nn-ws')
def stop():
    global tester
    if tester is None:
        return
    try:
        tester.terminate()
        emit('message', 'Stopped the process\n')
    finally:
        tester = None
