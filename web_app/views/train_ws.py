from pathlib import Path
from subprocess import PIPE, Popen

from flask_socketio import emit

from .. import socketio

python_executable = Path('.venv', 'Scripts', 'python.exe')
trainer_filepath = Path('train.py')
trainer = None


@socketio.on('connect', namespace='/train-ws')
def test_connect():
    emit('message', 'Connected successfully\n\n')


@socketio.on('start', namespace='/train-ws')
def start(args):
    global trainer
    if trainer is not None:
        emit('message', 'Already started, wait for a result\n\n')
        return
    try:
        trainer = Popen([str(python_executable), '-u', str(trainer_filepath),
                         str(args['use_gpu']), str(False)],
                        stdout=PIPE, stderr=PIPE)
        for output in trainer.stdout:
            emit('message', output.decode('utf-8'))
        for output in trainer.stderr:
            emit('message', output.decode('utf-8'))
        emit('message', '\n')
    finally:
        trainer = None


@socketio.on('message', namespace='/train-ws')
def trainer_message(message):
    emit('message', message, broadcast=True)


@socketio.on('info', namespace='/train-ws')
def trainer_info(data):
    emit('info', data, broadcast=True)


@socketio.on('progress_tracker', namespace='/train-ws')
def trainer_progress_tracker(status):
    emit('progress_tracker', status, broadcast=True)


@socketio.on('stop', namespace='/train-ws')
def stop():
    global trainer
    if trainer is None:
        return
    try:
        trainer.terminate()
        emit('message', 'Stopped the process\n')
    finally:
        trainer = None
