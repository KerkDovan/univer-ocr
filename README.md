# Univer OCR

This project is the course work at [ITMO University](http://www.ifmo.ru). Aim of the project is to develop [OCR software](https://en.wikipedia.org/wiki/Optical_character_recognition) using parallel programming and self-written [Deep Learning](https://en.wikipedia.org/wiki/Deep_learning) Framework.

## Installation

1. Install Python 3.7 from [official site](https://www.python.org/)

2. Create and enter [virtual environment](https://virtualenv.pypa.io/en/latest/):

    ```cmd
    virtualenv .venv --python=python3.7
    .venv/Scripts/activate.bat
    ```

3. Download and install [CUDA Toolkits](https://developer.nvidia.com/cuda-downloads). If you need to install CUDA Toolkits version different from `10.0`, also install corresponding version of [CuPy](https://docs-cupy.chainer.org/en/stable/install.html#install-cupy) instead of specified one in file `requirements/base.txt`.

4. Set environmental variable `CUDA_HOME` to directory of installed CUDA Toolkits:

    ```cmd
    set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0
    ```

    or via OS system settings.

5. Install requirements:

    ```cmd
    pip install -r requirements/base.txt
    ```

## Running

Simply run this command while being in created virtual environment:

```cmd
python start_web_app.py
```
