# Univer OCR

This project is the course work at [ITMO University](http://www.ifmo.ru). Aim of the project is to develop [OCR software](https://en.wikipedia.org/wiki/Optical_character_recognition) using parallel programming and self-written [Deep Learning](https://en.wikipedia.org/wiki/Deep_learning) Framework.

Project was developed and tested on Windows 10, however, it might work on Linux too.

If you have GPU, you may unleash it's power to significantly increase speed of neural net in this project. If you don't have one, or don't want to use it, skip steps `6` and `7` of *Installation*. Although this has not been tested either.

## Installation

1. Install `Python 3.7` from [official site](https://www.python.org/).

2. Install [virtualenv](https://virtualenv.pypa.io/en/latest/) via `pip`. This command must be run as administrator in Windows or using sudo in Linux.

    ```bash
    pip install virtualenv
    ```

3. Enter root folder of the project:

    ```bash
    cd /PATH/TO/PROJECT
    ```

4. Create virtual environment with `Python 3.7`:

    ```bash
    virtualenv .venv --python=python3.7
    ```

5. And activate it:

    In Windows:

    ```bat
    .venv\Scripts\activate.bat
    ```

    In Linux:

    ```bash
    source .venv/Scripts/activate
    ```

6. Download and install [CUDA Toolkits](https://developer.nvidia.com/cuda-downloads). Refer to [this table](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) to find out which version you need. If you need to install `CUDA Toolkits` version different from `10.0`, also install corresponding version of [CuPy](https://docs-cupy.chainer.org/en/stable/install.html#install-cupy) instead of specified one in the file `requirements/base.txt`.

7. Set environmental variable `CUDA_HOME` to directory of installed `CUDA Toolkits`:

    In Windows:

    ```bat
    set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0
    ```

    In Linux:

    ```bash
    export CUDA_HOME=/PATH/TO/CUDA/TOOLKITS
    ```

    Also you may do this via your OS system settings or by adding latter command into your `.bashrc` file. In this case your terminal must be restarted to be able to use the environmental variable.

8. While being with activated virtualenv, install requirements. If needed (in step `6`), change it. No administrator or sudo is needed here.

    ```bat
    pip install -r requirements/base.txt
    ```

## Usage

To run any scripts in this project you should either activate `virtualenv` (see step `5` of *Installation*) or use `Python` executable from it (`.venv\Scripts\python.exe` or `.venv/Scripts/python`).

1. To start web application run this command:

    ```bat
    python start_web_app.py
    ```

    Now you can access it in your browser at [http://127.0.0.1:80](http://127.0.0.1:80). If you need to start with different port, change it in file `start_web_app.py`.

    Though, it is not recommended to run training using web interface, because it noticeably slows down the process. It's better to train via command line. To do this see step `3` of *Usage*.

2. To generate train and validation data run this command:

    ```bat
    python run generate_data
    ```

    It will create directory `generated_files/data`.

3. To train the model run this command:

    ```bat
    python run train [use_gpu [console_mode [show_progress_bar [save_train_progress]]]]
    ```

    As you see, this command has a bunch of arguments:

    - `use_gpu`: may be `True` or `False`. If `True`, makes script to use your GPU, otherwise runs on CPU. Default is `False`.

    - `console_mode`: may be `True` or `False`. If `False`, makes your script to connect to web application (from step `1` of *Usage*), otherwise prints all output in the console. Default is `True`.

    - `show_progress_bar`: may be `True` or `False`. If `True`, displays progress bar for each epoch. Handy when running in console mode, but dramatically increases number of lines in log file, if you redirect output from console to file. Default is `False`.

    - `save_train_progress`: may be `True` or `False`. If `True`, saves **all** input and output pictures of **each** iteration of **each** epoch while training. This can help you visualize training process, but be very careful because it is extremely memory-consuming operation and may fill up your hard drive in no time. Saved pictures are located at `generated_files/train_progress`. Default is `False`.

    If you want to train the model from scratch, at first delete the file `web_app/components/my_model/model_weights.json`. It will initialize the model with random weights.
