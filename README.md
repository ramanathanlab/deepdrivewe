# deepdrivewe
Implementation of [WESTPA](https://westpa.github.io/westpa/index.html) using [Colmena](https://github.com/exalearn/colmena/tree/master).

## Installation

To install the package, run the following command:
```bash
git clone git@github.com:braceal/deepdrivewe.git
cd deepdrivewe
pip install -e .

# TODO: remove this after next colmena release
pip uninstall colmena
pip install git+https://github.com/braceal/colmena.git --no-cache-dir
```

Full installation including dependencies:
```bash
git clone git@github.com:braceal/deepdrivewe.git
cd deepdrivewe
conda create -n deepdrivewe python=3.10 -y
conda install omnia::ambertools -y
conda install conda-forge::openmm==7.7 -y
pip install -e .
```

To use deep learning models, install the correct version of [PyTorch](https://pytorch.org/get-started/locally/)
for your system and drivers. To use `mdlearn`, you may need an earlier version of PyTorch:
```bash
pip install torch==1.12
```

## Usage
To run the example, run the following command:
```bash
python -m deepdrivewe.examples.amber_hk.main --config examples/amber_nacl_hk/config.yaml
```

To kill all the workers, run the following command:
```bash
ps -e | grep -E 'sander|python|process_worker|parsl' | awk '{print $1}' | xargs kill
```

To check if any errors occurred in simulations or inference:
```bash
cat runs/naive_resampler_test_v2/result/inference.json | grep '"success": false'
cat runs/naive_resampler_test_v2/result/simulation.json | grep '"success": false'
```

To check the number of iterations completed:
```bash
h5ls -d runs/naive_resampler_test_v2/west.h5/iterations
```

### Running with SynD
To use the SynD simulation engine, install the following dependencies:
```bash
pip install git+https://github.com/jeremyleung521/SynD.git@rng-fix
```

To generate the basis state .npy files from a .txt file, run the following command:
```bash
python -m deepdrivewe.simulation.synd --basis-states examples/synd_ntl9/bstates.txt --output-dir examples/synd_ntl9/bstates
```

To run the example, run the following command:
```bash
nohup python -m deepdrivewe.examples.synd_ntl9.main --config examples/synd_ntl9/config.yaml &> nohup.log &
```

## Contributing

For development, it is recommended to use a virtual environment. The following
commands will create a virtual environment, install the package in editable
mode, and install the pre-commit hooks.
```bash
python -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
pip install -e '.[dev,docs]'
pre-commit install
```
To test the code, run the following command:
```bash
pre-commit run --all-files
tox -e py310
```
