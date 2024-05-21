# westpa_colmena
Implementation of WESTPA using Colmena

## Installation

To install the package, run the following command:
```bash
git clone git@github.com:braceal/westpa_colmena.git
cd westpa_colmena
pip install -e .
```

Full installation including dependencies:
```bash
git clone git@github.com:braceal/westpa_colmena.git
cd westpa_colmena
conda create -n westpa_colmena python=3.10 -y
conda install omnia::ambertools -y
pip install mdtraj==1.9.9
pip install -e .
```


## Contributing

For development, it is recommended to use a virtual environment. The following
commands will create a virtual environment, install the package in editable
mode, and install the pre-commit hooks.
```bash
python3.10 -m venv venv
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
