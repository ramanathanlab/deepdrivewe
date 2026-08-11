# deepdrivewe
Implementation of [WESTPA](https://westpa.github.io/westpa/index.html) using [Colmena](https://github.com/exalearn/colmena/tree/master).

## Installation

To install the package, run the following command:
```bash
git clone git@github.com:braceal/deepdrivewe.git
cd deepdrivewe
pip install -U pip setuptools wheel
pip install -e .
```

Full installation including dependencies:
```bash
git clone git@github.com:braceal/deepdrivewe.git
cd deepdrivewe
conda create -n deepdrivewe python=3.10 -y
conda install omnia::ambertools -y
conda install conda-forge::openmm==7.7 -y
conda install anaconda::redis -y
pip install -U pip setuptools wheel
pip install -e .
```

To use deep learning models, install the correct version of [PyTorch](https://pytorch.org/get-started/locally/).

### Installation on VISTA

To install the package on VISTA, run the following commands:
```bash
ml gcc/14.2.0 cuda/12.5 hdf5

conda create -n deepdrivewe python=3.12 -y
conda activate deepdrivewe
conda install conda-forge::openmm -y
conda install anaconda::redis -y
pip install torch --index-url https://download.pytorch.org/whl/cu124

git clone git@github.com:braceal/deepdrivewe.git
cd deepdrivewe
pip install -U pip setuptools wheel
pip install -e .
```
To run an example on VISTA, update the absolute paths in the submit script
and the YAML config file, and then run the following command:
```bash
sbatch examples/openmm_ntl9_ddwe_vista/submit.sh
```

### Installation on Polaris

To install the package on Polaris@ALCF, run the following commands:
```bash
module use /soft/modulefiles; module load conda
```

Follow the full installation instructions above, and install torch via:
```bash
pip install torch
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
cat runs/*/result/inference.json | grep '"success": false'
cat runs/*/result/simulation.json | grep '"success": false'
```

To check the number of iterations completed:
```bash
h5ls -d runs/naive_resampler_test_v2/west.h5/iterations
```

To watch the progress of the simulation:
```bash
tail -f runs/*/simulation/*/*/*.log
```

To pretty print potential errors:
```bash
deepdrivewe print-errors --run_dir runs/ntl9-v1
```

Run the following, for more information:
```bash
deepdrivewe --help
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

### Running with OpenMM
To run the example, run the following command:
```bash
OPENMM_CPU_THREADS=1 nohup python -m deepdrivewe.examples.openmm_ntl9_hk.main --config examples/openmm_ntl9_hk/config.yaml &> nohup.log &
```

Note that we set `OPENMM_CPU_THREADS=1` to restrict each OpenMM simulation to a single thread. This is necessary to prevent
the simulations from using all available CPU resources. You can also run the simulations on a GPU by adjusting the Parsl configuration.

### Running with streaming

For a full example, see `examples/openmm_ntl9_ddwe_stream`.

To run with streaming, add the following to the config:
```yaml
stream_config:
  # A redis server is used as the stream message broker
  redis_host: localhost
  redis_port: 6379
  # The Store used for stream items is configurable
  store_config:
    name: stream-store
    # Use the same redis server for object storage
    connector:
      kind: redis
      options:
        hostname: localhost
        port: 6379
    # FileConnector example
    # connector:
    #   kind: file
    #   options:
    #     store_dir: /tmp/proxystore-cache
```

Then start a redis server in the background:
```bash
redis-server --port 6379 --save "" --appendonly no --protected-mode no &> redis.log &
```

The redis server can later be killed using the job number:
```bash
jobs
kill %<num>
```

To check resource utilization of the redis server, run the following command:
```bash
watch "ps -p \$(pgrep -x redis-server | head -n1) -o pid,comm,%mem,rss,vsz"
```
**Note:** If there are multiple redis servers running, this command will choose the first one.

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
