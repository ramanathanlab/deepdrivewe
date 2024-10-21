#!/bin/bash
#SBATCH -J idev1
#SBATCH -o idev1.o%j
#SBATCH -N 3
#SBATCH --ntasks-per-node=1
#SBATCH -p gh
#SBATCH -t 00:60:00
#SBATCH -A ASC24062
#------------------------------------------------------
# Source the bashrc to add conda
source ~/.bashrc

# Load the required modules
ml gcc/14.2.0 cuda/12.5 hdf5
conda activate deepdrivewe

# Change to working directory
cd /scratch/08288/abrace/projects/ddwe/src/deepdrivewe

# Get the config file for this example
CONFIG_FILE=/scratch/08288/abrace/projects/ddwe/src/deepdrivewe/examples/openmm_ntl9_ddwe_vista/config.yaml

# Run the example
python -m deepdrivewe.examples.openmm_ntl9_ddwe.main --config $CONFIG_FILE
