#!/bin/bash
#PBS -N proc_cesm_dple
#PBS -q regular
#PBS -A NCGD0011
#PBS -l select=1:ncpus=1:mpiprocs=1:mem=109GB
#PBS -l walltime=06:00:00
#PBS -j oe

#-------------------------------------------------------------------------------
#-- end user inputs
#-------------------------------------------------------------------------------
module purge
unset LD_LIBRARY_PATH
source activate py3_geyser

./proc_cesm_dple.py
