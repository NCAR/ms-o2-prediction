#!/bin/bash
#SBATCH -J proc_ts_files
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -p dav
#SBATCH -A NCGD0011
#SBATCH -t 24:00:00
#SBATCH --mem 200GB
#SBATCH -C geyser
#SBATCH -e logs/proc_ts_files.%J.out
#SBATCH -o logs/proc_ts_files.%J.out

#-------------------------------------------------------------------------------
#-- end user inputs
#-------------------------------------------------------------------------------

CONDA_ENV=py3_geyser
CONDA_PATH=/glade/work/mclong/miniconda3/bin

if [ -z $MODULEPATH_ROOT ]; then
  unset MODULEPATH_ROOT
else
  echo "NO MODULEPATH_ROOT TO RESET"
fi
if [ -z $MODULEPATH ]; then
  unset MODULEPATH
else
  echo "NO MODULEPATH TO RESET"
fi
if [ -z $LMOD_SYSTEM_DEFAULT_MODULES ]; then
  unset LMOD_SYSTEM_DEFAULT_MODULES
else
  echo "NO LMOD_SYSTEM_DEFAULT_MODULES TO RESET"
fi

source /etc/profile
export TERM=xterm-256color
export HOME=/glade/u/home/${USER}

unset LD_LIBRARY_PATH
export PATH=/glade/work/mclong/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=False
export TMPDIR=/glade/scratch/${USER}/tmp
source activate py3_geyser

if [ ! -d logs ]; then
  mkdir -p logs
fi
