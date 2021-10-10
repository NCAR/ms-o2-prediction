#! /usr/bin/env python
#BSUB -P NCGD0011
#BSUB -W 24:00
#BSUB -n 1
#BSUB -J download
#BSUB -o /glade/scratch/mclong/logs/dl.%J
#BSUB -e /glade/scratch/mclong/logs/dl.%J
#BSUB -q geyser
#BSUB -N

import os
from subprocess import call
from datetime import datetime

now = datetime.now()

name_of_extraction = 'global_req_temp_salt_po4_no3_si'

odir = os.path.join('/glade/p/work/mclong/wod2013',name_of_extraction+'_'+now.strftime('%Y%m%d'))
if not os.path.exists(odir): call(['mkdir','-p',odir])

tmpdir = os.path.join('/glade/scratch/mclong/wod2013','wod_'+name_of_extraction)
if not os.path.exists(tmpdir): call(['mkdir','-p',tmpdir])

balls = [
    'https://data.nodc.noaa.gov/woa/WOD/SELECT/ocldb1502816882.5140.OSD.gz']

for b in balls:
    file_gz = os.path.join(odir,os.path.basename(b))
    file_out = os.path.splitext(file_gz)[0].replace(odir,tmpdir)
    call(['wget','--no-check-certificate','--directory-prefix='+odir,b])
    call('gunzip -c '+file_gz+' > '+file_out,shell=True)
    
    
