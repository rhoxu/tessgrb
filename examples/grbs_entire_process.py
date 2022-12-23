#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import tessGRB as tg

path = 'insert_desired_path_here'  # enter your own path. '/' at end is optional

# -- This function is the main analysis process. The get_first_files program must
#    have been completed before this should be started. The function finds all 
#    neighbouring ffis that contain some part of a 2-sigma error region where each
#    GRB may be located within. It then makes data cubes from those files, cuts 
#    the cubes to fit closely to that error region, and reduces the resulting cut.

#    Once again, the 'split' and 'number arguments outline how many different scripts 
#    you want running at a time (split), and which number this script is, out of 
#    that number of scripts. For example, the command below will be the third 
#    script out of seven to be run simultaneoulsy. To allow for flexibility, this 
#    system is in built - just duplicate this script the desired number of times 
#    while iterating the 'number' argument for each (from 1 - split).

#    Note that this process, at its peak, is highly memory consuming. It may well
#    reach upwards of 500Gb of memory usage during the reduction stage, as it is 
#    reading/writing vasts amount of data -- #
  
tg.grbs_entire(path, split=7, number=3) 


