#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tessGRB as tg

path = 'insert_desired_path_here'  # enter your own path. '/' at end is optional

# -- This function downloads the preliminary files containing each GRB. The split
#    and number arguments outline how many different scripts you want running at
#    a time (split), and which number this script is, out of that number of scripts.
#    For example, the command below will be the third script out of seven to be run 
#    simultaneoulsy. To allow for flexibility, this system is in built - just 
#    duplicate this script the desired number of times while iterating the 'number'
#    argument for each (from 1 - split).

#    Note that depending on the number of scripts you want running,
#    different numbers of GRBs will be assigned to each script. This is because 
#    some GRBs require far more files to download than others, and hopefully the 
#    in built splitting mechanism makes each script relatively similar in total 
#    time (likely won't be perfect) -- #
  
tg.get_files(path, split=7, number=3) 
