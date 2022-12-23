# tessgrb

Hello! This module is built to find GRB signals in TESS data. It involves a whole lot of downloading, and a whole lot of 
optimising to find the best sections of the downloaded data to look at. The data is reduced using TESSreduce 
(see https://github.com/CheerfulUser/TESSreduce), though some modifications to the package, as well as Astrocut, are required. 
The modified modules are located in the "modules" folder. The main module file is **tessGRB.py** in the main repository directory. I'm not too familiar with github just yet, so there is no convenient pip install process as of now - just download the module into your working directory.

The processes of interest are the `get_files()` and `grbs_entire()` functions. As described in the files within the "examples" folder, they go 
through the whole process for each *possibly* observed GRB which has a defined position error (from https://user-web.icecube.wisc.edu/~grbweb_public/Summary_table.html) too great to be analysed within a regular 90x90 TESSreduce frame. The number of 
individual GRBs that meet this criteria (as of Dec 2022) is 38. The `get_files()` function must be complete before `grbs_entire()` can be performed. 
The processes can be easliy divided into simultaneous tasks - please read the comments in the "examples" files to understand how this works. Note that
the `grbs_entire()` process is unfortunately hugely time/space/memory consuming, due to the sheer amount of data TESS spits out. Thus dividing into
a number of scripts may not be advantageous. 
