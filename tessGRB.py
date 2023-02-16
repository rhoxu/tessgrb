#!/usr/bin/env python3

import os
import sys
from glob import glob

import tessreduce as tr

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import math

from time import time as t

from astrocut import CubeFactory
from astrocut import CutoutFactory
from astropy.io import fits
from astropy import wcs
from astropy.time import Time
from astropy.io.fits import getdata
from astropy.table import Table
from astropy.visualization import (SqrtStretch, ImageNormalize)
from astropy import units as u


def _extract_fits(pixelfile):
    """
    Quickly extract fits
    """
    try:
        hdu = fits.open(pixelfile)
        return hdu
    except OSError:
        print('OSError ',pixelfile)
        return
    
def _grbs_information(path):
    """
    Generates pandas dataframe of observed GRB information.
    Includes:
    - .Name
    - .RA
    - .dec
    - .error (1D 1-sigma Position Error)
    - .sector 
    - .camera
    - .chip
    """
        
    if path[-1] != '/':
        path = path + '/'
    
    if not os.path.exists(path + 'Summary_table.txt'):
        os.system('wget https://user-web.icecube.wisc.edu/~grbweb_public/Summary_table.txt')
    
    file = open('Summary_table.txt')
    summary = file.readlines()
    file.close()

    namelist = []
    ra_list = [] 
    dec_list = []
    date_list = []
    err_list = []
    t90_list = []
    flu_list = []

    for ii in range(4,len(summary)-1):
        list1 = summary[ii].split(' ')
        list2 = []
        for ob in list1: 
                if ob != '':
                    list2.append(ob)


        nameIteration = list2[0]
        raIteration = float(list2[3])
        decIteration = float(list2[4])
        dateIteration = float(list2[14])
        errorIteration = float(list2[5])
        t90Iteration = float(list2[6])
        fluIteration = float(list2[9])

        namelist.append(nameIteration)
        ra_list.append(raIteration)
        dec_list.append(decIteration)
        date_list.append(dateIteration)
        err_list.append(errorIteration)
        t90_list.append(t90Iteration)
        flu_list.append(fluIteration)


    namelist = np.array(namelist)
    ralist = np.array(ra_list)
    declist = np.array(dec_list)
    timelist = np.array(date_list)
    errlist = np.array(err_list)
    t90_list = np.array(t90_list)
    flu_list = np.array(flu_list)


    totalFrame = pd.DataFrame({
        'Name': namelist,
        'RA': ralist,
        'dec': declist,
        'time': timelist,
        'error': errlist,
        't90': t90_list,
        'flu': flu_list})

    cutFrame = totalFrame.loc[totalFrame["time"] > 58119]
    
    return cutFrame


def observed_grbs(path='/home/phys/astronomy/hro52/Code/GammaRayBurstProject/TESSgrb',verbose=False):
    
    """
    Create dataframe of observed grbs
    """
    
    if path[-1] != '/':
        path = path + '/'
    
    print('Making Observed Frame')
    cutFrame = _grbs_information(path)
    headers = [x.lower() for x in cutFrame.columns]
    
    raInd = headers.index('ra')
    ra = cutFrame[cutFrame.columns[raInd]]
    
    decInd = headers.index('dec')
    dec = cutFrame[cutFrame.columns[decInd]]
    
    timeInd = headers.index('time')
    time = cutFrame[cutFrame.columns[timeInd]]
    
    observed = []
    good_sectorlist = []
    good_camlist = []
    good_ccdlist = []

    badNames = ['GRB201120B*','GRB211225A*','GRB180906A*','GRB210126A*','GRB210928B*','GRB190422C*']

    for ii in range(len(cutFrame)):
        try: 
            
            obj = tr.spacetime_lookup(ra[ii],dec[ii],time[ii],print_table=verbose)

            for jj in range(len(obj)):
                if (obj[jj][3] == True) & (cutFrame.iloc[ii].Name not in badNames):
                    observed.append(ii)
                    good_sectorlist.append(obj[jj][2])
                    good_camlist.append(obj[jj][4])
                    good_ccdlist.append(obj[jj][5])
        except:
            pass
            
    obsFrame = cutFrame.loc[observed]
    obsFrame['sector'] = good_sectorlist
    obsFrame['camera'] = good_camlist
    obsFrame['chip'] = good_ccdlist

    obsFrame = obsFrame.reset_index()
    obsFrame = obsFrame.drop(columns='index')
    
    print('Frame Made')
    
    return obsFrame

def error_sort(obs_frame):
    
    """
    Get dataframes of low (<0.05),mid(0.05-1),high(>1) error dataframe
    """
    
    errSort = obs_frame.sort_values('error')
    low_err_frame = errSort.drop(errSort[(errSort.error <= 0) | (errSort.error >= 0.05)].index)
    mid_err_frame = errSort.drop(errSort[(errSort.error < 0.05) | (errSort.error > 1)].index)
    high_err_frame = errSort.drop(errSort[(errSort.error <= 1)].index)

    return low_err_frame, mid_err_frame, high_err_frame

def _match_events(Events, Eventtime, Eventmask, Seperation = 5):
    """
    TESSBS Code ish. Matches flagged pixels that have coincident event times of +-5 cadences and are closer than 4 pix
    seperation.
    """
    i = 0
    eventmask2 = []
    while len(Events) > i:
        coincident = (np.isclose(Eventtime[i, 0], Eventtime[i:, 0], atol = Seperation) + np.isclose(
            Eventtime[i, 1], Eventtime[i:, 1], atol = Seperation))
        dist = np.sqrt((np.array(Eventmask)[i, 0]-np.array(Eventmask)[i:, 0])**2 + (
            np.array(Eventmask)[i, 1]-np.array(Eventmask)[i:, 1])**2)
        dist = dist < 5

        coincident = coincident * dist
        if sum(coincident*1) > 1:
            newmask = Eventmask[i].copy()

            for j in (np.where(coincident)[0][1:] + i):
                newmask[0] = np.append(newmask[0], Eventmask[j][0])
                newmask[1] = np.append(newmask[1], Eventmask[j][1])
            eventmask2.append(newmask)
            Events = np.delete(Events, np.where(coincident)[0][1:]+i)
            Eventtime = np.delete(Eventtime, np.where(
                coincident)[0][1:]+i, axis=(0))
            killer = sorted(
                (np.where(coincident)[0][1:]+i), key=int, reverse=True)
            for kill in killer:
                del Eventmask[kill]
        else:
            eventmask2.append(Eventmask[i])
        i += 1
    return Events, Eventtime, eventmask2


def _touch_masks(events,eventtime,eventmask):
    """
    Checks if any part of masks border each other, combines events if so.
    """
    
    i = 0
    skip = []
    copy = np.copy(eventmask)
    while len(eventmask)-1 > i:
        
        if i not in skip:
            
            event = copy[i]
            
            # -- Get x,y values of event -- #
            x1 = np.array([event[0]])
            y1 = np.array([event[1]])
            
            if len(x1.shape) > 1:
                x1 = event[0]
                y1 = event[1]
                        
            # -- Compare with other events -- #
            for j in range(i+1,len(copy)):
                compare_event = copy[j]
                
                x2 = np.array([compare_event[0]])
                y2 = np.array([compare_event[1]])
                
                if len(x2.shape) > 1:
                    x2 = compare_event[0]
                    y2 = compare_event[1]

                # -- Calculate distances between each pixel in two events -- #
                dist_array = []
                for ii in range(x1.shape[0]):
                    for jj in range(x2.shape[0]):
                        
                        dist = np.sqrt((x1[ii]-x2[jj])**2+(y1[ii]-y2[jj])**2)
                        dist_array.append(dist)

                dist_array = np.array(dist_array)

                # -- If pixels are equal/touch, combine events -- #
                if (1.0 in dist_array) | (0.0 in dist_array):
  
                    newX = np.append(copy[i][0],copy[j][0])
                    newY = np.append(copy[i][1],copy[j][1])
                    copy[i][0] = newX
                    copy[i][1] = newY
                    skip.append(j)
        i += 1
            
    # -- Delete obsolete events -- #
    newmask = np.delete(copy,skip,axis=0)
    newevents = np.delete(events,skip,axis=0)
    neweventtime = np.delete(eventtime,skip,axis=0)
    return newevents,neweventtime,newmask

def _lightcurve(Data, Mask, Normalise = False):
    """
    TESSBS Code. Takes a whole data cube, and a binary mask of pixels to include
    in lightcurve.
    """    
    
    if type(Mask) == list:
        mask = np.zeros((Data.shape[1],Data.shape[2]))
        mask[Mask[0],Mask[1]] = 1
        Mask = mask*1.0
    Mask[Mask == 0.0] = np.nan
    LC = np.nansum(Data*Mask, axis = (1,2))
    LC[LC == 0] = np.nan
    for k in range(len(LC)):
        if np.isnan(Data[k]*Mask).all(): # np.isnan(np.sum(Data[k]*Mask)) & (np.nansum(Data[k]*Mask) == 0):
            LC[k] = np.nan
    if Normalise:
        LC = LC / np.nanmedian(LC)
    return LC

def _save_space(Save,delete=False):
    """
    TESSBS Code. Creates a path if it doesn't already exist.
    """
    try:
        if not os.path.exists(Save):
            os.makedirs(Save)
    except FileExistsError:
        if delete:
            os.system('rm {}/'.format(Save))
        else:
            pass


def _remove_emptys(files):
    """
    Deletes corrupt fits files before creating cube
    """
    
    deleted = 0
    for i in range(len(files)):
        size = os.stat(files[i])[6]
        if size < 35500000:
            os.system('rm ' + files[i])
            deleted += 1
    return deleted

def get_files(path,split=1,number=1):
    """
    Download fits files for each GRB with large position error.
    --------------
    Input:
    
    path - desired path to set up files in. In this path, a folder for each 
           grb will be created, each of which will then have folders created
           inside them outlining which Cam/Chip combination has been downloaded.
           The fits files, and resulting cuts will be downloaded inside these
           cam/chip folders within the GRB folders within the path.
    split - chosen number of individual scripts the process will be divided 
            between.
    number - which number script (eg. of 5 splits, this is #3).
    """
    
    if number > split:
        print('Invalid number/split combo. Please ensure number is no bigger than the total number of splits being made.')
        return
    
    if path[-1] != '/':
        path = path + '/'
    
    # -- Get list of large error, maybe observed GRBs -- #
    obsFrame = observed_grbs()
    s,m,l = error_sort(obsFrame) # small,med,large
    
    home_dir = os.getcwd()
    
    # -- This process defines how to weight the download process. 'Late' GRBs 
    #    have at least 3 times as much data to download, so they are split 
    #    more than the early GRBs -- #
    timeSort = l.sort_values('time')
    early = timeSort.drop(timeSort[(timeSort.time >= 59031)].index)
    late = timeSort.drop(timeSort[(timeSort.time < 59031)].index).iloc[::-1]
    
    if split == 1:
        frame = l
        start = 0
        end = len(l)
    elif split == 2:
        if number == 1:
            frame = early
            start = 0
            end = len(early)
        else:
            frame = late
            start = 0
            end = len(late)
    
    else:
        pattern = []
        initial = 3
        while len(pattern) < split-2:
            for i in range(initial):
                pattern.append(initial-2)
                if len(pattern) >= split-2:
                    break
            initial += 1
            
        opposite = []
        for num in pattern:
            opposite.append(split-num)
            
        
        earlysplit = pattern[split-3]
        latesplit = split - earlysplit
        
        if latesplit > 16:
            latesplit = 16
            earlysplit = split - latesplit
        
        early_interval = math.floor(len(early)/earlysplit)
        late_interval = math.floor(len(late)/latesplit)
        
        if number == split:
            frame = late
            start = (number-1-earlysplit) * late_interval
            end = len(late)
        elif number == earlysplit:
            frame = early
            start = (number-1) * early_interval
            end = len(early)
        elif number < earlysplit:
            frame = early
            start = (number-1) * early_interval
            end = number * early_interval 
        elif number > earlysplit:
            frame = late
            start = (number-1-earlysplit) * late_interval
            end = (number-earlysplit) * late_interval 
    
    # -- Download from TESSarchives -- #
    for i in range(start,end):
        os.chdir(path)
        
        name = frame.iloc[i].Name
        sector = frame.iloc[i].sector
        cam = frame.iloc[i].camera
        chip = frame.iloc[i].chip
        
        os.mkdir(name)
        os.chdir(path + name)
        
        try:
            os.system('wget https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_%s_ffic.sh'%sector)
            
            file = open('tesscurl_sector_%s_ffic.sh'%sector)
            filelines = file.readlines()
    
            new_folder = 'Cam{}Chip{}'.format(cam,chip)
            os.mkdir(new_folder)
    
            os.chdir(path + name + '/' + new_folder)
            
            for j in range(len(filelines)):
                if "-{}-{}-".format(cam,chip) in filelines[j]:
                    os.system(filelines[j])
                    print('\n')
                    print('Downloading {} ({} of {})'.format(name,i+1,end-start))
                    print('\n')
        
            file.close()
        
        except: 
            'Downloading {} failed!'.format(name)
    
    print('-------------Download Complete------------')
    os.chdir(home_dir)
    
    
def grbs_entire(path,split=1,number=1):
    """
    Download fits files for each GRB with large position error.
    --------------
    Input:
    
    path - desired path to set up files in. In this path, a folder for each 
           grb will be created, each of which will then have folders created
           inside them outlining which Cam/Chip combination has been downloaded.
           The fits files, and resulting cuts will be downloaded inside these
           cam/chip folders within the GRB folders within the path.
    split - chosen number of individual scripts the process will be divided 
            between.
    number - which number script (eg. of 5 splits, this is #3).
    """
    
    if number > split:
        print('Invalid number/split combo. Please ensure number is no bigger than the total number of splits being made.')
        return
    
    if path[-1] != '/':
        path = path + '/'
    
    # -- Get list of large error, maybe observed GRBs -- #
    obsFrame = observed_grbs()
    s,m,l = error_sort(obsFrame) # small,med,large
    
    home_dir = os.getcwd()
    
    # -- This process defines how to weight the download process. 'Late' GRBs 
    #    have at least 3 times as much data to download, so they are split 
    #    more than the early GRBs -- #
    timeSort = l.sort_values('time')
    early = timeSort.drop(timeSort[(timeSort.time >= 59031)].index)
    late = timeSort.drop(timeSort[(timeSort.time < 59031)].index).iloc[::-1]
    
    if split == 1:
        frame = l
        start = 0
        end = len(l)
    elif split == 2:
        if number == 1:
            frame = early
            start = 0
            end = len(early)
        else:
            frame = late
            start = 0
            end = len(late)
    
    else:
        pattern = []
        initial = 3
        while len(pattern) < split-2:
            for i in range(initial):
                pattern.append(initial-2)
                if len(pattern) >= split-2:
                    break
            initial += 1
            
        opposite = []
        for num in pattern:
            opposite.append(split-num)
            
        
        earlysplit = pattern[split-3]
        latesplit = split - earlysplit
        
        if latesplit > 16:
            latesplit = 16
            earlysplit = split - latesplit
        
        early_interval = math.floor(len(early)/earlysplit)
        late_interval = math.floor(len(late)/latesplit)
        
        if number == split:
            frame = late
            start = (number-1-earlysplit) * late_interval
            end = len(late)
        elif number == earlysplit:
            frame = early
            start = (number-1) * early_interval
            end = len(early)
        elif number < earlysplit:
            frame = early
            start = (number-1) * early_interval
            end = number * early_interval 
        elif number > earlysplit:
            frame = late
            start = (number-1-earlysplit) * late_interval
            end = (number-earlysplit) * late_interval 
    
    # -- Perform tg.entire() -- #
    for i in range(start,end):
        os.chdir(path)
        name = frame.iloc[i].Name
        
        if not os.path.exists(path + name):
            print('{} not downloaded/too recent.'.format(name))
        else:
            grb = tessgrb(name=name,path=path,entire=True)
        
    os.chdir(home_dir)
    
class tessgrb():
    
    def __init__(self,name=None,ra=None,dec=None,eventtime=None,verbose=False,
                 sig=2,path='/home/phys/astronomy/hro52/Code/GammaRayBurstProject/TESSdata/ffi/',
                 find_cut=False,entire=False):
        """
        Class for tessgrb.
        """
        
        # Given
        self.name = name
        self.ra = ra
        self.dec = dec
        self.eventtime = eventtime
        self.verbose = verbose     # in depth printouts
        self.sig = sig             # error sigma
        
        if path[-1] != '/':
            path = path + '/'
        
        if self.name is not None:
            
            p = os.getcwd()
            frame = _grbs_information(p)
            try:
                frame_id = frame[frame.Name == self.name].index[0]
                self.ra = frame.RA[frame_id]
                self.dec = frame.dec[frame_id]
                self.eventtime = frame.time[frame_id]
                self.error = frame.error[frame_id]
                self.path = path + self.name
                if not os.path.exists(self.path):
                    print('No data path found.')
                    return
            except:
                print("GRB name invalid. Remember your *'s!")
                return
                
        else:
            print('No name given! I am bad at coding so RA,Dec stuff does not work yet!')
            return
        
        self.confirm_obs()            
            
        self.invert = None    # If tile is inverted in TESS orientation (T/F)
        self.cube = None      # Unpacked fits data cube 
        self.cube_file = None # File for data cube
        self.ref_wcs = None   # WCS object to use for coord mapping
        self.xDim = 2078      # x dimension of cube (2078)
        self.yDim = 2136      # y dimension of cube (2136)
        self.grb_px = None    # x,y pixels of GRB location on cube
        self.split = None     # if cut needs to be subdivided
        self.corners = None   # corners of cuts
        self.cutCentrePx = None  # x,y pixels of calculated cut centre
        self.cut_sizes = None    # x,y dimensions of calculated cut
        self.cutCentreCoords = None  # Real-world coords of cut centre 
        self.cut_file = None  # File for cut
        self.cut = None       # Unpacked fits cut
        self.neighbours = None   # List of cam,chip tuples of neighbours
        self.xDimTrue = (44,2092) # True 
        self.yDimTrue = (30,2078) # true
        self.tessreduce = None # tessreduce object
        self.tessreduce_file = None # reduced file path
        self.flux = None # flux array
        self.times = None # time vector
        self.cutNum = None

        if find_cut:
            self.find_cut()
            
        if entire:
            self.entire()
    
    def confirm_obs(self):
        
        """
        Confirm if GRB may have been observed by TESS
        """
              
        obj = tr.spacetime_lookup(self.ra,self.dec,self.eventtime,print_table=self.verbose)

        obs = False
        for jj in range(len(obj)):
            if obj[jj][3] == True:
                if self.verbose:
                    print('GRB may have been observed!')
                self.sector = obj[jj][2]
                self.camera = obj[jj][4]
                self.chip = obj[jj][5]
                obs=True
       
        if not obs:
            print("GRB not observed. Check GRB name, including *'s!")
            self.sector = "GRB not observed. Check GRB name, including *'s!"
            self.camera = "GRB not observed. Check GRB name, including *'s!"
            self.chip = "GRB not observed. Check GRB name, including *'s!"
            return

    def delete_files(self,cam,chip):
        """
        Deletes og fits files
        """
        
        path = self.path + '/Cam{}Chip{}'.format(cam,chip)
        
        if not os.path.exists(path):
            print('No data to delete!')
            return
        else:
            os.chdir(path)
            os.system('rm *ffic.fits')
            os.chdir(self.path)

    def _make_cube(self,cam,chip,delete_files=False,ask=True,verbose=True,cubing=True):
        """
        Takes a system path and collects all fits files to form a data cube
        --------------
        Input: 
        
        cam - desired camera
        chip - desired chip
        delete_files - T/F for deleting files after cube creation
        ask - T/F for asking for permission before cubing.
        verbose - Main verbose for intermediate steps (already exists etc..)
        cubing - single verbose for 'Cubing' command
        --------------
        Creates:
        
        self.cube - extracted fits data
        self.cube_file - path of cube_file
        """
        
        
        # -- Collect Files -- #
        file_path = '{}/Cam{}Chip{}'.format(self.path,cam,chip)
        files = file_path + '/*ffic.fits'
        input_files = glob(files)
        
        if not os.path.exists(file_path):
            print('No data to cube')
            return
        
        else:
            
            nameCube = '/{}-cam{}-chip{}-cube.fits'.format(self.name,cam,chip)
            cube_file_name = file_path + nameCube
            
            if os.path.exists(cube_file_name):
                if verbose:  
                    print('Cam {} Chip {} cube already exists!'.format(cam,chip))
                self.cube_file = cube_file_name
                self.cube = _extract_fits(self.cube_file)
                self._get_cubeWCS_data(cam,chip)
                self._get_cube_info()
            else:
                
                if self.verbose:
                    # -- Ask for user input regarding size of resulting datacube -- #
                    print('Number of files to cube = {}'.format(len(input_files)))
                    size = len(input_files) * 0.0355
                    print('Estimated cube size = {:.2f} GB'.format(size))
        
                done = False
                
                if ask:
                    while not done:
                        go = input('Proceed? [y/n]')
                        if go == 'y':
                            go = True
                            done = True
                        elif go == 'n':
                            go = False
                            done = True
                        else:
                            print('Answer format invalid!')
                            
                else:
                    go=True

                # -- If user input = 'y', form cube -- #
                if not go:
                    print('Aborted')
                    return 
                else:
                    
                    deleted = _remove_emptys(input_files)
                    input_files = glob(files)
                    if len(input_files) == 0:
                        print('No files.')
                        return
                    
                    if self.verbose & deleted > 0:
                        print('Deleted {} corrupted file/s.'.format(deleted))
                    if cubing:
                        print('Cubing')
                        
                    cube_maker = CubeFactory()
                    self.cube_file = cube_maker.make_cube(input_files,
                                                          cube_file= cube_file_name,
                                                          verbose=self.verbose,
                                                          max_memory=200)
                                                          
                    self.cube = _extract_fits(self.cube_file)
                    
                    # -- Obtains dimensions, px location of GRB, wcs info -- #
                    self._get_cubeWCS_data(cam,chip)
                    self._get_cube_info()
                    
                    print('Cam {} Chip {} cube complete.'.format(cam,chip))
                    
                    if delete_files:
                        self.delete_files(cam, chip)
                
    
    def _adjust_sig(self):
        
        ts = t()
        while (self.ref_wcs == 'WCS Process Failed') and ((t()-ts)<60):
            if self.sig < 0.2:
                print('True WCS Failure')
                return
            else:
                self.sig = self.sig-0.1
                self.sig = round(self.sig,1)
                self._get_cubeWCS_data(self.camera,self.chip,exception=False)
            
            
        print('Sig reduced to {:.1f}'.format(self.sig))
            
    def _get_cubeWCS_data(self,cam,chip,exception=True):
        """
        Used to obtain WCS information from a cube of data.
        --------------
        Input: 
        
        cam - desired camera
        chip - desired chip
        
        --------------
        Creates: 
        
        self.ref_wcs - reference wcs object
        """
        
        # -- Collect files -- #
        file_path = '{}/Cam{}Chip{}/*.fits'.format(self.path,cam,chip) 
        input_files = glob(file_path)
        
        if len(input_files) == 0:
            print('No files.')
            return
        
        # -- Checks if GRB is on chip -- #
        if (cam == self.camera) & (chip == self.chip):
            home_chip = True
        else:
            home_chip = False
                
        # -- Looks through files until a legitimate WCS object is found -- #
        i = 0
        yup = False
        while (not yup) & (i<50):
            file = _extract_fits(input_files[i])
            wcsItem = wcs.WCS(file[1].header)
            try:
                errpixels = self._get_err_px(wcsItem)
                grb = wcsItem.all_world2pix(self.ra,self.dec,0)
    
                # -- Terrible: condition requires GRB error size to be >40 px in diameter -- #
                if home_chip:
                    if ((max(errpixels[1])-min(errpixels[1])) > 75) & ((grb[0] >= self.xDimTrue[0]) & (grb[0] <= self.xDimTrue[1]) & (grb[1] >= self.yDimTrue[0]) & (grb[1] <= self.yDimTrue[1])):
                        yup = True
                    else:
                        i += 1
                else:
                    if ((max(errpixels[0])-min(errpixels[0])) > 40):
                        yup = True
                    else:
                        i += 1
            except:
                i += 1
            
        # -- Sets WCS to found object -- #
        if yup:
            self.ref_wcs = wcsItem
        else:
            self.ref_wcs = 'WCS Process Failed'
            if exception:
                print('WCS Information Failed. Reducing sig...')
                self._adjust_sig()
        
    def _get_cube_info(self):
        """
        Get information on px location of GRB and dimensions
        """

        self.grb_px = self.ref_wcs.all_world2pix(self.ra,self.dec,0)
        
        self.xDim = self.cube[1].data.shape[1]
        self.yDim = self.cube[1].data.shape[0]
        
    
    def _get_err_px(self,objWCS):
        """
        Get the px coords of error ellipse
        ---------
        Input:
        
        ref_wcs - wcs data of reference image
        ---------
        Output:
        
        errpixels - pixels of ellipse boundary
        """
        
        # -- Creates a 'circle' in realspace for RA,Dec -- #
        raEll = []
        decEll = []
        for ii in np.linspace(0,2*np.pi,10000):
            raEll.append(self.ra + self.sig*self.error*np.cos(ii))
            decEll.append(self.dec + self.sig*self.error*np.sin(ii))
        
        # -- Converts circle to pixel space -- #
        
        errpixels = objWCS.all_world2pix(raEll,decEll,0)    
        errpixels = np.array(errpixels)
        
        return errpixels
    
    def _cutoff_err_ellipse(self,ellipse):
        """
        Cuts off ellipse at ffi boundary.
        ----------
        Input: 
        
        ellipse - error ellipse pixels
        ----------
        Output:
        
        ellipse_reduced - ellipse pixels inside ffi
        """
        
        # -- Define Dimensions -- #
        xsize = self.xDim
        ysize = self.yDim
        
        
        # -- Finds where ellipse is inside ffi, new ellipse = og ellipsed indexed at the places -- #
        where = np.where((ellipse[0] >= 0) & (ellipse[0] <= xsize) & (ellipse[1] >= 0) & (ellipse[1] <= ysize))
        ellipse_reduced = ellipse[:,where]
        ellipse_reduced = ellipse_reduced[:,0,:]
        
        return ellipse_reduced
    
    def _find_box(self,ellipse):
        """
        Finds corners and x,y radii of proposed cutout box.
        -------------
        Input:
        
        ellipse - reduced ellipse
        -------------
        Output:
        
        cornerLB - LowerBottom corner of box (inside ffi)
        xRad - radius in x direction of box
        yRad - radius in y direction of box
        -------------
        Creates:
            
        self.cutCentrePx - central pixels of box
        """
        
        # -- Define Dimensions -- #
        xsize = self.xDimTrue
        ysize = self.yDimTrue
        
        if len(ellipse[0]) != 0:
            
            # -- Finds box min/max with 50 pixel buffer, ensures they are inside ffi -- #
            boxXmin = math.floor(min(ellipse[0])) - 50 
            boxYmin = math.floor(min(ellipse[1])) - 50 
            boxXmax = math.ceil(max(ellipse[0])) + 50
            boxYmax = math.ceil(max(ellipse[1])) + 50
        
            if boxXmin < xsize[0]:
                boxXmin = xsize[0]
            if boxYmin < ysize[0]:
                boxYmin = ysize[0] 
            if boxXmax > xsize[1]:
                boxXmax = xsize[1]
            if boxYmax > ysize[1]:
                boxYmax = ysize[1]
            
            upLength = math.ceil(abs(self.grb_px[1]-boxYmax))
            downLength = math.floor(abs(self.grb_px[1]-boxYmin))
            leftLength = math.floor(abs(self.grb_px[0]-boxXmin))
            rightLength = math.ceil(abs(self.grb_px[0]-boxXmax))
    
            # -- Finds corners of box based on above -- #
            cornerLB = (self.grb_px[0]-leftLength,self.grb_px[1]-downLength)
            cornerRB = (self.grb_px[0]+rightLength,self.grb_px[1]-downLength)
            cornerLU = (self.grb_px[0]-leftLength,self.grb_px[1]+upLength)
            cornerRU = (self.grb_px[0]+rightLength,self.grb_px[1]+upLength)
        
        else:
            cornerLB = (0,0)
            cornerRB = (self.xDim,0)
            cornerLU = (0,self.yDim)
            cornerRU = (self.xDim,self.yDim)
        
        # -- Ensures corners are inside ffi -- #
        cornerLB = (max([xsize[0],cornerLB[0]]),max(ysize[0],cornerLB[1]))
        cornerRB = (min([xsize[1],cornerRB[0]]),max(ysize[0],cornerRB[1]))
        cornerLU = (max([xsize[0],cornerLU[0]]),min(ysize[1],cornerLU[1]))
        cornerRU = (min([xsize[1],cornerRU[0]]),min(ysize[1],cornerRU[1]))
    
        # -- Calculates the x,y radii of the box -- #
        xRad = (cornerRB[0]-cornerLB[0])/2
        yRad = (cornerRU[1]-cornerRB[1])/2
                
        # -- Finds patch centre in px space -- #
        self.cutCentrePx = (cornerLB[0]+xRad,cornerLB[1]+yRad)
    
        return cornerLB,xRad,yRad
    
    def _quarter_cuts(self):
        """
        Finds centre pxs/coords of four quartered cuts.
        -------------
        Returns:
        
        split - type of split (quarter)        
        -------------
        Creates:
            
        self.cut_sizes - 4 x,y cut sizes
        self.cutCentrePx - 4 x,y cut centre px
        self.cutCentreCoords - 4 RA,DEC cut centre coords
        """
        
        split = 'quarter'
        centrePxs = []
        centreCoords = []
        
        # -- Create centres and append them -- #
        centre_pix1 = (self.cutCentrePx[0] - 1/4*self.cut_sizes[0],self.cutCentrePx[1] - 1/4*self.cut_sizes[1])
        centre_pix2 = (self.cutCentrePx[0] + 1/4*self.cut_sizes[0],self.cutCentrePx[1] - 1/4*self.cut_sizes[1])
        centre_pix3 = (self.cutCentrePx[0] - 1/4*self.cut_sizes[0],self.cutCentrePx[1] + 1/4*self.cut_sizes[1])
        centre_pix4 = (self.cutCentrePx[0] + 1/4*self.cut_sizes[0],self.cutCentrePx[1] + 1/4*self.cut_sizes[1])
        centrePxs.append(centre_pix1)
        centrePxs.append(centre_pix2)
        centrePxs.append(centre_pix3)
        centrePxs.append(centre_pix4)
        
        # -- Get quartered cut_sizes -- #
        cut_sizes = (self.cut_sizes[0]/2,self.cut_sizes[1]/2)
        
        # -- Append real world coords of centres -- #
        centreCoords.append(self.ref_wcs.all_pix2world(centre_pix1[0],centre_pix1[1],0))
        centreCoords.append(self.ref_wcs.all_pix2world(centre_pix2[0],centre_pix2[1],0))
        centreCoords.append(self.ref_wcs.all_pix2world(centre_pix3[0],centre_pix3[1],0))
        centreCoords.append(self.ref_wcs.all_pix2world(centre_pix4[0],centre_pix4[1],0))

        self.cut_sizes = cut_sizes
        self.cutCentrePx = centrePxs
        self.cutCentreCoords = centreCoords
        
        return split
    
    def _halve_cuts(self):
        """
        Finds centre pxs/coords of two halved cuts.
        -------------
        Returns:
        
        split - type of split (vert/hor)        
        -------------
        Creates:
            
        self.cut_sizes - 2 x,y cut sizes
        self.cutCentrePx - 2 x,y cut centre px
        self.cutCentreCoords - 2 RA,DEC cut centre coords
        """
        
        centrePxs = []
        centreCoords = []
        if self.cut_sizes[0] >= self.cut_sizes[1]:
            split = 'vert' 
            centre_pix1 = (self.cutCentrePx[0] - 1/4*self.cut_sizes[0],self.cutCentrePx[1])
            centre_pix2 = (self.cutCentrePx[0] + 1/4*self.cut_sizes[0],self.cutCentrePx[1])
            cut_sizes = (self.cut_sizes[0]/2,self.cut_sizes[1])
            centrePxs.append(centre_pix1)
            centrePxs.append(centre_pix2)
            centreCoords.append(self.ref_wcs.all_pix2world(centre_pix1[0],centre_pix1[1],0))
            centreCoords.append(self.ref_wcs.all_pix2world(centre_pix2[0],centre_pix2[1],0))

        else:
            split = 'hor' 
            centre_pix1 = (self.cutCentrePx[0],self.cutCentrePx[1] - 1/4*self.cut_sizes[1])
            centre_pix2 = (self.cutCentrePx[0],self.cutCentrePx[1] + 1/4*self.cut_sizes[1])
            cut_sizes = (self.cut_sizes[0],self.cut_sizes[1]/2)
            centrePxs.append(centre_pix1)
            centrePxs.append(centre_pix2)
            centreCoords.append(self.ref_wcs.all_pix2world(centre_pix1[0],centre_pix1[1],0))
            centreCoords.append(self.ref_wcs.all_pix2world(centre_pix2[0],centre_pix2[1],0))
        
    
        self.cut_sizes = cut_sizes
        self.cutCentrePx = centrePxs
        self.cutCentreCoords = centreCoords
        
        return split
    
    def _split_cuts(self):
        """
        Checks if proposed cut would be larger 2/5 or 4/5 total area.
        If >4/5, quarters. Else if > 2/5, halves. 
        """
        
        # -- Finds area of proposed cut -- #
        cut_area = self.cut_sizes[0]*self.cut_sizes[1]
        
        # -- If small area, no splitting -- #
        if cut_area < (2048*2048/(5/2)):
            split = None

        # -- If large area, quarter -- #
        elif cut_area > 4/5 * 2048*2048:
            split = self._quarter_cuts()            
        
        # -- If mid area, halve -- #
        else:
            split = self._halve_cuts()
        
        return split
    
    def _narrow_cut(self,LB,cut_sizes,ellipse):
        """
        Narrows new cut to improved area.
        -------------
        Input:
        
        LB - left bottom corner of cut
        cut_sizes - x,y size of cut
        ellipse - full error ellipse
        -------------
        Output:
        
        LB - left bottom corner of narrowed cut
        xRad - radius in x direction of narrowed cut
        yRad - radius in y direction of narrowed cut
        """

        # -- Finds corners and thus cut size -- #
        RB = (LB[0]+cut_sizes[0],LB[1])
        LU = (LB[0],LB[1]+cut_sizes[1])
    
        xsize = (LB[0],RB[0])
        ysize = (LB[1],LU[1])
        
        # -- Finds where ellipse is inside cut, new ellipse = og ellipsed indexed at the places -- #
        where = np.where((ellipse[0] >= xsize[0]) & (ellipse[0] <= xsize[1]) & (ellipse[1] >= ysize[0]) & (ellipse[1] <= ysize[1]))
        ellipse_reduced = ellipse[:,where]
        ellipse_reduced = ellipse_reduced[:,0,:]
    
        # -- If no ellipse inside cut, implied no narrowing required -- #
        if ellipse_reduced.shape[1] == 0:
            xRad = (xsize[1]-xsize[0])/2
            yRad = (ysize[1]-ysize[0])/2
            return LB,xRad,yRad
    
        # -- Finds box min/max with 50 pixel buffer -- #
        boxXmin = math.floor(min(ellipse_reduced[0])) - 50 
        boxYmin = math.floor(min(ellipse_reduced[1])) - 50 
        boxXmax = math.ceil(max(ellipse_reduced[0])) + 50
        boxYmax = math.ceil(max(ellipse_reduced[1])) + 50
        
        
        # -- Finds where ellipse is inside cut X range -- #
        where = np.where((ellipse[0] >= xsize[0]) & (ellipse[0] <= xsize[1])) 
        ellipseX = ellipse[:,where]
        ellipseX = ellipseX[:,0,:]
        
        # -- Finds where ellipse is inside cut Y range -- #
        where = np.where((ellipse[1] >= ysize[0]) & (ellipse[1] <= ysize[1]))
        ellipseY = ellipse[:,where]
        ellipseY = ellipseY[:,0,:]    
    
        # -- Sets boxX/Y min/max -- #
        if (boxXmin < xsize[0]) | (min(ellipseY[0]) < xsize[0]):
            boxXmin = xsize[0]
        if (boxYmin < ysize[0]) | (min(ellipseX[1]) < ysize[0]):
            boxYmin = ysize[0] 
        if (boxXmax > xsize[1]) | (max(ellipseY[0]) > xsize[1]):
            boxXmax = xsize[1]
        if (boxYmax > ysize[1]) | (max(ellipseX[1]) > ysize[1]):
            boxYmax = ysize[1]

        LB = (boxXmin,boxYmin)
    
        xRad = (boxXmax - boxXmin)/2
        yRad = (boxYmax - boxYmin)/2

        return LB, xRad, yRad
    
    
    def _print_cut(self,cam,chip,plot=True,proj=False):
        """
        Finds and displays ffi region with GRB location and proposed cutout region. 
        Returns information on the cutout to be used in astrocut.
        ---------
        Input:
        
        cam - desired camera
        chip - desired chip
        plot - T/F for plotting
        ---------
        Creates:
        
        self.cut_sizes - tuple/list of tuples of cutsizes
        self.og_cutsizes - tuple of pre-split cut sizes
        self.cutCentreCoords - tuple/list of tuples of cut(s) central RA/Dec
        self.cutCentrePx - tuple/list of tuples of cut(s) central px
        self.split - str about split nature (None,hor,vert,quarter)
        self.corners = tuple/list of tuples of cut corners x,y px
        """
    
        # -- Checks if GRB is on chip -- #
        if (cam == self.camera) & (chip == self.chip):
            home_chip = True
        else:
            home_chip = False
        
        # -- Creates and reduces error ellipse -- #
        ellipse = self._get_err_px(self.ref_wcs)
        #ellipse = self._cutoff_err_ellipse(ellipse)
                
        # -- Creates box and calculates its information -- #
        cornerLB,xRad,yRad = self._find_box(ellipse)
        cutout_coords = self.ref_wcs.all_pix2world(self.cutCentrePx[0],self.cutCentrePx[1],0)
        
        # -- Defines variables based on single cut -- #
        self.cut_sizes = (2*xRad,2*yRad)
        self.og_cutsizes = self.cut_sizes
        self.cutCentreCoords = cutout_coords
        
        # -- Finds split -- #
        self.split = self._split_cuts()
        
        # -- Creates list of corners/xrads/yrads for rectangle patch plotting -- #
        # -- based on split. Adjusts self.variables accordingly -- #
        if self.split is None:
            
            corners = [cornerLB]
            xRads = [xRad]
            yRads = [yRad]
            
        elif (self.split == 'vert') | (self.split == 'hor'):
            
            cut_sizes = self.cut_sizes
            ellipse = self._get_err_px(self.ref_wcs)
            
            # -- Define corners -- #
            if self.split == 'vert':
                cornerLB1 = cornerLB
                cornerLB2 = (cornerLB[0]+self.cut_sizes[0],cornerLB[1]) 
            elif self.split == 'hor':
                cornerLB1 = cornerLB
                cornerLB2 = (cornerLB[0],cornerLB[1]+self.cut_sizes[1])
                
            # -- Narrow cuts if need be -- #
            cornerLB1, xRad1, yRad1 = self._narrow_cut(cornerLB1,cut_sizes,ellipse)
            cornerLB2, xRad2, yRad2 = self._narrow_cut(cornerLB2,cut_sizes,ellipse)
            
            # -- Define lists -- #
            corners = [cornerLB1,cornerLB2]
            xRads = [xRad1,xRad2]
            yRads = [yRad1,yRad2]
        
            # -- Adjust variables -- #
            self.cut_sizes = [(xRad1*2,yRad1*2),(xRad2*2,yRad2*2)]
            centre1 = (cornerLB1[0]+xRad1,cornerLB1[1]+yRad1)
            centre2 = (cornerLB2[0]+xRad2,cornerLB2[1]+yRad2)
            coords1 = self.ref_wcs.all_pix2world(centre1[0],centre1[1],0)
            coords2 = self.ref_wcs.all_pix2world(centre2[0],centre2[1],0)
            self.cutCentrePx = [(centre1),(centre2)]
            self.cutCentreCoords = [(coords1),(coords2)]
            
        elif self.split == 'quarter':
            
            cut_sizes = self.cut_sizes
            ellipse = self._get_err_px(self.ref_wcs)
        
            # -- Define corners -- #
            cornerLB1 = cornerLB
            cornerLB2 = (cornerLB[0]+self.cut_sizes[0],cornerLB[1])
            cornerLB3 = (cornerLB[0],cornerLB[1]+self.cut_sizes[1])
            cornerLB4 = (cornerLB[0]+self.cut_sizes[0],cornerLB[1]+self.cut_sizes[1])
            
            # -- Narrow cuts if need be -- #
            cornerLB1, xRad1, yRad1 = self._narrow_cut(cornerLB1,cut_sizes,ellipse)
            cornerLB2, xRad2, yRad2 = self._narrow_cut(cornerLB2,cut_sizes,ellipse)
            cornerLB3, xRad3, yRad3 = self._narrow_cut(cornerLB3,cut_sizes,ellipse)
            cornerLB4, xRad4, yRad4 = self._narrow_cut(cornerLB4,cut_sizes,ellipse)

            # -- Define lists -- #
            corners = [(cornerLB1),(cornerLB2),(cornerLB3),(cornerLB4)]
            xRads = [xRad1,xRad2,xRad3,xRad4]
            yRads = [yRad1,yRad2,yRad3,yRad4]
            
            # -- Adjust variables -- #
            self.cut_sizes = []
            self.cutCentrePx = []
            self.cutCentreCoords = []
            for i in range(len(corners)):
                self.cut_sizes.append((xRads[i]*2,yRads[i]*2))
                centre = (corners[i][0]+xRads[i],corners[i][1]+yRads[i])
                self.cutCentrePx.append(centre)
                coords = self.ref_wcs.all_pix2world(centre[0],centre[1],0)
                self.cutCentreCoords.append(coords)
            
        self.corners = corners
            
        if plot:
            # -- Plots data -- #
            fig = plt.figure(constrained_layout=False, figsize=(6,6))
            
            if proj:
                ax = plt.subplot(projection=self.ref_wcs)
            else:
                ax = plt.subplot()
            
            # -- Real rectangle edge -- #
            rectangleTotal = patches.Rectangle((44,30), 2048, 2048,edgecolor='black',facecolor='none',alpha=0.5)
            
            # -- Sets title -- #
            if home_chip:
                ax.set_title('{} Camera {} Chip {}'.format(self.name,cam,chip))
                ax.scatter(self.grb_px[0],self.grb_px[1],color='g',label='Estimated GRB Location')
            else:
                ax.set_title('{} Camera {} Chip {} (GRB is on Cam {} Chip {})'.format(self.name,cam,chip,self.camera,self.chip))
            ax.set_xlim(0,self.xDim)
            ax.set_ylim(0,self.yDim)
            ax.grid()
            ax.add_patch(rectangleTotal)
            
            ax.plot(ellipse[0],ellipse[1],color='g',marker=',',label='2$\sigma$ Position Error Radius')
            
            # -- Adds cuts -- #
            colors = ['red','blue','purple','orangered']
            for i in range(len(corners)):
                if len(corners) == 1:
                    x = self.cut_sizes[0]
                    y = self.cut_sizes[1]
                    centrePx = self.cutCentrePx
                else:
                    x = self.cut_sizes[i][0]
                    y = self.cut_sizes[i][1]
                    centrePx = self.cutCentrePx[i]
                    
                rectangle = patches.Rectangle(corners[i],x,y,edgecolor=colors[i],
                                              facecolor='none',alpha=1,label='Cut {}'.format(i+1))
                
                #ax.scatter(centrePx[0],centrePx[1],color=colors[i],marker='x')
                ax.add_patch(rectangle)
                
            return fig, ax
            

    def find_cut(self,cam=None,chip=None,delete_files=False,
                 plot=True,ask=True,verbose=True,cubing=True,
                 proj=False,replot=False):
        """
        Finds the cut(s) required for this chip. Involves creating a datacube.
        ------------------------------------
        Input: 
        
        cam - desired camera
        chip - desired chip
        delete_files - T/F for deleting files after cube creation
        ask - T/F for asking for permission before cubing.
        verbose - Main verbose for intermediate steps (already exists etc..)
        cubing - single verbose for 'Cubing' command
        ------------------------------------
        Creates:
        
        Creates all the information needed for creating astrocuts.
        """
                
        if cam is None:
            cam = self.camera
        if chip is None:
            chip = self.chip
        
        file_path = '{}/Cam{}Chip{}'.format(self.path,cam,chip) 
        if not os.path.exists(file_path):
            print('No data path found.')
            
        else:
                
            # -- Check if small error -- # 
            if self.error <= 0.05:
                print('GRB Error Small! Use TESSreduce.')
                return
            
            else:
                if self.verbose:
                    print('Error = {:.2f} deg'.format(self.error))
                
                # -- Make Cube -- #
                try:
                    self._make_cube(cam,chip,delete_files,ask,verbose,cubing)
                except:
                    pass
                
                if self.cube is not None:
                    try:
                        if plot:
                            fig,ax = self._print_cut(cam,chip,plot,proj)
                            if replot:
                                return fig, ax
                        else:
                            self._print_cut(cam,chip,plot,proj)
                    except:
                        print('WCS non-convergence :( -- error too large.')
                        if replot:
                            return None, None
    
    def make_cut(self,cam=None,chip=None,ask=True,verbose=True,cubing=True):
        """
        Make cut(s) for this chip.
        ------------------------------------
        Input: 
        
        cam - desired camera
        chip - desired chip
        ask - T/F for asking for permission before cubing.
        verbose - Main verbose for intermediate steps (already exists etc..)
        cubing - single verbose for 'Cubing' command
        ------------------------------------
        Creates: 
        
        self.cut - cut object
        self.cut_file - path to cut
        """
        
        if cam is None:
            cam = self.camera
        if chip is None:
            chip = self.chip
        
        # -- Gets chip path, checks existance -- #
        file_path = '{}/Cam{}Chip{}'.format(self.path,cam,chip) 
        
        if not os.path.exists(file_path):
            print('No data to cut')
            return
        
        else:
            # -- gets cut information through find_cut() -- #
            self.find_cut(cam,chip,ask=False,plot=False,verbose=False,cubing=False)

            name_cuts = []
            cut_paths = []
            
            # -- Depending on split, creates lists for cutting -- #
            if self.split is None:
                
                file_path = '{}/Cam{}Chip{}'.format(self.path,cam,chip)
                
                name_cut = '{}-cam{}-chip{}-{}sigma-cut.fits'.format(self.name,cam,chip,self.sig)
                cut_path = file_path + '/' + name_cut
                
                name_cuts.append(name_cut)
                cut_paths.append(cut_path)
                extra = False

            elif self.split == 'quarter':
                extra = True
                for i in range(4):
                    file_path = '{}/Cam{}Chip{}'.format(self.path,cam,chip) 
                    name_cut = '{}-cam{}-chip{}-{}sigma-cut{}.fits'.format(self.name,cam,chip,self.sig,i+1)
                    cut_path = file_path + '/' + name_cut
            
                    name_cuts.append(name_cut)
                    cut_paths.append(cut_path)

            else: # split is a half, either vert/hor
                extra=True
                for i in range(2):
                    file_path = '{}/Cam{}Chip{}'.format(self.path,cam,chip) 
                    name_cut = '{}-cam{}-chip{}-{}sigma-cut{}.fits'.format(self.name,cam,chip,self.sig,i+1)
                    cut_path = file_path + '/' + name_cut
                    
                    name_cuts.append(name_cut)
                    cut_paths.append(cut_path)
                    
            # -- Make all cuts required -- #
            for i in range(len(cut_paths)):
                
                # -- Check cut existance -- #
                cut_path = cut_paths[i]
                if os.path.exists(cut_path):
                    if not extra:
                        print('Cam {} Chip {} cut already made!'.format(cam,chip))
                    else:
                        print('Cam {} Chip {} cut {} already made!'.format(cam,chip,i+1))
                    
                    self.cut_file = cut_path
                    self.cut = _extract_fits(self.cut_file)      
                else:
                    
                    # -- Get individual cut info -- #
                    if len(cut_paths) == 1:
                        xsize = self.cut_sizes[0]
                        ysize = self.cut_sizes[1]
                        printout = 'Cutting'
                        coords = self.cutCentreCoords
                    else:
                        xsize = self.cut_sizes[i][0]
                        ysize = self.cut_sizes[i][1]
                        printout = 'Cutting #{}'.format(i+1)
                        coords = self.cutCentreCoords[i]
                                                
                    name = name_cuts[i]
                    
                    my_cutter = CutoutFactory()
                    outpath = '{}/Cam{}Chip{}'.format(self.path,cam,chip)
                    
                    if verbose:
                        print(printout)
                                  
                    # -- Cut -- #
                    self.cut_file = my_cutter.cube_cut(self.cube_file, 
                                                       "{} {}".format(coords[0],coords[1]), 
                                                       (xsize,ysize), 
                                                       output_path = outpath,
                                                       target_pixel_file = name,
                                                       verbose=self.verbose) 
                    
                    self.cut = _extract_fits(self.cut_file)
                    
                    if not extra:
                        print('Cam {} Chip {} cut complete.'.format(cam,chip))
                    else:
                        print('Cam {} Chip {} cut {} complete.'.format(cam,chip,i+1))
                    
            
            
    def _all_cuts(self,cam,chip,cut_number,home_chip):
        """
        Used for confirming cut, plots all split cuts together.
        ------------------------------------
        Input: 
        
        cam - desired camera
        chip - desired chip
        cut_number - number of total cuts (2/4)
        home_chip - T/F if grb is on chip
        """
        
        # -- Gets size of single cut -- #
        xsize = self.og_cutsizes[0]
        ysize = self.og_cutsizes[1]
        
        # -- Creates plot 
        rat = xsize/ysize   # ratio
        plt.figure(figsize=(7,7/rat))
        
        plt.xlim(0,math.ceil(xsize))
        plt.ylim(0,math.ceil(ysize))
        
        plt.grid()
                
        colours = ['red','blue','purple','orangered']
        
        # -- plots cut WCS together
        for i in range(cut_number):
            
            # -- Gets cut path -- #
            file_path = '{}/Cam{}Chip{}'.format(self.path,cam,chip)      
            name = '{}-cam{}-chip{}-{}sigma-cut{}.fits'.format(self.name,cam,chip,self.sig,i+1)
            cut_path = file_path + '/' + name
            
            # -- Checks path exists -- #
            if not os.path.exists(cut_path):
                print('Not all cuts made. Either call make_cut, or define a single made cut.')
                return
            
            cut = _extract_fits(cut_path)
            cut_wcs = wcs.WCS(cut[2].header) 
            cut_grb_px = cut_wcs.all_world2pix(self.ra,self.dec,0)
            
            # -- Uses corner of cut to get a transformation vector -- #
            transform = np.array(self.corners[i]) - np.array(self.corners[0])
            
            ellipse = self._get_err_px(cut_wcs)
            
            # -- Finds where ellipse is inside cut -- #
            where = np.where((ellipse[0] >= 0) & (ellipse[0] <= self.cut_sizes[i][0]) & (ellipse[1] >= 0) & (ellipse[1] <= self.cut_sizes[i][1]))
            ellipse_reduced = ellipse[:,where]
            ellipse_reduced = ellipse_reduced[:,0,:]
            
            ellipse_reduced = np.array(ellipse_reduced)
            ellipse = np.zeros(shape=ellipse_reduced.shape)
            
            # -- Get shifted cut ellipse -- #
            ellipse[0] = ellipse_reduced[0] + transform[0]
            ellipse[1] = ellipse_reduced[1] + transform[1]
            
            # -- If GRB is on cut, find its transformed position -- #
            if (cut_grb_px[0]<self.cut_sizes[i][0]) & (cut_grb_px[0]>0) & (cut_grb_px[1]<self.cut_sizes[i][1]) & (cut_grb_px[1]>0):
                grb = (cut_grb_px[0] + transform[0],cut_grb_px[1] + transform[1])
              
            # -- Plot cut ellipse -- #
            plt.plot(ellipse[0],ellipse[1],marker=',',color=colours[i])

        # -- Plot GRB location -- #
        if home_chip:
            plt.title('Camera {} Chip {}'.format(cam,chip))
            plt.scatter(grb[0],grb[1],color='g')
        else:
            plt.title('Camera {} Chip {} (GRB is on Cam {} Chip {})'.format(cam,chip,self.camera,self.chip))
            
            
    def confirm_cut(self,cam=None,chip=None,cut=None,plot=True):
        """
        Displays cut with error region.
        ------------------------------------
        Input: 
        
        cam - desired camera
        chip - desired chip
        cut - desired cut number
        ------------------------------------
        Creates: 
        
        self.cut - cut object of this cut
        self.cut_file - file path of this cut
        self.cut_wcs - wcs object of this cut
        """
        
        
        if cam is None:
            cam = self.camera
        if chip is None:
            chip = self.chip
        
        # -- Checks if GRB on chip -- #
        if (cam == self.camera) & (chip == self.chip):
            home_chip = True
        else:
            home_chip = False
            
        # -- Finds cut info -- #
        self.find_cut(cam,chip,ask=False,plot=False,verbose=False,cubing=False)

        # -- Defines path of cut depending on split -- #
        if self.split is None:
            
            file_path = '{}/Cam{}Chip{}'.format(self.path,cam,chip) 
            name = '{}-cam{}-chip{}-{}sigma-cut.fits'.format(self.name,cam,chip,self.sig)
            cut_path = file_path + '/' + name
            extra = False
            cut_size = self.cut_sizes
            
        else:
            # -- used for counting total number of cuts -- #
            if self.split == 'quarter':
                end = '4'
            else:
                end = '2'
            
            if cut is not None:
                file_path = '{}/Cam{}Chip{}'.format(self.path,cam,chip)      
                name = '{}-cam{}-chip{}-{}sigma-cut{}.fits'.format(self.name,cam,chip,self.sig,cut)
                cut_path = file_path + '/' + name
                extra = True
                cut_size = self.cut_sizes[cut-1]
            # -- If no cut specified, but cut was split, display all together -- #
            else:
                self._all_cuts(cam,chip,int(end),home_chip)
                return
        
        # -- Checks if path exists -- #
        if os.path.exists(cut_path):
            
            self.cut_file = cut_path
            self.cut = _extract_fits(self.cut_file)
            self.cut_wcs = wcs.WCS(self.cut[2].header) 
            cut_grb_px = self.cut_wcs.all_world2pix(self.ra,self.dec,0)
            
            
            if plot:
                
                # -- Creates ellipse and reduces it based on cut -- #
                ellipse = self._get_err_px(self.cut_wcs)
                ellipse_red =  self._cutoff_err_ellipse(ellipse)
                rat = cut_size[0]/cut_size[1] # ratio
                
                plt.figure(figsize=(4,4/rat))
                plt.xlim(0,cut_size[0])
                plt.ylim(0,cut_size[1])
                
                # -- Titles/GRB plotting -- #
                if not extra:
                    if home_chip:
                        plt.title('Camera {} Chip {}'.format(cam,chip))
                        plt.scatter(cut_grb_px[0],cut_grb_px[1],color='g')
                    else:
                        plt.title('Camera {} Chip {} (GRB is on Cam {} Chip {})'.format(cam,chip,self.camera,self.chip))
            
                else:
                    if (cut_grb_px[0]<cut_size[0]) & (cut_grb_px[0]>0) & (cut_grb_px[1]<cut_size[1]) & (cut_grb_px[1]>0):
                        plt.title('Camera {} Chip {} Cut {}'.format(cam,chip,cut))
                        plt.scatter(cut_grb_px[0],cut_grb_px[1],color='g')
                    else:
                        if home_chip:
                            
                            a = np.arange(1,int(end)+1,1)
                            b = [cut]
                            other = list(set(a).symmetric_difference(set(b)))
                            plt.title('Camera {} Chip {} Cut {} (GRB is on another cut {})'.format(cam,chip,cut,other))
                        else:
                            plt.title('Camera {} Chip {} Cut {} (GRB is on Cam {} Chip {})'.format(cam,chip,cut,self.camera,self.chip))
            
                plt.grid()
                
                plt.plot(ellipse_red[0],ellipse_red[1],marker=',')
            
        else:
            print('No cut made/located to be confirmed! Call .make_cut() to assign.')

    def reduce(self,cam=None,chip=None):
        """
        Reduces all cuts on a chip using TESSreduce. bkg correlation 
        correction and final calibration are disabled due to time constraints.
        ------------------------------------
        Input:
            
        cam - desired camera
        chip - desired chip
        ------------------------------------
        Creates:
        
        self.tessreduce - tessreduce object, only useful for direct manipulation straight away
        self.tessreduce_file - file path to reduced file
        """
        
        if cam is None:
            cam = self.camera
        if chip is None:
            chip = self.chip
            
        self.find_cut(cam,chip,plot=False,ask=False,verbose=False,cubing=False)
            
        # -- Defines file path -- #
        file_path = '{}/Cam{}Chip{}'.format(self.path,cam,chip) 
            
        # -- Depending on split, creates list of cuts to reduce -- #
        cut_names = []
        reduced_names = []
        if self.split is None:
            nameCut = '{}-cam{}-chip{}-{}sigma-cut.fits'.format(self.name,cam,chip,self.sig)
            nameReducedCut = '{}-cam{}-chip{}-{}sigma-cutReduced.fits'.format(self.name,cam,chip,self.sig)
            cut_names.append(nameCut)
            reduced_names.append(nameReducedCut)
            extra = False 
        else:
            for i in range(len(self.cut_sizes)):
                nameCut = '{}-cam{}-chip{}-{}sigma-cut{}.fits'.format(self.name,cam,chip,self.sig,i+1)
                nameReducedCut = '{}-cam{}-chip{}-{}sigma-cut{}-Reduced.fits'.format(self.name,cam,chip,self.sig,i+1)
                cut_names.append(nameCut)
                reduced_names.append(nameReducedCut)
            extra = True
                            

        # -- Reduces every cut -- #
        for i in range(len(cut_names)):
            
            # -- Checks if reduced cut already exists -- #
            if os.path.exists(file_path+'/'+reduced_names[i]):
                if extra:
                    print('Cam {} Chip {} cut {} already reduced!'.format(cam,chip,i+1))
                else:
                    print('Cam {} Chip {} cut already reduced!'.format(cam,chip))

                self.tessreduce_file = file_path+'/'+reduced_names[i]
            
            else:
                
                ts = t()   # reduce timeStart
                
                # -- Reduce cut -- #
                if extra:
                    print('--Reduction Cam {} Chip {} Cut {} --'.format(cam,chip,i+1))
                else:
                    print('--Reduction Cam {} Chip {}--'.format(cam,chip))
                    
                self.cut_file = file_path + '/' + cut_names[i]
                self.cut = _extract_fits(self.cut_file)
    
                try:
                    
                    # -- Defining so can be deleted if failed -- #
                    self.tessreduce = 0
                    tCut = 0
                    data = 0
                    table = 0
                    tableTime = 0
                    timeMJD = 0
                    timeBJD = 0
                    hdulist = 0
                    
                    
                    self.tessreduce = tr.tessreduce(tpf=self.cut_file,reduce=True,
                                                        corr_correction=False,
                                                        calibrate = False)
                    
                    self.tessreduce_file = file_path+'/'+reduced_names[i]
        
                    print('--Reduction Complete (Time: {:.2f} mins)--'.format((t()-ts)/60))
                    
                    print('--Writing--')
                    tw = t()   # write timeStart
                    
                    # -- Inputs information into fits HDU -- #\
                                                
                    hdulist = fits.HDUList(self.cut)
                    hdulist[0].header['REDUCED'] = (True,'confirms if data stored in file is reduced by TESSreduce')
                    hdulist[0].header['ZP'] = (self.tessreduce.zp,'zeropoint (tesscounts) of reduced flux')
                    hdulist[0].header['ZP_E'] = (self.tessreduce.zp_e,'zeropoint error (tesscounts) of reduced flux')
                    
                    
                    print('getting data')
                    data = getdata(self.cut_file, 1)
                    table = Table(data)
                    
                    del(data)
                    
                    tableTime = table['TIME']
                    timeMJD = Time(self.tessreduce.lc[0],format='mjd')
                    timeBJD = timeMJD.btjd
                    
                    indices = []
                    for j in range(len(timeBJD)):
                        index = np.argmin(abs(np.array(tableTime) - timeBJD[j]))
                        indices.append(index)
                
                    
                    print('inputting data')
                    tCut = table[indices]
                    
                    
                    del(table)
                    del(timeMJD)
                    del(timeBJD)
                    del(tableTime)
                                        
                    tCut['TIME'] = self.tessreduce.lc[0]
                    tCut['FLUX'] = self.tessreduce.flux
                    tCut['FLUX_BKG'] = self.tessreduce.bkg
                                        
                    # -- Deletes Memory -- #
                    del(self.tessreduce)
                                        
                    hdulist[1] = fits.BinTableHDU(data=tCut,header=hdulist[1].header)            
                        
                    del(tCut)
                
                    # -- Writes data -- #
                    print('writing data')
                    hdulist.writeto(self.tessreduce_file,overwrite=True) 
                    
                    hdulist.close()
                    
                    print('--Writing Complete (Time: {:.2f} mins)--'.format((t()-tw)/60))
                    print('\n')
                
                except:
                    
                    # -- Deletes Memory -- #
                    del(self.tessreduce)
                    del(tCut)
                    del(data)
                    del(table)
                    del(tableTime)
                    del(timeMJD)
                    del(timeBJD)
                    try:
                        del(hdulist)
                    except:
                        pass
                    
                    if extra:
                        print('Reducing Cam {} Chip {} Cut {} Failed :( Time Elapsed: {:.2f} mins.'.format(cam,chip,i+1,(t()-ts)/60))
                    else:
                        print('Reducing Cam {} Chip {} Cut Failed :( Time Elapsed: {:.2f} mins.'.format(cam,chip,(t()-ts)/60))
                    print('\n')
                    pass 

    def _prelim_size_check(self,border):
        """
        Finds, based on initial chip cube, whether proposed neighbour cut is 
        large enough for the whole download process to be worth it.
        ------------------------------------
        Input: 
        
        border - str about which border to look at, defines conditions 
        ------------------------------------
        Output:
            
        size - size of predicted cut
        """
        
        ellipse = self._get_err_px(self.ref_wcs)      

        # -- Generates ellipse cutoff conditions based on border direction -- #
        if border == 'left':
            condition = (ellipse[0] <= 0) & (ellipse[1] >= 0) & (ellipse[1] <= self.yDim)
        elif border == 'right':
            condition = (ellipse[0] >= self.xDim) & (ellipse[1] >= 0) & (ellipse[1] <= self.yDim)
        elif border == 'up':
            condition = (ellipse[0] >= 0) & (ellipse[0] <= self.xDim) & (ellipse[1] >= self.yDim)
        elif border == 'down':
            condition = (ellipse[0] >= 0) & (ellipse[0] <= self.xDim) & (ellipse[1] <= 0)
        elif border == 'upleft':
            condition = (ellipse[0] <= 0) & (ellipse[1] >= self.yDim) 
        elif border == 'upright':
            condition = (ellipse[0] >= self.xDim) & (ellipse[1] >= self.yDim) 
        elif border == 'downleft':
            condition = (ellipse[0] <= 0) & (ellipse[1] <= 0) 
        elif border == 'downright':
            condition = (ellipse[0] >= self.xDim) & (ellipse[1] <= 0) 

        # -- Cuts ellipse -- #
        where = np.where(condition)
        ellipse = ellipse[:,where]
        ellipse = ellipse[:,0,:]
        
        # -- Calculate size of cut required to encompass ellipse region -- #
        if len(ellipse[0]) > 0:
            x1 = max(ellipse[0])
            x2 = min(ellipse[0])
            x = abs(x1 - x2)
            
            y1 = max(ellipse[1])
            y2 = min(ellipse[1])
            y = abs(y1 - y2)
            
            size = x*y
        
        else:
            size = 0
            
        return size

    def find_neighbour_chips(self,verbose=True):
        """
        Uses the camera/chip of the GRB and error ellipse pixels to 
        find the neighbouring camera/chip combinations that contain 
        some part of the ellipse.
        ------------------------------------
        Input:
            
        verbose - T/F for printing out neighbour info
        ------------------------------------
        Creates:
        
        self.neighbours - List of tuples of cam,chip combinations required
        """
        
        # -- Create chip and inversion array that contain information 
        #    on the orientations of the TESS ccd as given by manual. 
        #    Note that the chip array is flipped horizontally from 
        #    the manual as our positive x-axis goes to the right -- #
        chipArray = np.array([[(4,3),(1,2)],[(4,3),(1,2)],[(2,1),(3,4)],[(2,1),(3,4)]])
        invertArray = np.array([[(True,True),(False,False)],
                                [(True,True),(False,False)],
                                [(True,True),(False,False)],
                                [(True,True),(False,False)]])
        
        # -- Check north/south pointing and initialise cam array accordingly -- #
        if self.dec > 0:
            north = True
        else:
            north = False

            
        if north:
            camArray = np.array([4,3,2,1])
        else:
            camArray = np.array([1,2,3,4])
            
        
        # -- Find the chipArray index based on self.cam/chip -- #
        if self.camera != "GRB not observed. Check GRB name, including *'s!":
            for i in range(len(camArray)):
                if self.camera == camArray[i]:
                    camIndex = i
                    
            for i in range(len(chipArray[camIndex])):
                for j in range(len(chipArray[camIndex][i])):
                    if self.chip == chipArray[camIndex][i][j]:
                        chipIndex = (i,j) # row, column
                
            total_index = (camIndex,chipIndex) # [camIndex,(in-cam-row,in-cam-column)]
        else:
            print("GRB not observed. Check GRB name, including *'s!")
            return
        
        self._get_cubeWCS_data(self.camera,self.chip)
        
        if self.ref_wcs is None:
            print('WCS info failed.')
            return
        
        # -- Create error ellipse and use max/min values to find if the ellipse
        #    intersects the up,down,left,right box edges -- #
        ellipse = self._get_err_px(self.ref_wcs)
        
        right = False
        left = False
        up = False
        down = False
        
        if max(ellipse[0]) > self.xDim:
            right = True
        if min(ellipse[0]) < 0:
            left = True
        if max(ellipse[1]) > self.yDim:
            up = True
        if min(ellipse[1]) < 0:
            down = True
            
        # -- Check if inversion is required and adjust accordingly-- #
        self.invert = invertArray[total_index[0]][total_index[1][0]][total_index[1][1]]
       
        if self.invert:
            right2 = left
            left = right
            right = right2
            up2 = down
            down = up
            up = up2

        # -- Check for diagonals -- #
        upright = False
        upleft = False
        downright = False
        downleft = False
    
        if up and right:
            upright = True
        if up and left:
            upleft = True
        if down and right:
            downright = True
        if down and left:
            downleft = True
            
        # -- Calculate the neighbouring chip information. If px area of 
        #    neighbour chip is <70,000, chip is disregarded as unworthy 
        #    of full download process. -- #
        neighbour_chips = []    

        if left:
            if chipIndex[1] == 1:
                leftchip = camArray[camIndex],chipArray[camIndex][chipIndex[0]][0]
                
                if self.invert:
                    size = self._prelim_size_check('right')
                    if size > 70000:
                        leftchip = (leftchip[0],leftchip[1],'Right')
                        neighbour_chips.append(leftchip)
                    else:
                        neighbour_chips.append('Right chip too small')
                else:
                    size = self._prelim_size_check('left')
                    if size > 70000:
                        leftchip = (leftchip[0],leftchip[1],'Left')
                        neighbour_chips.append(leftchip)
                    else:
                        neighbour_chips.append('Left chip too small')
                
        if right:
            if chipIndex[1] == 0:
                rightchip = camArray[camIndex],chipArray[camIndex][chipIndex[0]][1]
                size = self._prelim_size_check('right')
                
                if self.invert:
                    size = self._prelim_size_check('left')
                    if size > 70000:
                        rightchip = (rightchip[0],rightchip[1],'Left')
                        neighbour_chips.append(rightchip)
                    else:
                        neighbour_chips.append('Left chip too small')
                else:
                    size = self._prelim_size_check('right')
                    if size > 70000:
                        rightchip = (rightchip[0],rightchip[1],'Right')
                        neighbour_chips.append(rightchip)
                    else:
                        neighbour_chips.append('Right chip too small')
                
        if up:
            if not (total_index[0] == 0) & (total_index[1][0] == 0):
                if total_index[1][0] == 0:
                    upCam = camIndex - 1
                    upCcd = (1,total_index[1][1])
                    upchip = camArray[upCam],chipArray[upCam][1][total_index[1][1]]
                else:
                    upCam = camIndex
                    upCcd = (0,total_index[1][1])
                    upchip = camArray[upCam],chipArray[upCam][0][total_index[1][1]]
                
                if self.invert:
                    size = self._prelim_size_check('down')
                    if size > 70000:
                        upchip = (upchip[0],upchip[1],'Down')
                        neighbour_chips.append(upchip)
                    else:
                        neighbour_chips.append('Down chip too small')
                else:
                    size = self._prelim_size_check('up')
                    if size > 70000:
                        upchip = (upchip[0],upchip[1],'Up')
                        neighbour_chips.append(upchip)
                    else:
                        neighbour_chips.append('Up chip too small')
                            
        if down:
            if not (total_index[0] == 3) & (total_index[1][0] == 1):
                if total_index[1][0] == 1:
                    downCam = camIndex + 1
                    downCcd = (0,total_index[1][1])
                    downchip = camArray[downCam],chipArray[downCam][0][total_index[1][1]]
                else:
                    downCam = camIndex
                    downCcd = (1,total_index[1][1])
                    downchip = camArray[downCam],chipArray[downCam][1][total_index[1][1]]
                
                if self.invert:
                    size = self._prelim_size_check('up')
                    if size > 70000:
                        downchip = (downchip[0],downchip[1],'Up')
                        neighbour_chips.append(downchip)
                    else:
                        neighbour_chips.append('Up chip too small')
                else:
                    size = self._prelim_size_check('down')
                    if size > 70000:
                        downchip = (downchip[0],downchip[1],'Down')
                        neighbour_chips.append(downchip)
                    else:
                        neighbour_chips.append('Down chip too small')

        
        if upright:
            if not (total_index[0] == 0) & (total_index[1][0] == 0) | (chipIndex[1] == 1):
                if total_index[1][0] == 0:
                    urCam = camIndex - 1
                    urchip = camArray[urCam],chipArray[urCam][1][1]
                else:
                    urCam = camIndex
                    urchip = camArray[urCam],chipArray[urCam][0][1]
                
                if self.invert:
                    size = self._prelim_size_check('downleft')
                    if size > 70000:
                        urchip = (urchip[0],urchip[1],'Downleft')
                        neighbour_chips.append(urchip)
                    else:
                        neighbour_chips.append('Downleft chip too small')
                else:
                    size = self._prelim_size_check('upright')
                    if size > 70000:
                        urchip = (urchip[0],urchip[1],'Upright')
                        neighbour_chips.append(urchip)
                    else:
                        neighbour_chips.append('Upright chip too small')
                    
        
        if upleft:
            if not (total_index[0] == 0) & (total_index[1][0] == 0) | (chipIndex[1] == 0):
                if total_index[1][0] == 0:
                    ulCam = camIndex - 1
                    ulchip = camArray[ulCam],chipArray[ulCam][1][0]
                else:
                    ulCam = camIndex
                    ulchip = camArray[ulCam],chipArray[ulCam][0][0]
                
                if self.invert:
                    size = self._prelim_size_check('downright')
                    if size > 70000:
                        ulchip = (ulchip[0],ulchip[1],'Downright')
                        neighbour_chips.append(ulchip)
                    else:
                        neighbour_chips.append('Downright chip too small')
                else:
                    size = self._prelim_size_check('upleft')
                    if size > 70000:
                        ulchip = (ulchip[0],ulchip[1],'Upleft')
                        neighbour_chips.append(ulchip)
                    else:
                        neighbour_chips.append('Upleft chip too small')

        if downright:
            if not (total_index[0] == 3) & (total_index[1][0] == 1) | (chipIndex[1] == 1):
                if total_index[1][0] == 1:
                    drCam = camIndex + 1
                    drchip = camArray[drCam],chipArray[drCam][0][1]
                else:
                    drCam = camIndex
                    drchip = camArray[drCam],chipArray[drCam][1][1]
                
                if self.invert:
                    size = self._prelim_size_check('upleft')
                    if size > 70000:
                        drchip = (drchip[0],drchip[1],'Upleft')
                        neighbour_chips.append(drchip)
                    else:
                        neighbour_chips.append('Upleft chip too small')
                else:
                    size = self._prelim_size_check('downright')
                    if size > 70000:
                        drchip = (drchip[0],drchip[1],'Downright')
                        neighbour_chips.append(drchip)
                    else:
                        neighbour_chips.append('Downright chip too small')
            
        if downleft:
            if not (total_index[0] == 3) & (total_index[1][0] == 1) | (chipIndex[1] == 0):
                if total_index[1][0] == 1:
                    dlCam = camIndex + 1
                    dlchip = camArray[dlCam],chipArray[dlCam][0][0]
                else:
                    dlCam = camIndex
                    dlchip = camArray[dlCam],chipArray[dlCam][1][0]
                
                if self.invert:
                    size = self._prelim_size_check('upright')
                    if size > 70000:
                        dlchip = (dlchip[0],dlchip[1],'Upright')
                        neighbour_chips.append(dlchip)
                    else:
                        neighbour_chips.append('Upright chip too small')
                else:
                    size = self._prelim_size_check('downleft')
                    if size > 70000:
                        dlchip = (dlchip[0],dlchip[1],'Downleft')
                        neighbour_chips.append(dlchip)
                    else:
                        neighbour_chips.append('Downleft chip too small')

        # -- prints information -- #
        if verbose:
            
            if north:
                print('Pointing: North')
            else:
                print('Pointing: South')
            
            print('This chip: Camera {}, Chip {}.'.format(self.camera,self.chip))
            print('------------------------------')
            print('Neighbouring Chips Required:')
            if neighbour_chips != []:
                for item in neighbour_chips:
                    if type(item) == str:
                        print(item)
                    else:
                        print('Camera {}, Chip {} ({}).'.format(item[0],item[1],item[2]))
                        
            else:
                print('No neighbour chips available/required.')
                        
        # -- Removes disregarded chip info to create self.neighbours -- #
        if neighbour_chips != []:
            self.neighbours = []
            for item in neighbour_chips:
                if type(item) == tuple:
                    self.neighbours.append(item[:-1])
                        
    def get_neighbour_chips(self):
        """
        Downloads neighbour chip fits files from internet
        """
        
        self.find_neighbour_chips(False)
        
        home_dir = os.getcwd()
        
        if self.neighbours is not None:
            i = 1
            for item in self.neighbours:
                cam = item[0]
                chip = item[1]
                
                os.chdir(self.path)
                
                new_folder = 'Cam{}Chip{}'.format(cam,chip)
                
                if os.path.exists(self.path + '/' + new_folder):
                    print('Cam {} Chip {} data already downloaded.'.format(cam,chip))
                else:
                    print('Cam {} Chip {} downloading.'.format(cam,chip))
                    os.system('mkdir {}'.format(new_folder))
                    file = open('tesscurl_sector_%s_ffic.sh'%self.sector)
                    filelines = file.readlines()
        
                    os.chdir(self.path + '/' + new_folder)
                    
                    for j in range(len(filelines)):
                        if "-{}-{}-".format(cam,chip) in filelines[j]:
                            os.system(filelines[j])
                            print('\n')
                            print('Downloading Cam {} Chip {} ({} of {})'.format(cam,chip,i,len(self.neighbours)))
                            print('\n')
                    
                    file.close()
                
                i += 1
        else:
            print('No neighbour chips!')
            return
                        
        os.chdir(home_dir)
            
    def get_neighbour_cubes(self,ask=True):
        """
        Make data cubes of neighbour chips
        """
        
        self.find_neighbour_chips(False)
        
        if self.neighbours is not None:
            i=1
            for item in self.neighbours:
                cam = item[0]
                chip = item[1]
                print('Cubing Cam {} Chip {} ({} of {})'.format(cam,chip,i,len(self.neighbours)))
                try:
                    self._make_cube(cam, chip, ask=ask, verbose=True,cubing=False)
                except:
                    print('Cubing Cam {} Chip {} Failed! :( '.format(cam,chip))
                    pass
                print('\n')
                i+=1
        else:
            print('No neighbour chips!')
            return
            
    def get_neighbour_cuts(self):
        """
        Make appropriate cuts for neighbour chips
        """
        
        self.find_neighbour_chips(False)
        
        if self.neighbours is not None:
            i=1
            for item in self.neighbours:
                cam = item[0]
                chip = item[1]
                print('Cutting Cam {} Chip {} ({} of {})'.format(cam,chip,i,len(self.neighbours)))
                try:
                    self.make_cut(cam,chip,ask=False,verbose=False,cubing=False)
                except:
                    print('Cutting Cam {} Chip {} Failed! :( '.format(cam,chip))
                    pass
                print('\n')
                i+=1
        else:
            print('No neighbour chips!')
            return
        
    def get_neighbour_reductions(self):
        """
        Make appropriate reductions for neighbour chips
        """
        
        self.find_neighbour_chips(False)
        
        if self.neighbours is not None:
            i = 1
            for item in self.neighbours:
                ts = t()  # timeStart
                cam = item[0]
                chip = item[1]
                print('Reducing Cam {} Chip {} ({} of {})'.format(cam,chip,i,len(self.neighbours)))

                try:
                    self.reduce(cam,chip)
                except:
                    print('Reducing Cam {} Chip {} Failed :( Time Elapsed: {:.2f} mins.'.format(cam,chip,(t()-ts)/60))
                    print('\n')
                i+=1
        else:
            print('No neighbour chips!')
            return
        
    def entire(self):
        """
        Finds, downloads, cubes, cuts, reduces. All in one.        
        """
        
        ts = t() # timeStart
        
        self.find_cut(plot=False,ask=False,verbose=False)
        
        self.get_neighbour_chips()
        
        if self.neighbours is not None:
            print('\n')
            print('------------{} -- {} Chips to Compute----------'.format(self.name,len(self.neighbours)))
            print('\n')
            
            print('---------Getting Chip Cubes---------')
            print('\n')
            self.get_neighbour_cubes(ask=False)
            
            print('---------Getting Chip Cuts---------')
            print('\n')
    
            print('Cutting Main Chip')
            self.make_cut(verbose=False)
            print('\n')
            self.get_neighbour_cuts()
                
            print('------Getting Chip Reductions------')
            print('\n')
    
            print('Reducing Main Chip')        
            self.reduce()
            self.get_neighbour_reductions()
            
            print('-----------{} Complete (Total Time: {:.2f} hrs)---------'.format(self.name,(t()-ts)/3600))

    def tpf_info(self,cam=None,chip=None,cut=None):
        """
        Gets flux,time data feom reduced tpf 
        ------------------------------------
        Input:
            
        cam - desired camera
        chip - desired chip
        """
        
        if cam is None:
            cam = self.camera
        if chip is None:
            chip = self.chip
        
            
        self.find_cut(plot=False,ask=False,verbose=False,cubing=False)
        
        file_path = '{}/Cam{}Chip{}'.format(self.path,cam,chip) 

        
        arr = np.array(self.cut_sizes)
        
        
        if len(arr.shape) > 1:
            if cut is None:
                print('Please specify cut!')
                return
            else:
                self.cutNum = cut
                tess_file = file_path + '/{}-cam{}-chip{}-{}sigma-cut{}-Reduced.fits'.format(self.name,cam,chip,self.sig,cut)

        else:
            tess_file = file_path + '/{}-cam{}-chip{}-{}sigma-cutReduced.fits'.format(self.name,cam,chip,self.sig)

        if not os.path.exists(tess_file):
            print('No reduced file.')
            
        else:
            tpf = tr.lk.TessTargetPixelFile(tess_file)
            print('Getting Flux')
            self.flux = tr.strip_units(tpf.flux)
            print('Getting Time')
            self.times = tpf.time.value
            
    def detect_events(self,cam=None,chip=None,cut=None,significance=3,minlength=2,plot=True,reuse=False):
        """
        Detects events that peak a given # of std above the local median, and last 
        for a given number of times. 
        ------------------------------------
        Input:
            
        cam - desired camera
        chip - desired chip
        significance - how many sig above median to require
        minlength - how many concurrant data points have to be above sig to detect
        plot - T/F for plotting
        reuse - T/F for returning information (see below)
        ------------------------------------
        Output (if reuse):
            
        events - list of initial time indices for each event
        eventtime - list of tuples of (time index start, time index stop)
        eventmask - list of lists of pixels included in each event 
        """      
        
        if reuse:
            rip = None,None,None,None
        else:
            rip = None
        
        # -- Set cam/chip setting -- #
        if cam is None:
            cam = self.camera
        if chip is None:
            chip = self.chip
        
        # -- Get information from reduced data -- #
        if (self.times is None) or (self.cutNum != cut):
            print('-------------Collecting flux,time arrays----------------')
            self.tpf_info(cam,chip,cut)
            print('-------------------------Done---------------------------')
            print('\n')
        
        # -- Check if event occured just before time window -- #
        if self.eventtime - self.times[0] < 0:
            print('GRB Time just outside captured window.')
            print('\n')
            return rip
        
        # -- Check if event occured during TESS break -- #
        splitStart = np.where(np.diff(self.times) == max(np.diff(self.times)))[0][0]
        if (self.eventtime >= self.times[splitStart]) & (self.eventtime < self.times[splitStart+1]):
            print('GRB occured during TESS split')
            return rip
                
        # -- Find the index closest to the eventtime -- #
        event_index = np.argmin(abs(self.times-self.eventtime))

        if event_index - (splitStart+1) < 30:
            print('Be careful! Event is just after TESS break!')
        
        
        self.confirm_cut(cut=cut,plot=False)
        
        # -- Create time/flux arrays in focused region around event -- #
        focus_times = self.times[event_index-10:event_index+20]
        focus_fluxes = self.flux[event_index-10:event_index+20,:,:]
        
        if self.eventtime <= focus_times[10]:
            detect = 10
        else:
            detect = 11
        
        # -- Calculate median/std of local region (30 indices before event) -- #
        med = np.nanmedian(self.flux[event_index-30:event_index], axis = 0)
        std = np.nanstd(self.flux[event_index-30:event_index],axis=0)
        
        # -- Find pixels that meet conditions -- #
        binary = (focus_fluxes >= (med+significance*std))
        summed_binary = np.nansum(binary[detect:detect+5],axis=0)
        X = np.where(summed_binary >= minlength)[0] # note that X = Y
        Y = np.where(summed_binary >= minlength)[1]
       
       # -- TESSBS Code, Idk what it does :) -- # 
        tarr = self.flux > 10000000
        tarr[event_index-10:event_index+20,:,:] = binary
        
        events = []
        eventmask = []
        eventtime = []
        
        for i in range(len(X)):
            temp = np.insert(tarr[:,X[i],Y[i]],0,False) # add a false value to the start of the array
            testf = np.diff(np.where(~temp)[0])
            indf = np.where(~temp)[0]
            testf[testf == 1] = 0
            testf = np.append(testf,0)
        
            if len(indf[testf>minlength]) > 0:
                for j in range(len(indf[testf>minlength])):
                    start = indf[testf>minlength][j]
                    end = (indf[testf>minlength][j] + testf[testf>minlength][j]-1)
                    #if np.nansum(Eventmask[start:end,X[i],Y[i]]) / abs(end-start) > 0.5:
                    events.append(start)
                    eventtime.append([start, end])
                    masky = [np.array(X[i]), np.array(Y[i])]
                    eventmask.append(masky)    
                   
        events = np.array(events) # initial indices in timelist of events
        eventtime = np.array(eventtime) # start and end indices of events
        
        eventmask2 = eventmask # non numpy mask just in case
        
        # -- Find coincident events -- # 
        events, eventtime, eventmask = _match_events(events, eventtime, eventmask)
        eventmask = np.array(eventmask)
        
        # -- Cuts events to only those which begin an hour or less after eventtime -- #  
        timeInterval = min(np.diff(self.times))
        allowedStart = np.floor(1/ (24 * timeInterval))
        event_checks = (events - event_index) < allowedStart
        events = events[np.where(event_checks)[0]]
        eventtime = eventtime[np.where(event_checks)[0]]
        eventmask = eventmask[np.where(event_checks)[0]]
        
        # -- Checks if any events touch / are too large for match_events -- #
        events, eventtime, eventmask = _touch_masks(events, eventtime,eventmask)
        
        if events.shape[0] == 0:
            print('No events!')
            return rip
        
        # -- Order events based on brightness -- #
        elif events.shape[0] > 1:
            
            order = self._rank_brightness(events, eventtime, eventmask)
    
            events = events[order]
            eventtime = eventtime[order]
            eventmask = eventmask[order]
        
        if plot:
            if cut is None:
                cut_size = self.cut_sizes
            else:
                cut_size = self.cut_sizes[cut-1]
            rat = cut_size[1]/cut_size[0]
            fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=False, figsize=(6,6*rat))
            ax.set_xlim(0,cut_size[0])
            ax.set_ylim(0,cut_size[1])
            
            cut_grb_px = self.cut_wcs.all_world2pix(self.ra,self.dec,0)
            ellipse = self._get_err_px(self.cut_wcs)
            ellipse_red =  self._cutoff_err_ellipse(ellipse)
            ax.plot(ellipse_red[0],ellipse_red[1])
            ax.scatter(cut_grb_px[0],cut_grb_px[1])
            
            ax.grid()
            
            counter = 1
            for event in eventmask:
                ax.scatter(event[1],event[0],marker='s',color='r',s=2,edgecolor='black')
                point = (np.median(event[1]),np.median(event[0])+5)
                ax.annotate('Event ' + str(counter),point,fontsize=10)
                
                fig, ax2 = plt.subplots(ncols=2, nrows=1, constrained_layout=False, figsize=(11.5,3.5))
                
                if counter != 1:
                    ax2[0].set_title('Event ' + str(counter) + ' Lightcurve')
                ax2[0].plot(focus_times,focus_fluxes[:,event[0],event[1]])
                ax2[0].axvline(self.eventtime,linestyle='--',alpha=1,color='black')
                ax2[0].ticklabel_format(useOffset=False)
                ax2[0].set_xlim(focus_times[0],focus_times[-1])
                ax2[0].set_xlabel('Time (MJD)')
                ax2[0].set_ylabel('TESS Counts (e$^-$/s)')
                ax2[0].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5)) 
                
                
                ax2[1].set_title('Event ' + str(counter) + ' Pixels')
                ax2[1].set_xlim(np.min(event[1])-15,np.max(event[1]+15))
                ax2[1].set_ylim(np.min(event[0])-10,np.max(event[0]+10))
                ax2[1].scatter(event[1]+0.5,event[0]+0.5,marker='s',color='r',s=50,edgecolor='black')
                ax2[1].grid()
                
                counter += 1
            
        if reuse:
            return events,eventtime,eventmask
        
    def _rank_brightness(self,events,eventtime,eventmask):
        """
        Ranks events by peak brightness.
        """        
    
        # -- Find maximums -- #
        maximums = []
        for i in range(len(events)):
            flux = self.flux[events[i]:eventtime[i,1],eventmask[i,0],eventmask[i,1]]
            maximum = np.max(flux),i
            maximums.append(maximum)
    
        # -- Get event index order based on maximums -- #
        maximums.sort(reverse=True)
        maximums = np.array(maximums)
        order = maximums[:,1]
        order = order.astype(int)
        return order
    
    def LCvideo(self,cam,chip,event,eventtime,eventmask):
        """
        Creates an mp4 video of chosen event.
        --------------
        Input: 
        
        event - initial time index in self.time of event
        eventtime - start/finish time index of event
        eventmask - pixels in event (time irrelevant)
        --------------
        """
        
        # -- Creates a mask of pixels in the event -- #
        mask = np.zeros((self.flux.shape[1],self.flux.shape[2]))
        mask[eventmask[0],eventmask[1]] = 1
        
        # -- Finds brightest pixel -- #
        position = np.where(mask)
        Mid = ([position[0][0]],[position[1][0]])
        maxcolor = 0 # Set a bad value for error identification
        for j in range(len(position[0])):
            lcpos = np.copy(self.flux[eventtime[0]:eventtime[1],position[0][j],position[1][j]])
            nonanind = np.isfinite(lcpos)
            temp = sorted(lcpos[nonanind].flatten())
            temp = np.array(temp)
            if len(temp) > 0:
                temp  = temp[-1] # get brightest point
                if temp > maxcolor:
                    maxcolor = temp
                    Mid = ([position[0][j]],[position[1][j]])
                    
        Mid = (np.round(np.mean(position[0])),np.round(np.mean(position[1])))
                    
        # -- get lightcurve of flagged pixels -- #
        LC = _lightcurve(self.flux, mask)
        lclim = np.copy(LC[eventtime[0]:eventtime[1]])
        
        # -- find limits of LC -- #
        temp = sorted(lclim[np.isfinite(lclim)].flatten())
        temp = np.array(temp)
        maxy  = temp[-1] # get 8th brightest point
        
        temp = sorted(LC[np.isfinite(LC)].flatten())
        temp = np.array(temp)
        miny  = temp[10] # get 10th faintest point
        
        ymin = miny - 0.1*miny
        ymax = maxy + 0.1*maxy
        
        ind = np.argmin(abs(self.times-self.eventtime))

        if ind % 2 == 1:
            if eventtime[1] % 2 == 1:
                endspan = eventtime[1]
            else:
                endspan = eventtime[1] - 1
        else:
            if eventtime[1] % 2 == 0:
                endspan = eventtime[1]
            else:
                endspan = eventtime[1] - 1
        
        Section = np.arange(ind-8,endspan+8,2)
        
        # Create an ImageNormalize object using a SqrtStretch object
        norm = ImageNormalize(vmin=ymin/len(position[0]), vmax=maxcolor, stretch=SqrtStretch())
        
        height = 1100/2
        width = 2200/2
        my_dpi = 100
        
        FrameSave = '/home/phys/astronomy/hro52/Code/GammaRayBurstProject/TESSdata/ffi/{}/Cam{}Chip{}/EventFrames'.format(self.name,cam,chip)
        _save_space(FrameSave,delete=True)
        
        print('Making Frames')
        for j in range(len(Section)):
            
            filename = FrameSave + '/Frame_' + str(int(j)).zfill(4)+".png"

            fig = plt.figure(figsize=(width/my_dpi,height/my_dpi),dpi=my_dpi)
            plt.subplot(1, 2, 1)
            plt.title('Event light curve')
            plt.axvspan(self.times[ind]-self.times[0],self.times[endspan]-self.times[0],color='orange',alpha = 0.5)
            plt.plot(self.times - self.times[0], LC,'k.')
        
        
            plt.ylim(ymin,ymax)
            plt.xlim(self.times[ind-8]-self.times[0],self.times[endspan+8]-self.times[0])
        
            plt.ylabel('Counts')
            plt.xlabel('Time (days)')
            plt.axvline(self.times[Section[j]]-self.times[0],color='red',lw=2)
        
            plt.subplot(1,2,2)
            plt.title('TESS image')
            self.flux[np.isnan(self.flux)] = 0
            plt.imshow(self.flux[Section[j]],origin='lower',cmap='gray',norm=norm)
            current_cmap = plt.cm.get_cmap()
            current_cmap.set_bad(color='black')
            # plt.colorbar()
        
            xlims = (Mid[1]-6.5,Mid[1]+6.5)
            ylims = (Mid[0]-6.5,Mid[0]+6.5)
            
            plt.xlim(xlims[0],xlims[1])
            plt.ylim(ylims[0],ylims[1])
            plt.ylabel('Row')
            plt.xlabel('Column')
            fig.tight_layout()
        
            ax = fig.gca()
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
            plt.savefig(filename,dpi=100)
            plt.close();
        
        VidSave = '/home/phys/astronomy/hro52/Code/GammaRayBurstProject/TESSdata/LCvideos'
        _save_space(VidSave)

        framerate = 2
        ffmpegcall = 'ffmpeg -y -nostats -loglevel 8 -f image2 -framerate ' + str(framerate) + ' -i ' + FrameSave + '/Frame_%04d.png -vcodec libx264 -pix_fmt yuv420p ' + VidSave + '/' + self.name + '-LCvideo.mp4'
        os.system(ffmpegcall)
        print('Video Made')
        
        
    def present_event(self,cam,chip,event,save=False,form='pixels',presentation='whole',cut=None):
        
        
        file_path = '{}/Cam{}Chip{}'.format(self.path,cam,chip) 
        
        arr = np.array(self.cut_sizes)
        
        
        if len(arr.shape) > 1:
            if cut is None:
                print('Please specify cut!')
                return
            else:
                tess_file = file_path + '/{}-cam{}-chip{}-{}sigma-cut{}-Reduced.fits'.format(self.name,cam,chip,self.sig,cut)

        else:
            tess_file = file_path + '/{}-cam{}-chip{}-{}sigma-cutReduced.fits'.format(self.name,cam,chip,self.sig)
        
        self.cut_file = tess_file
        self.cut = _extract_fits(self.cut_file)
        self.cut_wcs = wcs.WCS(self.cut[2].header) 
        
        cut_grb_px = self.cut_wcs.all_world2pix(self.ra,self.dec,0)
        ellipse = self._get_err_px(self.cut_wcs)
        
        if form.lower() == 'coords':
            a,b = self.cut_wcs.all_world2pix(event[0],event[1],0)
            event = (b,a)
        elif form.lower() != 'pixels':
            print('Please enter form as "pixels" or "coords".')
        
        if presentation.lower() == 'whole':
            
            LB = (min(ellipse[0])-50,min(ellipse[1])-50)
            RU = (max(ellipse[0])+50,max(ellipse[1])+50)
            
            ysize = RU[1]-LB[1]
            xsize = RU[0] - LB[0]
            
            rat = ysize/xsize

            rat = 1.214           
            
            fig = plt.figure(constrained_layout=False, figsize=(6,6*rat))
            ax = plt.subplot(projection=self.cut_wcs)
            
            ax.set_xlim(LB[0],RU[0])
            ax.set_ylim(LB[1],RU[1])
            
            ax.plot(ellipse[0],ellipse[1],color='g',label='Estimated GRB Location')
            ax.scatter(cut_grb_px[0],cut_grb_px[1],color='g',label='2σ Position Error Radius')
            
            ax.grid()
            
            x,y = ax.coords
            
            x.set_ticks(number=5)
            y.set_ticks(number=5)
            
            #x.set_ticks_position('b')
            #x.set_ticklabel_position('b')
            #x.set_axislabel_position('b')
            
            #y.set_ticks_position('l')
            #y.set_ticklabel_position('l')
            #y.set_axislabel_position('l')
            
            x.set_ticklabel(size=12)
            y.set_ticklabel(size=12)
            
            x.set_axislabel('Right Ascension',size=12)
            y.set_axislabel('Declination',size=12)

            if event[1] > 1/2 * xsize:
                if event[0] > 4/5*ysize:
                    point = (event[1] - 400,event[0]-90)
                else:
                    point = (event[1]-400,event[0]+30)
            else:
                if event[0] > 4/5*ysize:
                    point = (event[1],event[0]-90)
                else:
                    point = (event[1],event[0]+30)

            ax.scatter(event[1],event[0],marker='s',color='r',s=16,edgecolor='black')
            #ax.annotate('{}'.format(self.name),point,fontsize=15)
            
            #ax.legend()
            
            if save:
                plt.savefig('{}_in_whole'.format(self.name))
            
            return fig,ax
            
        elif presentation.lower() == 'cut':
            
            xsize = self.cut_sizes[cut-1][0]
            ysize = self.cut_sizes[cut-1][1]
            rat = ysize/xsize
            
            fig = plt.figure(constrained_layout=False, figsize=(6,6))
            ax = plt.subplot(projection=self.cut_wcs)

            ax.set_xlim(0,xsize)
            ax.set_ylim(0,ysize)
            
            ax.plot(ellipse[0],ellipse[1])
            ax.scatter(cut_grb_px[0],cut_grb_px[1])
            
            x,y = ax.coords
            
            x.set_ticks(spacing= .25 * u.hourangle)
            y.set_ticks(spacing= 2 * u.deg)
            
            x.set_ticks_position('bt')
            x.set_ticklabel_position('bt')
            x.set_axislabel_position('bt')
            
            y.set_ticks_position('rl')
            y.set_ticklabel_position('rl')
            y.set_axislabel_position('rl')
            
            x.set_ticklabel(size=12)
            y.set_ticklabel(size=12)
            
            x.set_axislabel('Right Ascension',size=12)
            y.set_axislabel('Declination',size=12)
            
            ax.grid()
                        
            
            if event[1] > 1/2 * xsize:
                if event[0] > 4/5*ysize:
                    point = (event[1] - 200,event[0]-30)
                else:
                    point = (event[1]-200,event[0]+15)
            else:
                if event[0] > 4/5*ysize:
                    point = (event[1],event[0]-30)
                else:
                    point = (event[1],event[0]+15)

            ax.scatter(event[1],event[0],marker='s',color='r',s=16,edgecolor='black')
            ax.annotate('{}'.format(self.name),point,fontsize=15)
            if save:
                plt.savefig('{}_in_cut'.format(self.name))
            
            return fig,ax
            
        else:
            print('Please state "presentation" as "whole/cut".')
            return None,None
        
    def display_event(self,cam,chip,eventcoords=None,save=False):
        
        
        fig,ax = self.find_cut(cam,chip,replot=True,proj=True)
        ax.set_title('')
        
        x,y = ax.coords
        
        x.set_ticks(spacing= 1 * u.hourangle)
        y.set_ticks(spacing= 5 * u.deg)
        
        x.set_ticks_position('bt')
        x.set_ticklabel_position('bt')
        x.set_axislabel_position('bt')
        
        y.set_ticks_position('rl')
        y.set_ticklabel_position('rl')
        y.set_axislabel_position('rl')
        
        x.set_ticklabel(size=12)
        y.set_ticklabel(size=12)
        
        x.set_axislabel('Right Ascension',size=12)
        y.set_axislabel('Declination',size=12)

        xsize = ax.get_xlim()[1]
        ysize = ax.get_ylim()[1]
        
        if eventcoords is not None:
            grbLoc = self.ref_wcs.all_world2pix(eventcoords[0],eventcoords[1],0)
            if grbLoc[0] > 2/3 * xsize:
                if grbLoc[1] > 4/5*ysize:
                    point = (grbLoc[0] - 600,grbLoc[1]-90)
                else:
                    point = (grbLoc[0]-600,grbLoc[1]+30)
            else:
                if grbLoc[1] > 4/5*ysize:
                    point = (grbLoc[0],grbLoc[1]-90)
                else:
                    point = (grbLoc[0],grbLoc[1]+30)
                
    
            ax.scatter(grbLoc[0],grbLoc[1],marker='s',color='r',s=16,edgecolor='black')
            ax.annotate('{}'.format(self.name),point,fontsize=15)
        
        ax.legend()
        
        if save:
            plt.savefig('{}_with_cuts'.format(self.name))
        
        
        return fig,ax           
            
            
            