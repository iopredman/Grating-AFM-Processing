import pySPM
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from pathlib import Path
import copy
import os
from IPython.display import display
import math as m

directory_in_str = str(os.getcwd())
directory = os.fsencode(directory_in_str)

#globals
lengthx = 2300 #<- length of line in nanometers
npillars = 3 #<- number of pillars to look for in scan
pixels = 1024 #<- pixels in a horizontal line
period = 574.7 # <- period of grating in nm
periodpixellength = int(pixels/lengthx*period)
pixeltonmsf = lengthx/pixels

def removestreaks(profile):
    correctedprofile = profile.filter_scars_removal(.7,inline=False)
    return correctedprofile

def pillarlocator(profile, npillars, pillaraccuracy = 0.9): #finds first minimum and then adds approximate period length number of times of expected pillars, firstpillaraccuracy parameters can be adjusted between 0 and 1 to avoid measuring incompleted pillars
    scan1 = list(profile[0:int(pillaraccuracy*periodpixellength)])
    scan1min = min(scan1)
    scan1index = scan1.index(scan1min)
    trenches = []
    trenches.append(scan1index)
    i = 1
    while i < npillars+1:
        nextindex = scan1index + periodpixellength*i
        trenches.append(nextindex)
        i+=1
    return trenches

def peaklocator(profile, trenches): #pass output of pillar locator and overall profile to find maximum values within each region
    pillars = []
    i = 0
    while i < len(trenches)-1:
        pillar = max(profile[trenches[i]:trenches[i+1]])
        pillarindex = list(profile).index(pillar)
        pillars.append(pillarindex)
        i+=1
    return pillars

def trenchlocator(profile, peaks, pillaraccuracy = 0.9): #pass output of peak locator to find true lowest trenches, as opposed to approximation of pillarlocator()
    trenches = []
    trench = min(profile[0:peaks[0]])
    trenches.append(list(profile).index(trench))
    i = 0
    if npillars == 1:
        while i < len(peaks):
            try:
                trench = min(profile[peaks[i]:peaks[i+1]])
                trenches.append(list(profile).index(trench))
                i+=1
            except:
                previousperiod = periodpixellength #this is an explicit change from the generic program necessary for mono pillar measurement
                trench = min(profile[peaks[i]:(peaks[i]+int(previousperiod*pillaraccuracy))])
                trenches.append(list(profile).index(trench))
                break
    else:
        while i < len(peaks):
            try:
                trench = min(profile[peaks[i]:peaks[i+1]])
                trenches.append(list(profile).index(trench))
                i+=1
            except:
                previousperiod = peaks[i] - peaks[i-1]
                trench = min(profile[peaks[i]:(peaks[i]+int(previousperiod*pillaraccuracy))])
                trenches.append(list(profile).index(trench))
                break
    if len(trenches) > npillars + 1:
            raise Exception("Too Many Trenches")
    return trenches

def trenchpillarcombiner(profile):
    peaks = peaklocator(profile, pillarlocator(profile, npillars))
    trenches = trenchlocator(profile, peaklocator(profile, pillarlocator(profile, npillars)))
    combinedlist = peaks + trenches
    combinedlist.sort()
    return combinedlist

def flatten(profile, npillars, flatline1delta = 0, flatline2delta = 0): #<- Flatten through lines drawn through each trench
    flatline1 = 0 + flatline1delta
    flatline2 = len(profile.pixels)-1-flatline2delta
    x1lines = trenchlocator(profile.pixels[flatline1], peaklocator(profile.pixels[flatline1], pillarlocator(profile.pixels[flatline1], npillars)))
    x2lines = trenchlocator(profile.pixels[flatline2], peaklocator(profile.pixels[flatline2], pillarlocator(profile.pixels[flatline2], npillars)))
    lines = []
    i = 0
    while i < len(x1lines):
        line = [x1lines[i],0,x2lines[i],len(profile.pixels)-1]
        lines.append(line)
        i+=1
    correctedprofile = profile.offset(lines)
    return correctedprofile

def fixzero1D(profile): #<- Pass 1D list
    minimum = min(profile)
    difference = 0 - minimum
    profile = list(map(lambda n: n + difference, profile))
    return profile

def fixzero2D(profile): #<- Pass 2D SPM Profile
    minimum = np.min(profile.pixels)
    with np.nditer(profile.pixels, op_flags=['readwrite']) as it:
        for x in it:
            x[...] = x - minimum
    return profile

def averageprofile(profile, outputerror = False): #function to take average profile of lines in an spm
    oneDprofile = profile.mean(axis = 0)
    oneDerror = profile.std(axis = 0)
    if outputerror:
        return (oneDprofile, oneDerror)
    else:
        return oneDprofile


def derivativeprofile(profile, n=3): #function that takes the average profile (one line) and returns the derivative, calculated n points away
    derivativelist = []
    i = n
    while i < len(profile)-n:
        derivative = (profile[i+3] - profile[i-3])/(n*2*pixeltonmsf)
        derivativelist.append(derivative)
        i += 1
    return derivativelist

def wallanglecalc(profile, startingindex, endingindex, bp = 0.10, tp = 0.90):
    pillar10percentvalue = bp * (max(profile[startingindex:(endingindex+1)])-min(profile[startingindex:(endingindex+1)])) + min(profile[startingindex:endingindex+1])
    pillar90percentvalue = tp * (max(profile[startingindex:(endingindex+1)])-min(profile[startingindex:(endingindex+1)])) + min(profile[startingindex:endingindex+1])
    insert = list(profile[startingindex:endingindex]) + [pillar10percentvalue] + [pillar90percentvalue]
    insert.sort()
    sortedprofile = profile[startingindex:endingindex]
    sortedprofile.sort()
    pillar10percentindex = insert.index(pillar10percentvalue) - 1
    pillar90percentindex = insert.index(pillar90percentvalue) - 2
    opposite = abs(sortedprofile[pillar90percentindex] - sortedprofile[pillar10percentindex])
    adjacent = abs(pillar90percentindex - pillar10percentindex)
    angle = 57.2958*abs(m.atan(opposite/(pixeltonmsf*adjacent)))
    return angle

def pillarwidthcalc(profile, startingindex, endingindex, height = 0.10): #Calculates pillar with at height input, default at 10% height, always goes one pixel to right
    widthmeasurepoint = (height * (max(profile[startingindex:endingindex])-(0.5*(profile[startingindex]+profile[endingindex])))) + min(profile[startingindex:endingindex])
    peakcenterindex = profile[startingindex:endingindex].index(max(profile[startingindex:endingindex]))
    firstwall = profile[startingindex:endingindex][:peakcenterindex]
    secondwall = profile[startingindex:endingindex][peakcenterindex:]
    insertfirstwall = firstwall + [widthmeasurepoint]
    insertsecondwall = secondwall + [widthmeasurepoint]
    insertfirstwall.sort()
    insertsecondwall.sort()
    insertsecondwall.reverse()
    firstwallheightindex = insertfirstwall.index(widthmeasurepoint) - 1
    secondwallheightindex = insertsecondwall.index(widthmeasurepoint) - 1
    pillarwidth = (secondwallheightindex+len(firstwall)-firstwallheightindex)*pixeltonmsf
    return pillarwidth


if __name__ == "__main__":
    siteindex = 0
    siteindexes = []
    
    #height outputs
    maxpillarheights = []
    pillarheights = []

    #derivative angle outputs
    pillardvangles = []

    #wall angle calculated at 10% to 90%
    pillarangles = []
    
    #pillarwidth
    pillarwidths = []
    dutycycle = []

    i = 0
    while i<npillars*2:
        pillarheights.append([])
        pillardvangles.append([])
        pillarangles.append([])
        if i % 2 == 0:
            pillarwidths.append([])
        if i % 2 == 0:
            dutycycle.append([])
        i += 1

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".spm"):
            Scan = pySPM.Bruker(filename)  
            topo = Scan.get_channel()
            topoE = copy.deepcopy(topo) #doesn't modify original file, remove to make edits permanent

            #Section to modify 2D Data
            topoE = removestreaks(topoE)
            topoE = flatten(topoE, npillars)
            topoE = fixzero2D(topoE)

            #Section to plot 2D profile, comment out if not using
            fig,ax = plt.subplots()
            ax.imshow(topoE.pixels)
            plt.savefig(f'{filename.split('.')[0]} {filename.split('.')[1]}.png')
            mpl.pyplot.close()

            #Section to modify average profile
            averageprofileoutput = averageprofile(topoE.pixels, outputerror = True)
            averageprofilelist = fixzero1D(averageprofileoutput[0])
            derivativeprofilelist = derivativeprofile(averageprofilelist)

            #Section to plot average profile, comment out if not using
            x = np.linspace(0,len(topoE.pixels[0]),len(topoE.pixels[0]))
            y = list(averageprofilelist)
            fig,ax  = plt.subplots()
            ax.plot(x, y, linewidth=2.0)
            plt.savefig(f'{filename.split('.')[0]} {filename.split('.')[1]} Average Profile.png')
            mpl.pyplot.close()

            #Section to write average profile to excel
            df = pd.DataFrame(
                {'Average Height (nm)': averageprofilelist, 'standard deviation': averageprofileoutput[1]})
            writer = pd.ExcelWriter(f"{filename}.xlsx", engine='xlsxwriter')
            df.to_excel(writer, sheet_name= "average profile", index=False)
            writer._save()

            #Section to calculate important quantities
            l = list(averageprofilelist)
            d = list(map(lambda n: abs(n), derivativeprofilelist))
            maxpillarheights.append(max(l) - min(l))
            importantpoints = trenchpillarcombiner(l)
            p = importantpoints
            
            i = 0
            while i < npillars*2:
                pillarheights[i].append(abs(l[p[i+1]]-l[p[i]]))
                pillardvangles[i].append(57.2958*m.atan(max(d[p[i]:p[i+1]])))
                pillarangles[i].append(wallanglecalc(l,p[i],p[i+1]))
                if i % 2 == 0:
                    pillarwidths[int(i/2)].append(pillarwidthcalc(l, p[i], p[i+2]))
                if i % 2 == 0:
                    dutycycle[int(i/2)].append((pillarwidthcalc(l, p[i], p[i+2], height = 0.5))/period)
                i += 1

            siteindexes.append(siteindex)
            siteindex += 1

    foldername = str(os.getcwd()).split('\\')[-1]
    df = pd.DataFrame({"Sample Index": siteindexes})
    
    df['Max Pillar Height (nm)'] = maxpillarheights
    
    i = 0
    while i < npillars*2:
        if i % 2 == 0:
            side = 'Left'
            pillarn = int(i/2 + 1)
        else:
            side = 'Right'
            pillarn = int((i-1)/2+1)
        df[f'Pillar {pillarn} {side} Height'] = pillarheights[i]
        i += 1

    i = 0
    while i < npillars*2:
        if i % 2 == 0:
            side = 'Left'
            pillarn = int(i/2 + 1)
        else:
            side = 'Right'
            pillarn = int((i-1)/2+1)
        df[f'Pillar {pillarn} {side} Derivative Angle'] = pillardvangles[i]
        i += 1
        
    i = 0
    while i < npillars*2:
        if i % 2 == 0:
            side = 'Left'
            pillarn = int(i/2 + 1)
        else:
            side = 'Right'
            pillarn = int((i-1)/2+1)
        df[f'Pillar {pillarn} {side} Wall Angle'] = pillarangles[i]
        i += 1
        
    i = 0
    while i < npillars*2:
        if i % 2 == 0:
            pillarn = int(i/2 + 1)
            df[f'Pillar {pillarn} Width'] = pillarwidths[int(i/2)]
        i += 1

    i = 0
    while i < npillars*2:
        if i % 2 == 0:
            pillarn = int(i/2 + 1)
            df[f'Pillar {pillarn} Duty Cycle'] = dutycycle[int(i/2)]
        i += 1

    df.style
    writer = pd.ExcelWriter(f"{foldername} Pillar Characterization.xlsx", engine='xlsxwriter')
    df.to_excel(writer, sheet_name= "Pillars", index=False)
    writer._save()