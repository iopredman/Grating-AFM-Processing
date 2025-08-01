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
from scipy.optimize import curve_fit

directory_in_str = str(os.getcwd())
directory = os.fsencode(directory_in_str)

#globals
check1 = input('specify globals? (y/n)')
if check1 == 'y':
    lengthx = int(input('Length of image in nm: ') or 1800) #<- length of line in nanometers
    npillars = int(input('Number of full pillars: ') or 2) #<- number of pillars to look for in scan
    pixels = int(input('Pixels per horizontal line: ') or 1024) #<- pixels in a horizontal line
    period = int(input('Period of grating: ') or int(574.7)) # <- period of grating in nm
else:
    lengthx, npillars, pixels, period = 1800, 2, 512, 574.7
check2 = input('specify plot outputs? (y/n)')
if check2 == 'y':
    oneDAcheck = input('Plot 1D Average Line Out? (y/n)')
    oneDEcheck = input('Plot 1D Average Error? (y/n)')
    twoDIcheck = input('Plot 2D image? (y/n)')
    LERcheck = input('Plot Line Edge Roughness? (y/n)')
else:
    oneDAcheck, oneDEcheck, twoDIcheck, LERcheck = 'y','y','y','y'
periodpixellength = int(pixels/lengthx*period)
pixeltonmsf = lengthx/pixels

def removestreaks(profile):
    return profile.filter_scars_removal(.7,inline=False)

def pillarlocator(profile, npillars, pillaraccuracy=0.9):
    start = list(profile[:int(pillaraccuracy * periodpixellength)]).index(min(profile[:int(pillaraccuracy * periodpixellength)]))
    return [start + periodpixellength * i for i in range(npillars + 1)]

def peaklocator(profile, trenches):
    return [list(profile).index(max(profile[trenches[i]:trenches[i+1]])) for i in range(len(trenches) - 1)]

def trenchlocator(profile, peaks, pillaraccuracy=0.9):
    trenches = [list(profile).index(min(profile[:peaks[0]]))]
    for i in range(len(peaks)):
        try:
            trench = min(profile[peaks[i]:peaks[i+1]])
        except:
            period = periodpixellength if npillars == 1 else peaks[i] - peaks[i-1]
            trench = min(profile[peaks[i]:peaks[i] + int(period * pillaraccuracy)])
        trenches.append(list(profile).index(trench))
        if npillars == 1 or i == len(peaks) - 1:
            break
    if len(trenches) > npillars + 1:
        raise Exception("Too Many Trenches")
    return trenches

def trenchpillarcombiner(profile):
    peaks = peaklocator(profile, pillarlocator(profile, npillars))
    trenches = trenchlocator(profile, peaks)
    return sorted(peaks + trenches)

def flatten(profile, npillars, flatline1delta=0, flatline2delta=0):
    flatline1, flatline2 = flatline1delta, len(profile.pixels) - 1 - flatline2delta
    x1 = trenchlocator(profile.pixels[flatline1], peaklocator(profile.pixels[flatline1], pillarlocator(profile.pixels[flatline1], npillars)))
    x2 = trenchlocator(profile.pixels[flatline2], peaklocator(profile.pixels[flatline2], pillarlocator(profile.pixels[flatline2], npillars)))
    lines = [[x1[i], 0, x2[i], len(profile.pixels) - 1] for i in range(len(x1))]
    return profile.offset(lines)

def fixzero1D(profile): #<- Pass 1D SPM Profile
    return [n - min(profile) for n in profile]

def fixzero2D(profile): #<- Pass 2D SPM Profile
    minimum = np.min(profile.pixels)
    with np.nditer(profile.pixels, op_flags=['readwrite']) as it:
        for x in it:
            x[...] = x - minimum
    return profile
    
def averageprofile(profile, outputerror=False):
    avg, err = profile.mean(0), profile.std(0)
    return (avg, err) if outputerror else avg

def derivativeprofile(profile, n=3):
    return [(profile[i+n] - profile[i-n]) / (2 * n * pixeltonmsf) for i in range(n, len(profile) - n)]

def wallanglecalc(profile, start, end, bp=0.10, tp=0.90):
    segment = profile[start:end+1]
    min_val, max_val = min(segment), max(segment)
    p10, p90 = bp * (max_val - min_val) + min_val, tp * (max_val - min_val) + min_val
    sorted_segment = sorted(segment + [p10, p90])
    p10_idx, p90_idx = sorted_segment.index(p10) - 1, sorted_segment.index(p90) - 2
    opp, adj = abs(sorted_segment[p90_idx] - sorted_segment[p10_idx]), abs(p90_idx - p10_idx)
    return 57.2958 * abs(m.atan(opp / (pixeltonmsf * adj)))

def pillarwidthcalc(profile, start, end, height=0.10):
    segment = profile[start:end]
    peak = max(segment)
    widthpoint = height * (peak - 0.5 * (profile[start] + profile[end])) + min(segment)
    peak_idx = segment.index(peak)
    firstwall, secondwall = segment[:peak_idx], segment[peak_idx:]
    firstwallheight = sorted(firstwall + [widthpoint]).index(widthpoint) - 1
    secondwallheight = len(secondwall) - sorted(secondwall + [widthpoint], reverse=True).index(widthpoint) - 1
    return (secondwallheight + len(firstwall) - firstwallheight) * pixeltonmsf

def assign_pillar_data(data, column_suffix, step=1):
    for i in range(0, npillars * 2, step):
        side = 'Left' if i % 2 == 0 else 'Right'
        pillarn = i // 2 + 1
        df[f'Pillar {pillarn} {side} {column_suffix}'] = data[i // step]

def sigmoid(x,a,b,c,x0):
    y = a + (b-a)/(1+np.exp(-c*(x-x0)))
    return y

def line_edge(profile):
    x0s = []
    for badrow, row in enumerate(profile):
        rowx0s = []
        pat = trenchpillarcombiner(row)
        for i in range(len(pat)-1):
            segment = row[pat[i]:pat[i+1]]
            guesses = [
                [max(segment), min(segment),0.1,pat[i]+(pat[i+1]-pat[i])/2],
                [min(segment), max(segment),0.1,pat[i]+(pat[i+1]-pat[i])/2],
                [min(segment), max(segment),0.1,pat[i]+(pat[i+1]-pat[i])/1.5],
                [max(segment), min(segment),0.1,pat[i]+(pat[i+1]-pat[i])/1.5]
                ]
            bounds = (
                [-1000, -1000, 0, 0],
                [1000, 1000, 1, len(row)]
            )
            for guess in guesses:
                try:
                    popt, pcov = curve_fit(sigmoid, list(range(pat[i],pat[i+1])), segment, guess, bounds=bounds)
                    break
                except:
                    if guess == guesses[-1]:
                        x = range(len(segment))
                        y = segment
                        y2 = sigmoid(list(range(pat[i],pat[i+1])), *popt)
                        plt.plot(x, y)
                        plt.plot(x, y2)
                        plt.savefig(f'Failed fit at {filename.split('.')[0]} {filename.split('.')[1]} row {badrow}')
                        mpl.pyplot.close()
                        # raise Exception(f'Failed at {filename}, row {row}')
                    pass
            rowx0s.append(popt[3])
        x0s.append(rowx0s)
    return x0s

if __name__ == "__main__":
    filestobeprocessed, siteindex, siteindexes, pillarheights, pillardvangles, pillarangles, pillarwidths, dutycycle, error, LERs = 0, 0, [], [], [], [], [], [], [], []

    for i in range(npillars * 2):
        pillarheights.append([])
        pillardvangles.append([])
        pillarangles.append([])
        LERs.append([])
        if i % 2 == 0:
            pillarwidths.append([])
            dutycycle.append([])

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".spm"):
            filestobeprocessed += 1
    
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".spm"):
            Scan = pySPM.Bruker(filename)  
            topo = Scan.get_channel()
            topoE = copy.deepcopy(topo) #doesn't modify original file, remove to make edits permanent

            #modify 2D Data
            topoE = removestreaks(topoE)
            topoE = flatten(topoE, npillars)
            topoE = fixzero2D(topoE)

            #plot 2D profile
            if twoDIcheck == 'y':
                fig,ax = plt.subplots()
                if LERcheck == 'y':
                    # x1 = []
                    # x2 = []
                    # for rowx0s in line_edge(topoE.pixels):
                    #     x1.append(rowx0s[0])
                    #     x2.append(rowx0s[1])
                    # y = range(topoE.pixels.shape[0])
                    # fig,ax  = plt.subplots()
                    # ax.plot(x1, y, linewidth=1.5, color='red')
                    # ax.plot(x2, y, linewidth=1.5, color='red')

                    # x_values = []  # List to store all x values dynamically
                    # for rowx0s in line_edge(topoE.pixels):
                    #     # Ensure there are enough sublists in x_values to accommodate all x values in rowx0s
                    #     while len(x_values) < len(rowx0s):
                    #         x_values.append([])  # Add a new empty list for each new x column
                    #     # Append each x value to the corresponding sublist
                    #     for i, x in enumerate(rowx0s):
                    #         x_values[i].append(x)

                    # y = range(topoE.pixels.shape[0])  # y-axis values
                    # fig, ax = plt.subplots()

                    # # Plot each x series against y
                    # for x in x_values:
                    #     ax.plot(x, y, linewidth=1.5, color='red')    
                    x_values = []
                    for element in range(len(line_edge(topoE.pixels)[0])):
                        x_values.append([])
                    for rowx0s in line_edge(topoE.pixels):
                        for i, element in enumerate(rowx0s):
                            x_values[i].append(element)
                    y = range(topoE.pixels.shape[0])
                    fig,ax  = plt.subplots()
                    for x in x_values:
                        ax.plot(x, y, linewidth=1.5, color='red')
                ax.imshow(topoE.pixels)
                plt.savefig(f'{filename.split('.')[0]} {filename.split('.')[1]}.png')
                mpl.pyplot.close()

            #plot 2D line edge
            if LERcheck == 'y':
                x1 = []
                x2 = []
                for rowx0s in line_edge(topoE.pixels):
                    x1.append(rowx0s[0])
                    x2.append(rowx0s[1])
                y = range(topoE.pixels.shape[0])
                fig,ax  = plt.subplots()
                ax.plot(x1, y, linewidth=2.0)
                ax.plot(x2, y, linewidth=2.0)
                plt.savefig(f'{filename.split('.')[0]} {filename.split('.')[1]} Line Edge Profile.png')
                mpl.pyplot.close()

            #modify 1D average profile
            averageprofileoutput = averageprofile(topoE.pixels, outputerror = True)
            averageprofilelist = fixzero1D(averageprofileoutput[0])
            averageprofileerror = averageprofileoutput[1]
            derivativeprofilelist = derivativeprofile(averageprofilelist)

            #Section to plot 1D average profile
            if oneDAcheck == 'y':
                x = np.linspace(0,len(topoE.pixels[0]),len(topoE.pixels[0]))
                y = list(averageprofilelist)
                fig,ax  = plt.subplots()
                ax.plot(x, y, linewidth=2.0)
                plt.savefig(f'{filename.split('.')[0]} {filename.split('.')[1]} Average Profile.png')
                mpl.pyplot.close()

            #Section to plot 1D error profile
            if oneDEcheck == 'y':
                x = np.linspace(0,len(topoE.pixels[0]),len(topoE.pixels[0]))
                y = list(averageprofileerror)
                fig,ax  = plt.subplots()
                ax.plot(x, y, linewidth=2.0)
                plt.savefig(f'{filename.split('.')[0]} {filename.split('.')[1]} Average Profile Error.png')
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
            importantpoints = trenchpillarcombiner(l)
            p = importantpoints

            for i in range(npillars * 2):
                pillarheights[i].append(abs(l[p[i + 1]] - l[p[i]]))
                pillardvangles[i].append(57.2958 * m.atan(max(d[p[i]:p[i + 1]])))
                pillarangles[i].append(wallanglecalc(l, p[i], p[i + 1]))
                LERs[i].append(3*np.std(x_values[i]))
                if i % 2 == 0:
                    width = pillarwidthcalc(l, p[i], p[i + 2])
                    pillarwidths[i // 2].append(width)
                    dutycycle[i // 2].append(width / period)

            error.append(np.mean(averageprofileerror))
            siteindexes.append(siteindex)
            print(f'Processed Files: {siteindex+1}/{filestobeprocessed}')
            siteindex += 1

    foldername = str(os.getcwd()).split('\\')[-1]
    df = pd.DataFrame({"Sample Index": siteindexes})
    assign_pillar_data(pillarheights, 'Height')
    assign_pillar_data(pillardvangles, 'Derivative Angle')
    assign_pillar_data(pillarangles, 'Wall Angle')
    assign_pillar_data(LERs, 'LER')
    assign_pillar_data(pillarwidths, 'Width', step=2)
    assign_pillar_data(dutycycle, 'Duty Cycle', step=2)
    df['Average Error'] = error
    df.style
    writer = pd.ExcelWriter(f"{foldername} Pillar Characterization.xlsx", engine='xlsxwriter')
    df.to_excel(writer, sheet_name= "Pillars", index=False)
    writer._save()

    print('Processing Completed')
    input('Press enter to close')