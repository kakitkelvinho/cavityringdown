import numpy as np
from .ringdown import Ringdown
from .ringdowns import Ringdowns
import csv
import os


class Experiment:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def lsgroups(self, match='PA'):
        groupnames = [f for f in os.listdir(self.folder_path) if match in f]
        return groupnames

    def loadgroup(self, groupname):
        if groupname not in self.lsgroups():
            return ValueError("Invalid group name, use method lsgroups to see valid names")
        filenames = [f for f in os.listdir(os.path.join(self.folder_path, groupname)) if f.endswith('.csv')]
        ringdowns = Ringdowns(name=groupname)
        for file in filenames:
            subpath = os.path.join(groupname, file)
            fullpath = os.path.join(self.folder_path, subpath)
            t, timetrace = get_csv(fullpath)
            ringdown = Ringdown(t=t, timetrace=timetrace)
            ringdowns.append(ringdown)
        return ringdowns
            
            

        



def get_csv(filename: str, index:int=0, verbose=False):
    '''Given the filename (full file path) return from the csv an array of time
    and the timetrace
    filename (str): full filepath to csv
    index (int): which channel, by default 0 as that is the first channel
    verbose (bool): determines whether to output error messages
    '''
    timetrace = []
    with open(filename, mode='r', encoding='ascii') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        tInc = 0.
        tmp = 0.
        for row in csv_reader:
            if line_count == 0:
                headers = row
                # get tInc
                tInc = [header for header in headers if "tInc" in header]
                tInc = float(tInc[0].split('=')[-1].split('s')[0])
                line_count += 1
            else:
                channel = row[index]
                try:
                    float(channel)
                except Exception as e:
                    if verbose:
                        print(f"{e} with row {line_count}:\n value: {channel}")
                        print("This file is corrupted. Try another")
                    pass
                timetrace.append(float(channel))
                line_count += 1
    timetrace = np.array(timetrace)
    t = np.arange(0, len(timetrace)*tInc, tInc)
    return t, timetrace
 
def main():
    folder_path = '/home/kelvin/LabInnsbruck/WindowsData/20240715_Ringdown/'
    experiment = Experiment(folder_path)
    print(experiment.lsgroups())



