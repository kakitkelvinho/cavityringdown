from loading import Experiment
from analysis.ringdowns import Ringdowns
from analysis.ringdown import Ringdown

def main():
    folder_path = '/home/kelvin/LabInnsbruck/WindowsData/20240715_Ringdown/'
    experiment = Experiment(folder_path)
    print(experiment.lsgroups())

if __name__ == '__main__':
    main()

