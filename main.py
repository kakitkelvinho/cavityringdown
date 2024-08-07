from analysis.loading import Experiment

def main():
    folder_path = '/home/kelvin/LabInnsbruck/WindowsData/20240715_Ringdown/'
    experiment = Experiment(folder_path)
    print(experiment.lsgroups())
    ringdowns = experiment.loadgroup('PA_50')


if __name__ == '__main__':
    main()

