import numpy as np
import csv
 
def get_csv(filename: str, index:int=0):
    timetrace = []
    with open(filename, mode='r', encoding='ascii') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        tInc = 0.
        tmp = 0.
        for row in csv_reader:
            if line_count == 0:
                headers = row
                tInc = [header for header in headers if "tInc" in header]
                tInc = float(tInc[0].split('=')[-1].split('s')[0])
                line_count += 1
            else:
                channel = row[index]
                try:
                    float(channel)
                except Exception as e:
                    print(f"{e} with row {line_count}, replace {channel} with {tmp}")
                    channel = tmp
                timetrace.append(float(channel))
                tmp = channel
                line_count += 1
    timetrace = np.array(timetrace)
    return timetrace, tInc
 


