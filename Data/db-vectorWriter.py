import csv

with open('vectorData.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    cols = 40
    for i in range(cols):
        writer.writerow( ['-1']*3 + ['0']*i + ['1'] + ['0']*(cols - i - 1) + ['-1'] )
    for i in range(cols):
        writer.writerow( ['-1']*3 + ['1']*i + ['0'] + ['1']*(cols - i - 1) + ['-1'] )
