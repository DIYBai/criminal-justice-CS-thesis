import csv

with open('vectorData.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    cols = 29
    writer.writerow( ['0.0']*cols + ['-1.0'] )
    for i in range(cols):
        writer.writerow( ['0.0']*i + ['1.0'] + ['0.0']*(cols - i - 1) + ['-1.0'] )
    writer.writerow( ['1.0']*cols + ['-1.0'] )
    for i in range(cols):
        writer.writerow( ['1.0']*i + ['0.0'] + ['1.0']*(cols - i - 1) + ['-1.0'] )
