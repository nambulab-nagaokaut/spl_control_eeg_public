"""
Vertical axis: Subject numbers, horizontal axis: Read the CSV file containing data as elements and perform the Friedman test.
"""

from scipy import stats
import csv

buffer = []
with open('./sheet.csv') as f: # This does not support BOM (Byte Order Mark), so please be cautious with the CSV save settings
    reader = csv.reader(f)
    for row in reader:
        buffer.append(row)
print(buffer)

result = stats.friedmanchisquare(*buffer) # Unpack the buffer and pass it.
print(result)