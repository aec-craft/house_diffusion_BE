import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

# Read the CSV file into a list
with open('/home/akmal/cosine.csv', 'r') as file:
    data = [float(line.strip()) for line in file]

# Generate x-axis values from 1 to 1000
data = [0.02 - x for x in data]
x_values = range(1, 1001)

# Create the line graph
plt.plot(x_values, data, linestyle='-')
plt.xlabel('X-axis')
plt.ylabel('Values')
plt.title('Line Graph')
plt.grid(True)
plt.show()
