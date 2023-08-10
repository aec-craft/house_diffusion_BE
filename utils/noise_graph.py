import pandas as pd
import matplotlib.pyplot as plt
import torch
import math
import numpy as np


def get_alpha(file_name):
    with open(f'/home/akmal/{file_name}.csv', 'r') as file:
        data = [float(line.strip()) for line in file]

    data = [1 - x for x in data]
    data = torch.cumprod(torch.tensor(data), axis=0)

    return data


cosine = get_alpha("cosine")
linear = get_alpha("linear")
sqrt = get_alpha("sqrt")

x_values = range(1, 1001)

# Create the line graph
# plt.plot(x_values, cosine, linestyle='-', color="red", label="cosine")
# plt.plot(x_values, linear, linestyle='-', color="blue", label="linear")
plt.plot(x_values, sqrt, linestyle='-', color="green", label="sqrt")

plt.legend()
plt.xlabel('X-axis')
plt.ylabel('Alpha-hat')
plt.title('Sqrt')
plt.show()


