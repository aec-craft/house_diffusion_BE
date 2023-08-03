import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV files
cosine_data = pd.read_csv('/home/akmal/APIIT/FYP Code/house_diffusion/scripts/ckpts/Cosine/progress.csv')
linear_data = pd.read_csv('/home/akmal/APIIT/FYP Code/house_diffusion/scripts/ckpts/Linear/progress.csv')

# Extract the first 15000 rows and the 'loss' column
rows_to_show = 15000
cosine_loss = cosine_data['loss_q0'].to_numpy()[:rows_to_show]
linear_loss = linear_data['loss_q0'].to_numpy()[:rows_to_show]

x_values = np.arange(1, rows_to_show + 1)

# Plot the line graph for file1_data
plt.plot(x_values, cosine_loss, label='Cosine Loss')

# Plot the line graph for file2_data
# plt.plot(x_values, linear_loss, label='Linear Loss')

# Add labels and legend
plt.xlabel('Row Number')
plt.ylabel('Loss Value')
plt.legend()

# Show the plot
plt.show()
