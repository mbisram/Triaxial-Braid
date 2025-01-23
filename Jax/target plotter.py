import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

# Read the CSV file into a DataFrame
csv_name = 'csv output filename here'  
df = pd.read_csv(f'{csv_name}.csv')[:-1]
df['Epoch'] = df['Epoch'].astype(int)

# Define target values for each parameter (for plotting)
target_values = {
    'Vfa': 0.72,
    'Vfb': 0.63,
    'ha': 0.299,
    'hb': 0.234,
    'MAa': 0.510,
    'Loss': 0.
}

for column in df.columns[1:]:  # Skip the 'Epoch' column
    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(df['Epoch'], df[column], label=f'{column}')
    plt.axhline(y=target_values[column], color='r', linestyle='--', label=f'Target {column}')
    plt.xlabel('Iteration', fontsize=14)  
    
    if column in ['ha', 'hb']:
        plt.ylabel(f'{column} (mm)', fontsize=14) 
    else:
        plt.ylabel(column, fontsize=14)  
        
    plt.legend(fontsize=14)  
    
    # Set the number of x-ticks to reduce overcrowding (e.g., every 100 epochs)
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True, prune='both', nbins=10))
    
    plt.show()
    
# Plotting the loss function separately
plt.figure(figsize=(6, 4), dpi=300)
plt.plot(df['Epoch'], df['Loss'], label='Loss')
plt.xlabel('Iteration', fontsize=14) 
plt.ylabel('Loss', fontsize=14) 

# Set x-ticks to display every 100 epochs on the main plot
plt.xticks(range(0, 1001, 100))

# Adding an inset to zoom in on the later epochs
ax_inset = plt.axes([0.5, 0.4, 0.35, 0.35])
ax_inset.plot(df['Epoch'], df['Loss'], color='orange')
ax_inset.set_xlim(df['Epoch'].iloc[-200], df['Epoch'].iloc[-1])  # Zoom into the last 200 epochs
ax_inset.set_ylim(df['Loss'].iloc[-200:].min(), df['Loss'].iloc[-200:].max())
ax_inset.set_xticks(range(df['Epoch'].iloc[-200], df['Epoch'].iloc[-1]+1, 50))  # Fewer ticks for clarity

plt.show()
target1 = np.array([4.5721180e+04, 5.0474207e+04, 1.1229098e+04, 2.7199340e-01, 3.3305216e-01, 3.3305219e-01, 1.7697279e+04, 3.8809763e+03, 4.1473545e+03]) 
target2 = np.array([4.5843020e+04, 5.0072859e+04, 1.1231162e+04, 2.7297780e-01, 3.3309799e-01, 3.3309799e-01, 1.7782066e+04, 4.1784258e+03, 4.1949980e+03])