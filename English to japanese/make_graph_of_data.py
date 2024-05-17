import pandas as pd
import matplotlib.pyplot as plt

# Path to the log file
log_file_path = 'C://Users//romed//Desktop//English to japanese//graph it french edition.txt'

# Initialize lists to store data
epochs = []
batches = []
losses = []
accuracies = []

# Open and read the log file
with open(log_file_path, 'r') as file:
    for line in file:
        line = line.strip()  # Remove any leading/trailing whitespace
        if line:  # Check if line is not empty
            parts = line.split(',')
            if len(parts) >= 4:  # Ensure there are enough parts in the line
                # Extracting epoch
                epoch_part = parts[0].strip()
                epoch_num = int(epoch_part.split()[1].split('/')[0])

                # Extracting batch
                batch_part = parts[1].strip()
                batch_num = int(batch_part.split()[1].split('/')[0])

                # Extracting loss
                loss_part = parts[2].strip()
                loss_value = float(loss_part.split(':')[-1])

                # Extracting accuracy
                accuracy_part = parts[3].strip()
                accuracy_value = float(accuracy_part.split(':')[-1])

                # Append to lists
                epochs.append(epoch_num)
                batches.append(batch_num)
                losses.append(loss_value)
                accuracies.append(accuracy_value)

# Create a DataFrame from the collected data
df = pd.DataFrame({
    'Epoch': epochs,
    'Batch': batches,
    'Loss': losses,
    'Accuracy': accuracies
})

print(df.head())

# Calculate a 'Batch ID' for plotting (cumulative count of batches)
df['Batch ID'] = range(1, len(df) + 1)

# Plotting
fig, ax1 = plt.subplots(figsize=(12, 6))

# Loss plot
color = 'tab:red'
ax1.set_xlabel('Batch ID')
ax1.set_ylabel('Training Loss', color=color)
ax1.plot(df['Batch ID'], df['Loss'], label='Loss', color=color, marker='o', linestyle='', markersize=2)
ax1.tick_params(axis='y', labelcolor=color)

# Create a second y-axis for the accuracy plot
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Accuracy', color=color)
ax2.plot(df['Batch ID'], df['Accuracy'], label='Accuracy', color=color, marker='o', linestyle='', markersize=2)
ax2.tick_params(axis='y', labelcolor=color)

# Title and show
plt.title('Training Loss and Accuracy for Each Batch')
fig.tight_layout()  # Adjust layout to make room for the second y-axis
plt.show()
