from pickle import load
from matplotlib.pylab import plt
from numpy import arange
 
# Load the training and validation loss dictionaries
train_loss = load(open('train_loss_bace.pkl', 'rb'))
val_loss = load(open('valid_loss_bace.pkl', 'rb'))
test_loss = load(open('test_loss_bace.pkl', 'rb'))
test_result = load(open('test_result_bace.pkl', 'rb'))
print(test_result)

# Retrieve each dictionary's values
train_values = train_loss.values()
val_values = val_loss.values()
test_values = test_loss.values()
 
# Generate a sequence of integers to represent the epoch numbers
epochs = range(1, len(train_values) + 1)
 
# Plot and label the training and validation loss values
plt.plot(epochs, train_values, label='Training Loss')
plt.plot(epochs, val_values, label='Validation Loss')
plt.plot(epochs, test_values, label='Test Loss')
 
# Add in a title and axes labels
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
 
# Set the tick locations
plt.xticks(arange(0, len(train_values) + 1, 10))
 
# Display the plot
plt.legend(loc='best')
plt.show()