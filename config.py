# dir with ESC50 data
esc50_path = 'data/esc50'

runs_path = 'results'
# sub-epoch (batch-level) progress bar display
disable_bat_pbar = False
#disable_bat_pbar = True

# do not change this block
n_classes = 50
folds = 5
test_folds = [1, 2, 3, 4, 5]
# use only first fold for internal testing
#test_folds = [1]

# sampling rate for waves
sr = 44100
n_mels = 128
hop_length = 512
#n_mfcc = 42

# Use our improved ResNet-style CNN
model_constructor = "AudioCNN(n_mels=config.n_mels, output_size=config.n_classes)"

# ###TRAINING
# ratio to split off from training data
val_size = .2  # could be changed
device_id = 0
#batch_size = 32
batch_size = 64
# in Colab to avoid Warning
#num_workers = 2
#num_workers = 0
num_workers = 4
# for local Windows or Linux machine
# num_workers = 6#16
persistent_workers = True
#persistent_workers = False
epochs = 200
#epochs = 1
# early stopping after epochs with no improvement
#patience = 20
patience = 20  # Increased patience for the more complex model
lr = 5e-4  # Adjusted learning rate
weight_decay = 1e-4
warm_epochs = 10
gamma = 0.9  # Adjusted scheduler gamma
step_size = 10  # Adjusted step size

# Verwenden Sie Mixed Precision Training, um GPU-Speicher zu sparen und Geschwindigkeit zu erhöhen
use_mixed_precision = True

# ### TESTING
# model checkpoints loaded for testing
test_checkpoints = ['terminal.pt']  # ['terminal.pt', 'best_val_loss.pt']
# experiment folder used for testing (result from cross validation training)
#test_experiment = 'results/2025-04-07-00-00'
test_experiment = 'results/sample-run'