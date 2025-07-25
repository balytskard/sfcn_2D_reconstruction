import os
import numpy as np
import pandas as pd
import random
import time

from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import inversefed
from get_brain_2d_data import DataGenerator
from classificator import SFCN, validate, train_one_epoch


start = time.time()

setup = inversefed.utils.system_startup()
device, dtype = setup["device"], setup["dtype"]

def _get_meanstd(dataset):
    channel = dataset[0][0].shape[0]
    cc = torch.cat([dataset[i][0].reshape(channel, -1) for i in range(len(dataset))], dim=1)
    data_mean = torch.mean(cc, dim=1)
    data_std = torch.std(cc, dim=1)

    mean_tensor = torch.as_tensor([data_mean], **setup)
    std_tensor = torch.as_tensor([data_std], **setup)
    
    return mean_tensor, std_tensor


# Set seeds for reproducibility
torch.manual_seed(1)
random.seed(1)
np.random.seed(1)

trained_model = True
CUDA_LAUNCH_BLOCKING=1


params = {
    "batch_size": 5,
    "imagex": 160,
    "imagey": 192,
    "imagez": 160,
    "column": "Group_bin",
}

csv_dir = 'pd_data_csv'
nifti_dir = '/work/forkert_lab/mitacs_dataset/affine_using_nifty_reg'
num_images = 1

image_path = 'reconstruction_result_brain2d'
ground_truth_name = 'brain2d_ground_truth.png'
reconstruction_name = 'test.png'
os.makedirs(image_path, exist_ok=True)

train_csv = os.path.join(csv_dir, "train_pd_complete_adni.csv")
val_csv = os.path.join(csv_dir, "test_pd_complete_adni.csv")

# Create data loaders
train = pd.read_csv(train_csv)
val = pd.read_csv(val_csv)

train_IDs = train['Subject'].to_numpy()
training_dataset = DataGenerator(train_IDs, (params['imagex'], params['imagey'], params['imagez']), train_csv, params['column'])
trainloader = DataLoader(training_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=1)

val_IDs = val['Subject'].to_numpy()
val_dataset = DataGenerator(val_IDs, (params['imagex'], params['imagey'], params['imagez']), val_csv, params['column'])
validloader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=1)

# Mean and standard deviation were computed earlier
dm = [-0.5934]
ds = [0.3767]

dm = torch.as_tensor([dm], **setup)
ds = torch.as_tensor([ds], **setup)
print(f"[INFO] mean={dm}, std={ds}")


# Create and train the model
print("Creating the model...")
model = SFCN().to(device=device, dtype=dtype)
# model.load_state_dict(torch.load("sfcn_full_brain2d.pt"))
criterion = inversefed.data.loss.Classification()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=np.exp(-0.1))

# Training loop
best_val_loss = float('inf')
patience = 10
patience_counter = 0
num_epochs = 1

print(f"Starting training for {num_epochs} epochs...")
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, validloader, criterion, device)
    scheduler.step()
    print("Epoch {}: Train Loss={:.4f}, Train Acc={:.4f}, Val Loss={:.4f}, Val Acc={:.4f}".format(
        epoch+1, train_loss, train_acc, val_loss, val_acc))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "test.pt")
        print("Model saved at epoch {}".format(epoch+1))
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

print("Training completed!")

ground_truth = torchvision.io.read_image(os.path.join(image_path, ground_truth_name)).float() / 255.0
ground_truth = ground_truth.to(device=device, dtype=dtype)
ground_truth = ground_truth.unsqueeze(0)
labels = torch.as_tensor([0], device=setup["device"], dtype=torch.long)
loss_fn = inversefed.data.loss.Classification()

model.eval()
model.zero_grad()
model_output = model(ground_truth)
ground_truth.requires_grad_(True)

target_loss, loss_name, loss_format = loss_fn(model_output, labels)
print(f"Target loss computed successfully: {target_loss}")

input_gradient = torch.autograd.grad(
    outputs=target_loss, 
    inputs=model.parameters(), 
    create_graph=False,
    retain_graph=False,
    allow_unused=True
)

# Filter out None gradients and replace with zero tensors
filtered_gradients = []
for i, (grad, param) in enumerate(zip(input_gradient, model.parameters())):
    if grad is not None:
        filtered_gradients.append(grad.detach().clone())

input_gradient = filtered_gradients


# Gradient inversion configuration
config = dict(
    signed=False,
    boxed=True,
    cost_fn='sim',
    indices='def',
    weights='equal', 
    lr=0.0005, 
    optim='adam',
    restarts=5,
    max_iterations=40,
    total_variation=1e-5,
    init='randn',                
    filter='none',
    lr_decay=True,
    scoring_choice='loss'
)

# Perform gradient inversion
rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=1)
output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=(1, params['imagex'], params['imagez']))


# Save the reconstructed image
output = torch.as_tensor(output, device=device, dtype=dtype)
output = output.mul(ds).add(dm)
torchvision.utils.save_image(output, os.path.join(image_path, reconstruction_name))
print(f"Reconstruction saved")


# Evaluate reconstruction quality
output_np = output.detach().cpu().numpy().squeeze() 
ground_truth_np = ground_truth.detach().cpu().numpy().squeeze()

test_mse = (output.detach() - ground_truth).pow(2).mean()
feat_mse = (model(output.detach()) - model(ground_truth)).pow(2).mean()  
test_psnr = psnr(output_np, ground_truth_np, data_range=1.0)
test_ssim = ssim(output_np, ground_truth_np, data_range=1.0)

print(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} "
          f"| PSNR: {test_psnr:4.2f} | SSIM: {test_ssim:2.4f}")


end = time.time()
print(f"Total time taken: {(end - start)/60:.2f} minutes")