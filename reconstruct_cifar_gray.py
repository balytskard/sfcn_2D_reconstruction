import os
import random
import time
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

from classificator import SFCN, validate, train_one_epoch   # DO NOT FORGET TO CUT SFCN MODEL FOR COMPATIBILITY
import inversefed


start = time.time()

setup = inversefed.utils.system_startup()
device, dtype = setup["device"], setup["dtype"]

torch.manual_seed(1)
random.seed(1)
np.random.seed(1)

trained_model = True
CUDA_LAUNCH_BLOCKING=1


# ================= PARAMETERS FOR CUSTOMIZATION =================
# CIAFR10 images dimensions
params = {
    "imagex": 32,
    "imagey": 32
}

image_path = 'reconstruction_result_cifar10_grayscale'
os.makedirs(image_path, exist_ok=True)
reconstructed_img_name = 'test.png'
ground_truth_name = 'cifar10_ground_truth.png'

model_path = 'models_by_blocks_cifar10'
os.makedirs(model_path, exist_ok=True)
new_model_name = 'training_test.pt'
pretrained_model_name = 'model_cifar10_gray_1block.pt'

num_images = 1
# =================================================================


defs = inversefed.training_strategy('conservative')
loss_fn, trainloader, validloader, dm, ds =  inversefed.construct_dataloaders('CIFAR10', defs)

# Get random image from the training set
idx = random.randint(0, len(trainloader) - 1)
img_tensor, label = trainloader.dataset[idx]
print(img_tensor.shape)
if isinstance(img_tensor, torch.Tensor):
    img = img_tensor.clone()
    if img.shape[0] == 1:
        img = img * ds[0] + dm[0]
        img = torch.clamp(img, 0, 1)
        img_pil = transforms.ToPILImage()(img)
    else:
        img_pil = transforms.ToPILImage()(img)
else:
    img_pil = img_tensor

# Save random image image as ground truth
img_pil.save(os.path.join(image_path, ground_truth_name))
print(f"Image saved as {os.path.join(image_path, ground_truth_name)}")


dm = torch.as_tensor([dm], **setup)
ds = torch.as_tensor([ds], **setup)
print(f"[INFO] mean={dm}, std={ds}")


# Create and train the model
print("Creating the model...")
model = SFCN().to(device=device, dtype=dtype)

# ================== LOADING MODEL WEIGHTS ==================
# Comment if you retrain model and do not need to load weights
# model = torch.jit.load(os.path.join(model_path, pretrained_model_name))


criterion = inversefed.data.loss.Classification()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=np.exp(-0.1))

# ================== TRAINING PROCESS STARTS ==================
# Uncomment if you need to train a new model
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
        torch.save(model.state_dict(), os.path.join(model_path, new_model_name))
        print("Model saved at epoch {}".format(epoch+1))
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

print("Training completed!")
# ================== TRAINING PROCESS ENDS ==================


ground_truth = torchvision.io.read_image(os.path.join(image_path, ground_truth_name)).float() / 255.0
ground_truth = ground_truth.to(device=device, dtype=dtype)
ground_truth = ground_truth.unsqueeze(0)

labels = torch.as_tensor([label], device=setup["device"], dtype=torch.long)

loss_fn = inversefed.data.loss.Classification()

model.eval()
model.zero_grad()

# Make sure the input requires gradients for backpropagation
ground_truth.requires_grad_(True)

# Forward pass
model_output = model(ground_truth)

target_loss, loss_name, loss_format = loss_fn(model_output, labels)


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
full_norm = torch.stack([g.norm() for g in input_gradient]).mean()


# Gradient inversion configuration
config = dict(
    signed=False,
    boxed=True,
    cost_fn='sim',
    indices='def',
    weights='equal', 
    lr=0.003, 
    optim='adam',
    restarts=1,
    max_iterations=400,
    total_variation=1e-5,
    init='randn',                
    filter='none',
    lr_decay=True,
    scoring_choice='loss'
)


# Perform gradient inversion
rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=num_images)
output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=(1, params['imagex'], params['imagey']))


# Save the reconstructed image
output = torch.as_tensor(output , device=device, dtype=dtype)
torchvision.utils.save_image(output, os.path.join(image_path, reconstructed_img_name))
print(f"Reconstruction saved")


# Evaluate reconstruction quality
output_np = output.detach().cpu().numpy().squeeze() 
ground_truth_np = ground_truth.detach().cpu().numpy().squeeze()

test_mse = (output.detach() - ground_truth).pow(2).mean()
feat_mse = (model(output.detach()) - model(ground_truth)).pow(2).mean()  
test_psnr = psnr(output_np, ground_truth_np, data_range=1.0)
test_ssim = ssim(output_np, ground_truth_np, data_range=1.0)


print(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} "
          f"| PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} |")


end = time.time()
print(f"Total time taken: {(end - start)/60:.2f} minutes")