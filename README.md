# 2D Image Reconstruction with SFCN

## Main Files

- `reconstruct_mnist.py` - MNIST image reconstruction
- `reconstruct_cifar_gray.py` - CIFAR10 grayscale image reconstruction  
- `main.py` - 2D brain MRI slice reconstruction

## Usage

```bash
python <file_name.py>
```
No additional arguments required.

## Configuration

**SFCN Model**: Located in `classificator.py`

- For brain MRI: No changes needed
- For MNIST/CIFAR10: Modify image dimensions and number of classes in `classificator.py`
- Optional: Adjust number of blocks/layers

## Utilities

- `get_brain_2d_data.py` - Generator class for 2D MRI slices (imported by `main.py`)