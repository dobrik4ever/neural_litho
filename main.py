import torch
import torch.nn as nn
from datetime import datetime
import shutil
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
import json
import torch.nn.functional as F
from skimage import transform


device_name = 'mps' # cuda or cpu or mps
device = torch.device(f'{device_name}:0')

# === Parameters === #

# == Environment == #
INPLACE = False  # If True, the current run will not be saved in a separate folder but instead will overwrite the 'inplace' run
NOW = datetime.now().strftime('%Y.%m.%d_%H:%M:%S')  # ID for the current run
json_dict = {'time': NOW}
NOW = 'inplace' if INPLACE else NOW
# Folder where the experiments will be stored
folder_with_experiments = os.path.join('model_runs', f'run_{NOW}')
json_dict['folder'] = folder_with_experiments
json_fname = os.path.join(folder_with_experiments, 'results.json')
folder_training_data = '/Users/dobrik4ever/Library/CloudStorage/OneDrive-Personal/Documents/Pythonscripts/LERBA/Neural/Neural_Lithography/data/printed_data'  # Folder with training data
folder_training_input = os.path.join(
    folder_training_data, 'mask')  # Folder with training input
folder_training_output = os.path.join(
    folder_training_data, 'afm')  # Folder with training output

# == Train and Optimization parameters == #
resist_thickness = 30.0

USE_PRETRAINED_MODEL = True

if USE_PRETRAINED_MODEL:
    pretrained_model_fname = 'model_runs/run_2024.07.29_18:23:15/model.pt'
    if not os.path.exists(pretrained_model_fname):
        USE_PRETRAINED_MODEL = False
        raise RuntimeError(f'Pretrained model {pretrained_model_fname} not found')
else:
    pretrained_model_fname = None


class Parameters:

    @classmethod
    def to_dict(cls):
        output = {}
        for key, val in cls.__dict__.items():
            if key.startswith("__"):
                continue
            if isinstance(val, (int, float, str)):
                output[key] = val
            elif callable(val):
                output[key] = repr(val)
        return output


class TrainingParameters(Parameters):
    plotting_interval = 1
    batch_size = 10
    split_percent = 0.1  # How many percent of the data will be used for testing
    shuffle_dataset = False
    learning_rate = 1e-4
    num_epochs = 1000
    early_stopping_patience = 5
    optimizer = torch.optim.Adam
    loss_function = torch.nn.MSELoss()


class OptimizationParameters(Parameters):
    target_mask_fname = '../Neural_Lithography/data/printed_data/mask/data_0.npy' #'Mask_target.npy'
    target_dart_fname = '../Neural_Lithography/data/printed_data/afm/data_0.npy' #'Dart_target.npy'
    plotting_interval = 100
    learning_rate = 1
    num_epochs = 10000
    optimizer = torch.optim.SGD
    loss_function = torch.nn.MSELoss()


json_dict['training'] = {'parameters': TrainingParameters.to_dict()}
json_dict['optimization'] = {'parameters': OptimizationParameters.to_dict()}
# === Utilities === #

if not os.path.exists(folder_with_experiments):  # Make sure the folder exists
    os.mkdir(folder_with_experiments)

shutil.copyfile('main.py', os.path.join(folder_with_experiments,
                f'main_{NOW}.py'))  # Copy file to the run folder

with open(os.path.join(folder_with_experiments, 'results.json'), 'w') as f:
    json.dump(json_dict, f, indent=4)

def detensor(tensor):
    return tensor.cpu().detach().numpy()[0, 0]


def plot_training(val_loader, model, train_losses, val_losses, folder: str):
    with torch.no_grad():
        mask_gt = val_loader.dataset[0][0][None, ...]
        mask_gt_tensor = torch.tensor(
            mask_gt, device=device, dtype=torch.float32)
        dart_pred = detensor(model.forward(mask_gt_tensor))
        mask_gt = mask_gt[0][0]
        dart_gt = val_loader.dataset[0][1][0]
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    ax_plot = ax[1, 0]
    ax_plot.plot(train_losses, label='train')
    ax_plot.plot(val_losses, label='validation')
    ax_plot.legend()
    ax_plot.set_yscale('log')
    ax_plot.set_title('Losses')
    ax_plot.set_xlabel('Epoch')
    ax_plot.set_ylabel('Loss')
    ax_plot.grid()

    ax_mask_gt = ax[0, 0]
    fig.colorbar(ax_mask_gt.imshow(mask_gt, cmap='gray'), ax=ax_mask_gt)
    ax_mask_gt.set_title('Mask Dr.Litho')

    ax_dart_gt = ax[0, 1]
    fig.colorbar(ax_dart_gt.imshow(dart_gt), ax=ax_dart_gt)
    ax_dart_gt.set_title('DART Dr.Litho')

    ax_dart_pred = ax[1, 1]
    fig.colorbar(ax_dart_pred.imshow(dart_pred), ax=ax_dart_pred)
    ax_dart_pred.set_title('DART Neural Litho')

    fig.savefig(os.path.join(folder, f'training.png'))
    plt.close('all')


def plot_optimization(mask_pred, dart_pred, dart_sim, mask_sim, array_losses, folder: str):
    dart_pred = detensor(dart_pred)
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    # First row plots
    im0 = ax[0, 0].imshow(mask_sim, cmap='gray')
    ax[0, 0].set_title('Mask target')
    fig.colorbar(im0, ax=ax[0, 0])

    diff = np.abs(dart_sim - dart_pred)
    ind = np.unravel_index(np.argmax(diff), diff.shape)

    mask_pred = detensor(mask_pred)
    im1 = ax[0, 1].imshow(mask_pred, cmap='gray')
    ax[0, 1].axhline(ind[0], color='r', linestyle='--')
    ax[0, 1].set_title('Mask predicted')
    fig.colorbar(im1, ax=ax[0, 1])

    ax[0, 2].plot(array_losses)
    ax[0, 2].set_yscale('log')
    ax[0, 2].grid()
    ax[0, 2].set_title(f'loss {array_losses[-1]:.4f}')

    # Second row plots
    im2 = ax[1, 0].imshow(dart_sim, cmap='plasma')
    ax[1, 0].set_title('DART target')
    fig.colorbar(im2, ax=ax[1, 0])

    im3 = ax[1, 1].imshow(dart_pred, cmap='plasma')
    ax[1, 1].axhline(ind[0], color='r', linestyle='--')
    ax[1, 1].set_title('DART predicted')
    fig.colorbar(im3, ax=ax[1, 1])

    ax[1, 2].plot(dart_sim[ind[0]], label='DART target')
    ax[1, 2].plot(dart_pred[ind[0]], label='DART predicted')
    ax[1, 2].set_ylim([0, dart_sim.max() + dart_sim.max() * 0.1])
    ax[1, 2].set_title(f'DART difference {diff.max():.4f}')
    ax[1, 2].legend()

    fig.savefig(os.path.join(folder, f'optimization.png'))
    plt.close()

# === Dataset === #


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_folder: str, target_folder: str):
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.mask_files = [os.path.join(input_folder, fname)
                           for fname in os.listdir(input_folder)]
        self.dart_files = [os.path.join(target_folder, fname)
                           for fname in os.listdir(target_folder)]
        assert len(self.mask_files) == len(self.dart_files)
        self.__len = len(self.mask_files)

    def __len__(self):
        return self.__len

    def __getitem__(self, idx):
        mask_fname = self.mask_files[idx]
        dart_fname = self.dart_files[idx]

        assert os.path.basename(mask_fname) == os.path.basename(dart_fname)
        N = 104
        mask = transform.resize(np.load(mask_fname), [N,N]).astype(np.float32)[np.newaxis, ...]
        dart = transform.resize(np.load(dart_fname), [N,N]).astype(np.float32)[np.newaxis, ...] #/ resist_thickness # Resist thickness

        return mask, dart


def create_dataloaders(dataset, split_percent=0.1, bs=10, shuffle=False):
    test_size = int(len(dataset) * split_percent)
    train_size = len(dataset) - test_size
    test_dataset, train_dataset = torch.utils.data.random_split(
        dataset, [test_size, train_size])

    def to_device(x): return tuple(x_.to(device)
                                   for x_ in torch.utils.data.dataloader.default_collate(x))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bs, shuffle=shuffle, collate_fn=to_device)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=bs, shuffle=shuffle, collate_fn=to_device)
    return train_dataloader, test_dataloader

# === Model === #


class NeuralPointwiseNet(nn.Module):
    """ kernel_size=1
    """

    def __init__(self, ch=64, leak=0.2):
        super(NeuralPointwiseNet, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=2*ch, kernel_size=1)
        self.conv1 = nn.Conv2d(
            in_channels=2*ch, out_channels=ch, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=ch, out_channels=1, kernel_size=1)
        self.lrelu0 = nn.LeakyReLU(leak)
        self.lrelu1 = nn.LeakyReLU(leak)

    def forward(self, x):
        x = self.conv0(x)
        x = self.lrelu0(x)
        x = self.conv1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        return x


class NeuralAreawiseNet(nn.Module):
    """ kernel_size=3
    """

    def __init__(self, ch=32, leak=0.2, kernel_size=3, padding=1):
        super(NeuralAreawiseNet, self).__init__()

        self.conv0 = nn.Conv2d(
            in_channels=1, out_channels=2*ch, kernel_size=kernel_size, padding=padding)
        self.conv1 = nn.Conv2d(
            in_channels=2*ch, out_channels=2*ch, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(
            in_channels=2*ch, out_channels=ch, kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(
            in_channels=ch, out_channels=ch, kernel_size=kernel_size, padding=padding)
        self.conv4 = nn.Conv2d(
            in_channels=ch, out_channels=1, kernel_size=kernel_size, padding=padding)

        self.lrelu0 = nn.LeakyReLU(leak)
        self.lrelu1 = nn.LeakyReLU(leak)
        self.lrelu2 = nn.LeakyReLU(leak)
        self.lrelu3 = nn.LeakyReLU(leak)

    def forward(self, x):

        x = self.conv0(x)
        x = self.lrelu0(x)
        x = self.conv1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.lrelu2(x)
        x = self.conv3(x)
        x = self.lrelu3(x)
        x = self.conv4(x)

        return x


# class NeuralLitho(nn.Module):

#     def __init__(self):
#         super().__init__()
#         self.point1 = NeuralPointwiseNet()
#         self.area1 = NeuralAreawiseNet()
#         self.sigmac_range = 0.25
#         self.sigmac_param = nn.Parameter(torch.tensor(0.2))
#         self.shrink_approx = NeuralPointwiseNet()
#         self.out_layer = NeuralAreawiseNet()

#     def create_gaussian_kernel(self, sigma):
#         stepxy = int(sigma * 6 + 1)
#         rangexy = stepxy // 2
#         xycord = torch.linspace(-rangexy, rangexy, steps=stepxy).to(device)
#         kernel = torch.exp(-(xycord[:, None]**2 + xycord[None, :]**2) / (2 * sigma**2))
#         kernel = kernel / torch.sum(kernel)
#         return kernel[None, None]

#     def get_aerial_image(self, masks):
#         illum_kernel = self.create_gaussian_kernel(0.2)  # Fixed sigmao
#         aerial_image = conv2d(masks, illum_kernel, intensity_output=True)
#         return aerial_image

#     def get_resist_image(self, aerial_image):
#         # Thresholding
#         exposure = self.point1(aerial_image)
#         sigmac = torch.sigmoid(self.sigmac_param) * self.sigmac_range
#         diffusion_kernel = self.create_gaussian_kernel(sigmac.item())
#         diffusion = conv2d(exposure, diffusion_kernel, intensity_output=True)
#         shrinkage = self.shrink_approx(diffusion)
#         resist_image = self.out_layer(exposure * shrinkage)
#         return resist_image

#     def forward(self, input_tensor):
#         aerial_image = self.get_aerial_image(input_tensor)
#         resist_image = self.get_resist_image(aerial_image)
#         return resist_image

import torch
import torch.nn as nn

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   


class UNet(nn.Module):

    def __init__(self, n_class=1):
        super().__init__()
                
        self.dconv_down1 = double_conv(1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out   
    

def conv2d(obj, psf, shape="same", intensity_output=False):
    _, _, im_height, im_width = obj.shape
    output_size_x = obj.shape[-2] + psf.shape[-2] - 1
    output_size_y = obj.shape[-1] + psf.shape[-1] - 1

    p2d_psf = (0, output_size_y - psf.shape[-1], 0, output_size_x - psf.shape[-2])
    p2d_obj = (0, output_size_y - obj.shape[-1], 0, output_size_x - obj.shape[-2])
    psf_padded = F.pad(psf, p2d_psf, mode="constant", value=0)
    obj_padded = F.pad(obj, p2d_obj, mode="constant", value=0)

    obj_fft = torch.fft.fft2(obj_padded)
    otf_padded = torch.fft.fft2(psf_padded)

    frequency_conv = obj_fft * otf_padded
    convolved = torch.fft.ifft2(frequency_conv)

    if shape == "same":
        convolved = central_crop(convolved, im_height, im_width)
    else:
        raise NotImplementedError

    if intensity_output:
        convolved = torch.abs(convolved)

    return convolved

def central_crop(variable, tw=None, th=None, dim=2):
    if dim == 2:
        w = variable.shape[-2]
        h = variable.shape[-1]
        if th is None:
            th = tw
        x1 = int(round((w - tw) / 2.0))
        y1 = int(round((h - th) / 2.0))
        cropped = variable[..., x1: x1 + tw, y1: y1 + th]
    elif dim == 1:
        h = variable.shape[-1]
        y1 = int(round((h - th) / 2.0))
        cropped = variable[..., y1: y1 + th]
    else:
        raise NotImplementedError
    return cropped


def train_litho_model(parameters: TrainingParameters) -> UNet:
    dataset = Dataset(folder_training_input, folder_training_output)
    train_dataloader, test_dataloader = create_dataloaders(
        dataset, split_percent=parameters.split_percent,
        bs=parameters.batch_size, shuffle=parameters.shuffle_dataset)

    model = UNet().to(device)
    optimizer = parameters.optimizer(
        model.parameters(), lr=parameters.learning_rate)
    train_losses = []
    valid_losses = []
    best_val_loss = np.inf
    early_stopping_counter = 0
    json_dict['training']['losses'] = {
        'train': train_losses, 'validation': valid_losses}
    t_range = trange(parameters.num_epochs)
    try:
        for epoch in t_range:

            # Training
            train_losses_local = []
            model.train()
            for mask, dart in train_dataloader:

                optimizer.zero_grad()
                output = model(mask)
                loss = parameters.loss_function(output, dart)

                train_losses_local.append(loss.item())
                loss.backward()
                optimizer.step()

            train_losses.append(np.mean(train_losses_local))

            # Validation
            valid_losses_local = []
            model.eval()
            for mask, dart in train_dataloader:

                output = model(mask)
                loss = parameters.loss_function(output, dart)
                valid_losses_local.append(loss.item())

            valid_losses.append(np.mean(valid_losses_local))

            # Plotting
            if epoch % parameters.plotting_interval == 0:
                plot_training(test_dataloader, model, train_losses,
                            valid_losses, folder_with_experiments)

            # Early stopping and saving
            if valid_losses[-1] < best_val_loss:
                early_stopping_counter = 0

                best_val_loss = valid_losses[-1]
                torch.save(model.state_dict(), os.path.join(
                    folder_with_experiments, f'model.pt'))
            else:
                early_stopping_counter += 1
                if early_stopping_counter > parameters.early_stopping_patience:
                    break

            t_range.set_description(
                f'Epoch {epoch} Loss: {valid_losses[-1]:.6f} counter: {early_stopping_counter} of {parameters.early_stopping_patience} ')
            t_range.update()
    except KeyboardInterrupt:
        print('Training interrupted')
        
    with open(os.path.join(folder_with_experiments, 'results.json'), 'w') as f:
        json.dump(json_dict, f, indent=4)

    return model


def optimize_mask(parameters: OptimizationParameters, model: UNet):

    dart_sim = transform.resize(np.load(parameters.target_dart_fname), [104, 104])
    mask_sim = transform.resize(np.load(parameters.target_mask_fname), [104, 104])

    # Put everything in tensors
    mask = np.ones_like(mask_sim)*0.5
    # mask = np.random.uniform(0, 1, mask_sim.shape) 
    model_input = torch.tensor(
        mask[np.newaxis, np.newaxis, ...], device=device, dtype=torch.float32, requires_grad=True)
    dart_target = torch.tensor(
        dart_sim[np.newaxis, np.newaxis, ...], device=device, dtype=torch.float32)
    # Lower and more adaptive learning rate
    optimizer = parameters.optimizer(
        [model_input], lr=parameters.learning_rate)
    array_losses = []

    model.eval()
    json_dict['optimization']['loss'] = array_losses

    t_range = trange(parameters.num_epochs)
    try:
        for epoch in t_range:
            optimizer.zero_grad()
            # x = torch.clamp(model_input, 0.0, 1.0)
            x = model_input
            dart_pred = model.forward(x)
            loss = parameters.loss_function(dart_pred, dart_target)
            loss.backward()

            optimizer.step()

            array_losses.append(loss.item())
            t_range.set_description(f'Epoch {epoch} Loss: {array_losses[-1]:.6f}')
            t_range.update()

            # Plotting
            if epoch % parameters.plotting_interval == 0:
                plot_optimization(model_input, dart_pred, dart_sim,
                                mask_sim, array_losses, folder_with_experiments)
    except KeyboardInterrupt:
        print('Optimization interrupted')

    np.save(os.path.join(folder_with_experiments,
            'final_mask.npy'), detensor(model_input))
    np.save(os.path.join(folder_with_experiments,
            'final_dart.npy'), detensor(dart_pred))
    with open(os.path.join(folder_with_experiments, 'results.json'), 'w') as f:
        json.dump(json_dict, f, indent=4)


if __name__ == '__main__':
    print('Starting', NOW)
    if USE_PRETRAINED_MODEL:
        print(f'Using pretrained model {pretrained_model_fname}')
        model = UNet().to(device)
        model.load_state_dict(torch.load(pretrained_model_fname))
        model.to(device)
        model.eval()
    else:
        model = train_litho_model(TrainingParameters)
    optimize_mask(OptimizationParameters, model)
    print('Finished')
