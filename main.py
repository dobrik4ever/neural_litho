import torch
import torch.nn as nn
from datetime import datetime
import shutil
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
import json


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

USE_PRETRAINED_MODEL = False

if USE_PRETRAINED_MODEL:
    pretrained_model_fname = 'pretrained_model.pt'
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
    plotting_interval = 10
    batch_size = 10
    split_percent = 0.1  # How many percent of the data will be used for testing
    shuffle_dataset = False
    learning_rate = 1e-4
    num_epochs = 1000
    early_stopping_patience = 5
    optimizer = torch.optim.Adam
    loss_function = torch.nn.MSELoss()


class OptimizationParameters(Parameters):
    target_mask_fname = 'Mask_target.npy'
    target_dart_fname = 'Dart_target.npy'
    plotting_interval = 100
    learning_rate = 1e-1
    num_epochs = 1000
    optimizer = torch.optim.SGD
    loss_function = torch.nn.MSELoss()


json_dict['training'] = {'parameters': TrainingParameters.to_dict()}
json_dict['optimization'] = {'parameters': OptimizationParameters.to_dict()}
# === Utilities === #

if not os.path.exists(folder_with_experiments):  # Make sure the folder exists
    os.mkdir(folder_with_experiments)

shutil.copyfile('main.py', os.path.join(folder_with_experiments,
                f'main_{NOW}.py'))  # Copy file to the run folder


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
        mask = np.load(mask_fname).astype(np.float32)[np.newaxis, ...]
        dart = np.load(dart_fname).astype(np.float32)[np.newaxis, ...]

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


class NeuralLitho(nn.Module):

    def __init__(self):
        super().__init__()
        self.pointwise_net = NeuralPointwiseNet()
        self.areawise_net = NeuralAreawiseNet()

    def forward(self, input_tensor):
        point = self.pointwise_net(input_tensor)
        area = self.areawise_net(input_tensor)
        return point + area


def train_litho_model(parameters: TrainingParameters) -> NeuralLitho:
    dataset = Dataset(folder_training_input, folder_training_output)
    train_dataloader, test_dataloader = create_dataloaders(
        dataset, split_percent=parameters.split_percent,
        bs=parameters.batch_size, shuffle=parameters.shuffle_dataset)

    model = NeuralLitho().to(device)
    optimizer = parameters.optimizer(
        model.parameters(), lr=parameters.learning_rate)
    train_losses = []
    valid_losses = []
    best_val_loss = np.inf
    early_stopping_counter = 0
    json_dict['training']['losses'] = {
        'train': train_losses, 'validation': valid_losses}
    t_range = trange(parameters.num_epochs)
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
            best_val_loss = valid_losses[-1]
            torch.save(model.state_dict(), os.path.join(
                folder_with_experiments, f'model.pt'))
        else:
            early_stopping_counter += 1
            if early_stopping_counter > parameters.early_stopping_patience:
                break

        t_range.set_description(
            f'Epoch {epoch} Loss: {valid_losses[-1]:.6f} Stopping counter: {early_stopping_counter} of {parameters.early_stopping_patience}')
        t_range.update()

    with open(os.path.join(folder_with_experiments, 'results.json'), 'w') as f:
        json.dump(json_dict, f, indent=4)

    return model


def optimize_mask(parameters: OptimizationParameters, model: NeuralLitho):

    dart_sim = np.load(parameters.target_dart_fname)
    mask_sim = np.load(parameters.target_mask_fname)

    # Put everything in tensors
    mask = np.ones_like(mask_sim)*0.5
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

    np.save(os.path.join(folder_with_experiments,
            'final_mask.npy'), detensor(model_input))
    np.save(os.path.join(folder_with_experiments,
            'final_dart.npy'), detensor(dart_pred))
    with open(os.path.join(folder_with_experiments, 'results.json'), 'w') as f:
        json.dump(json_dict, f, indent=4)


if __name__ == '__main__':
    if USE_PRETRAINED_MODEL:
        print(f'Using pretrained model {pretrained_model_fname}')
        model = NeuralLitho()
        model.load_state_dict(torch.load(pretrained_model_fname))
        model.to(device)
        model.eval()
    else:
        model = train_litho_model(TrainingParameters)
    optimize_mask(OptimizationParameters, model)
    print('Finished')
