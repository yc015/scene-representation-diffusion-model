import torch
from torch.utils.data import Dataset

from torchvision.io import read_image

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import spearmanr, pearsonr

import time

torch_device = "cuda" if torch.cuda.is_available() else "cpu"
tic, toc = (time.time, time.time)


class ModuleHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = None
        self.features = []

    def hook_fn(self, module, input, output):
        self.module = module
        self.features.append(output.detach())

    def close(self):
        self.hook.remove()

        
def clear_dir(dir_name, file_extention):
    for filename in os.listdir(dir_name):
        file_path = os.path.join(dir_name, filename)
        if (os.path.isfile(file_path) or os.path.islink(file_path)) and file_path.endswith(file_extention):
            os.remove(file_path)

        
def dice_coeff(pred, target):
    smooth = 1e-8
    num = pred.size(0)
    m1 = pred.view(num, -1).float()  # Flatten
    m2 = target.view(num, -1).float()  # Flatten
    intersection = (m1 * m2).sum(1).float()

    return (2. * intersection + smooth) / (m1.sum(1) + m2.sum(1) + smooth)


def weighted_f1(pred, target, beta=1):
    smooth = 1e-8
    num = pred.size(0)
    m1 = pred.view(num, -1).float()  # Flatten
    m2 = target.view(num, -1).float()  # Flatten
    TP = (m1 * m2).sum(1).float()
    FN = ((1 - m1) * m2).sum(1).float()
    precision = TP / m1.sum(1)
    recall = TP / m2.sum(1)
    
    return ((1 + (beta ** 2)) * precision * recall + smooth) / ((beta ** 2) * precision + recall + smooth)


def plt_test_results(probe, test_dataloader, test_data, loss_func, metrics=dice_coeff, head=None, save_plt=False, 
                     save_filename="fig.png"):
    probe.eval()
    test_results = test(probe, torch_device, test_dataloader, loss_func=loss_func, return_raw_outputs=True, 
                        metrics=dice_coeff, head=head)

    test_size = 20
    if save_plt:
        plt.ioff()
    fig, ax = plt.subplots(3, test_size, figsize=(2 * test_size, 7.5))
    
    for i in range(test_size):
        dice_score = dice_coeff(torch.argmax(probe(test_data[i][1].unsqueeze(0).to(torch_device)[..., :]), dim=1).cpu(),
                                test_data[i][2].unsqueeze(0))[0]
        if i == 0:
            ax[0][0].set_title(f"Original Image", fontsize=18)
            ax[1][0].set_title(f"Pred Mask\nDice {dice_score:.3f}", fontsize=18)
            ax[2][0].set_title(f"Syn Mask", fontsize=18)
        else:
            ax[1][i].set_title(f"Dice {dice_score:.3f}", fontsize=18)
        ax[0][i].axis("off")
        ax[0][i].imshow(test_data[i][0].permute([1, 2, 0]))
        ax[1][i].imshow(torch.argmax(test_results[2][i].cpu().detach(), dim=0), vmin=0, vmax=1)
        ax[1][i].axis("off")
        ax[2][i].imshow(test_data[i][2])
        ax[2][i].axis("off")

    plt.tight_layout()
    
    if save_plt:
        plt.savefig(save_filename, dpi=120)
    
    plt.show()
    
    
def train(model, device, train_loader, optimizer, epoch, loss_func, 
          class_names=None, report=False, verbose_interval=5, metrics=dice_coeff,
          head=None, verbose=True):
    """
    :param model: pytorch model (class:torch.nn.Module)
    :param device: device used to train the model (e.g. torch.device("cuda") for training on GPU)
    :param train_loader: torch.utils.data.DataLoader of train dataset
    :param optimizer: optimizer for the model
    :param epoch: current epoch of training
    :param loss_func: loss function for the training
    :param class_names: str Name for the classification classses. used in train report
    :param report: whether to print a classification report of training 
    :param train_verbose: print a train progress report after how many batches of training in each epoch
    :return: average loss, train accuracy
    """
    assert (verbose_interval is None) or verbose_interval > 0, "invalid verbose_interval, verbose_interval(int) > 0"
    starttime = tic()
    # Set the model to the train mode: Essential for proper gradient descent
    model.train()
    loss_sum = 0
    foreground_scores = []
    background_scores = []
    # Iterate through the train dataset
    for batch_idx, (_, data, target) in enumerate(train_loader):
        batch_size = data.shape[0]
        
        data, target = data.to(device), target.to(device)
        if not head is None:
            head_dim = data.shape[-1] // 8
            data = data[..., head * head_dim: (head + 1) * head_dim]
        optimizer.zero_grad()
        output = model(data)
        # Compute the loss
        loss = loss_func(output, target)
        # Call Gradient descent
        loss.backward()
        optimizer.step()
        
        dice_score = metrics(torch.argmax(output, dim=1), target)
        foreground_scores.append(dice_score.cpu().numpy())

        dice_score = metrics(torch.argmax(output, dim=1) == 0, target == 0)
        background_scores.append(dice_score.cpu().numpy())
        
        # If print train progress
        if verbose_interval and batch_idx % verbose_interval == 0 and batch_idx != 0:
            print('Train Epoch: {} [{}/{} ({:.3f})]\tLoss: {:.6f}  FG Dice Coeff:{:.3f} BG Dice Coeff:{:.3f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100 * batch_idx / len(train_loader), 
                loss.item(), 
                torch.mean(torch.Tensor(np.hstack(foreground_scores))),
                torch.mean(torch.Tensor(np.hstack(background_scores)))))
        loss_sum += loss.sum().item()
            
    foreground_scores = np.hstack(foreground_scores)
    background_scores = np.hstack(background_scores)        
    
    train_acc = [torch.mean(torch.Tensor(foreground_scores)), torch.mean(torch.Tensor(background_scores))]
    loss_avg = loss_sum / len(train_loader)
    
    endtime = toc()
    if verbose:
        print('\nTrain set: Average loss: {:.4f} ({:.3f} sec)  FG Dice Coeff:{:.3f} BG Dice Coeff:{:.3f}\n'.\
              format(loss_avg, 
                     endtime-starttime,
                     train_acc[0],
                     train_acc[1]))

    return loss_avg, [foreground_scores, background_scores]


def test(model, device, test_loader, loss_func, return_raw_outputs=False, metrics=dice_coeff, head=None, verbose=True):
    """
    :param model: pytorch model (class:torch.nn.Module)
    :param device: device used to train the model (e.g. torch.device("cuda") for training on GPU)
    :param test_loader: torch.utils.data.DataLoader of test dataset
    :param loss_func: loss function for the training
    :param class_names: str Name for the classification classses. used in train report
    :param test_report: whether to print a classification report of testing after each epoch
    :param return_raw_outputs: whether return the raw outputs of model (before argmax). used for auc computation
    """
    # Set the model to evaluation mode: Essential for testing model
    model.eval()
    test_loss = 0
    foreground_scores = []
    background_scores = []
    if return_raw_outputs:
        raw_predictions = []
        
    # Do not call gradient descent on the test set
    # We don't adjust the weights of model on the test set
    with torch.no_grad():
        for _, data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if not head is None:
                head_dim = data.shape[-1] // 8
                data = data[..., head * head_dim: (head + 1) * head_dim]
            output = model(data)
            
            test_loss += loss_func(output, target).sum().item()  # sum up batch loss
            
            if return_raw_outputs:
                raw_predictions += output
                
            dice_score = metrics(torch.argmax(output, dim=1), target)
            foreground_scores.append(dice_score.cpu().numpy())
            
            dice_score = metrics(torch.argmax(output, dim=1) == 0, target == 0)
            background_scores.append(dice_score.cpu().numpy())

    foreground_scores = np.hstack(foreground_scores)
    background_scores = np.hstack(background_scores)
            
    test_loss /= len(test_loader)
    
    test_acc = [torch.mean(torch.Tensor(foreground_scores)), torch.mean(torch.Tensor(background_scores))]

    if verbose:
        print('Test set: Average loss: {:.4f},             FG Dice Coeff:{:.3f} BG Dice Coeff:{:.3f}\n'.format(
            test_loss,
            test_acc[0],
            test_acc[1]))
        
    # If return the raw outputs (before argmax) from the model
    if return_raw_outputs:
        return test_loss, [foreground_scores, background_scores], raw_predictions
    else:
        return test_loss, [foreground_scores, background_scores]


def train_continuous_depth(model, device, train_loader, optimizer, epoch, loss_func, 
                           class_names=None, report=False, verbose_interval=5, smooth_loss=None,
                           head=None, verbosity=True, alpha=0.2,
                           lasso=False, l1_lambda=0.1):
    """
    :param model: pytorch model (class:torch.nn.Module)
    :param device: device used to train the model (e.g. torch.device("cuda") for training on GPU)
    :param train_loader: torch.utils.data.DataLoader of train dataset
    :param optimizer: optimizer for the model
    :param epoch: current epoch of training
    :param loss_func: loss function for the training
    :param class_names: str Name for the classification classses. used in train report
    :param report: whether to print a classification report of training 
    :param train_verbose: print a train progress report after how many batches of training in each epoch
    :return: average loss
    """
    assert (verbose_interval is None) or verbose_interval > 0, "invalid verbose_interval, verbose_interval(int) > 0"
    starttime = tic()
    # Set the model to the train mode: Essential for proper gradient descent
    model.train()
    loss_sum = 0
    # Iterate through the train dataset
    rank_corr_scores = []
    linear_corr_scores = []
    rmses = []
    for batch_idx, (image, data, target) in enumerate(train_loader):
        batch_size = data.shape[0]
        image = image.to(device)
        
        data, target = data.to(device), target.to(device).to(torch.float).unsqueeze(1)
        if not head is None:
            head_dim = data.shape[-1] // 8
            data = data[..., head * head_dim: (head + 1) * head_dim]
        optimizer.zero_grad()
        output = model(data)
        # Compute the loss
        loss = loss_func(output, target)
        if smooth_loss:
            loss += alpha * smooth_loss(output, image)
            
        if lasso:
            weights = torch.cat([x.view(-1) for x in model.parameters()])
            l1_reg  = l1_lambda * torch.norm(weights, 1)
            loss += l1_reg
        # Call Gradient descent
        loss.backward()
        optimizer.step()
        
        depth_gts = target.cpu().detach().clone().numpy()
        depth_preds = output.cpu().detach().clone().numpy()
        
        if verbosity:
            for i in range(depth_gts.shape[0]):
                depth_gt = depth_gts[i].ravel()
                depth_pred = depth_preds[i].ravel()
                corr = spearmanr(depth_gt, depth_pred, alternative="two-sided", nan_policy="raise", axis=None)[0]
                if not np.isnan(corr):
                    rank_corr_scores.append(corr)
                corr = pearsonr(depth_gt, depth_pred)[0]
                if not np.isnan(corr):
                    linear_corr_scores.append(corr)
                rmse = np.sqrt(np.mean((depth_gt - depth_pred) ** 2))
                rmses.append(rmse)
        # If print train progress
        if verbosity and batch_idx % verbose_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.3f})]\tLoss: {:.6f} Rank Corr: {:.3f} Linear Corr: {:.3f} RMSE: {:.3f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100 * batch_idx / len(train_loader), 
                loss.item(), 
                np.mean(rank_corr_scores),
                np.mean(linear_corr_scores),
                np.mean(rmses),
                ))
        loss_sum += loss.sum().item()  
    
    loss_avg = loss_sum / len(train_loader)
    
    endtime = toc()
    if verbosity:
        print('\nTrain set: Average loss: {:.4f} ({:.3f} sec) Avg Rank Corr: {:.3f} Avg Linear Corr: {:.3f} RMSE: {:.3f}\n'.\
              format(loss_avg, 
                     endtime-starttime,
                     np.mean(rank_corr_scores),
                     np.mean(linear_corr_scores),
                     np.mean(rmses)
                     ))
    
    return loss_avg


def test_continuous_depth(model, device, test_loader, loss_func, return_raw_outputs=False, head=None, 
                          scheduler=None, smooth_loss=None,
                          verbosity=True, alpha=0.2,
                          lasso=False, l1_lambda=0.1):
    """
    :param model: pytorch model (class:torch.nn.Module)
    :param device: device used to train the model (e.g. torch.device("cuda") for training on GPU)
    :param test_loader: torch.utils.data.DataLoader of test dataset
    :param loss_func: loss function for the training
    :param class_names: str Name for the classification classses. used in train report
    :param test_report: whether to print a classification report of testing after each epoch
    :param return_raw_outputs: whether return the raw outputs of model (before argmax). used for auc computation
    """
    # Set the model to evaluation mode: Essential for testing model
    model.eval()
    test_loss = 0
    rank_corr_scores = []
    linear_corr_scores = []
    rmses = []
    if return_raw_outputs:
        raw_predictions = []
        
    # Do not call gradient descent on the test set
    # We don't adjust the weights of model on the test set
    with torch.no_grad():
        for image, data, target in test_loader:
            image = image.to(device)
            data, target = data.to(device), target.to(device).to(torch.float).unsqueeze(1)
            if not head is None:
                head_dim = data.shape[-1] // 8
                data = data[..., head * head_dim: (head + 1) * head_dim]
            output = model(data)
            
            test_loss += loss_func(output, target).sum().item()  # sum up batch loss
            if smooth_loss:
                test_loss += alpha * smooth_loss(output, image).sum().item()
                
            if lasso:
                weights = torch.cat([x.view(-1) for x in model.parameters()])
                l1_reg  = l1_lambda * torch.norm(weights, 1)
                test_loss += l1_reg
            
            if return_raw_outputs:
                raw_predictions += output
             
            depth_gts = target.cpu().detach().clone().numpy()
            depth_preds = output.cpu().detach().clone().numpy()
            # if verbosity:
            if verbosity:
                for i in range(depth_gts.shape[0]):
                    depth_gt = depth_gts[i].ravel()
                    depth_pred = depth_preds[i].ravel()
                    corr = spearmanr(depth_gt, depth_pred, alternative="two-sided", nan_policy="raise", axis=None)[0]
                    if not np.isnan(corr):
                        rank_corr_scores.append(corr)
                    corr = pearsonr(depth_gt, depth_pred)[0]
                    if not np.isnan(corr):
                        linear_corr_scores.append(corr)
                    rmse = np.sqrt(np.mean((depth_gt - depth_pred) ** 2))
                    rmses.append(rmse)
            
    if scheduler:
        scheduler.step(test_loss)
    test_loss /= len(test_loader)
    
    if verbosity:
        print('Test set: Average loss: {:.4f} Avg Rank Corr: {:.3f} Avg Linear Corr: {:.3f} RMSE: {:.3f}\n'.format(
            test_loss, np.mean(rank_corr_scores), np.mean(linear_corr_scores), np.mean(rmses)
            ))
        
    # If return the raw outputs (before argmax) from the model
    if return_raw_outputs:
        return test_loss, raw_predictions, [rank_corr_scores, linear_corr_scores, rmses]
    else:
        return test_loss,
    
    
def plt_test_results_continuous_depth(probe, test_dataloader, test_data, loss_func, smooth_loss, head=None,
                                      sampled=False, save=False, save_plt=False, norm_output=True,
                                      save_filename="fig.png"):
    test_results = test_continuous_depth(probe, torch_device, test_dataloader, loss_func=loss_func, 
                                         return_raw_outputs=True, head=head, smooth_loss=smooth_loss, 
                                         alpha=1)

    test_size = 20
    fig, ax = plt.subplots(3, test_size, figsize=(2 * test_size, 7.8))

    if sampled:
        test_samples = np.random.choice(np.arange(len(test_dataloader.dataset)), test_size, replace=False)
    else:
        test_samples = range(test_size)
    
    for i in range(test_size):
        ind = test_samples[i]
        
        depth_pred = test_results[1][ind].cpu().detach()[0].clone()
        depth_pred = depth_pred.numpy().ravel()
        depth_gt = test_data[ind][2].clone().numpy().ravel()
        
        rank_corr = spearmanr(depth_gt, depth_pred, alternative="two-sided")[0]
        linear_corr = pearsonr(depth_gt, depth_pred)[0]

        if i == 0:
            ax[0][0].set_title(f"Original Image", fontsize=18)
            ax[1][0].set_title(f"Pred Depth\nRank Corr {rank_corr:.3f}\nLinear Corr {linear_corr:.3f}", fontsize=15)
            ax[2][0].set_title(f"Syn Depth", fontsize=18)
        else:
            pass
            ax[1][i].set_title(f"Rank Corr {rank_corr:.3f}\nLinear Corr {linear_corr:.3f}", fontsize=15)
        ax[0][i].axis("off")
        ax[0][i].imshow(test_data[ind][0].permute([1, 2, 0]))
        if norm_output:
            ax[1][i].imshow(test_results[1][ind].cpu().detach()[0], vmin=0, vmax=1)
        else:
            ax[1][i].imshow(test_results[1][ind].cpu().detach()[0])
        ax[1][i].axis("off")
        ax[2][i].imshow(test_data[ind][2])
        ax[2][i].axis("off")

    plt.tight_layout()
    
    if save_plt:
        plt.savefig(save_filename, dpi=120)
    
    plt.show()

    
def read_saliency_probing_scores(results_dir, postfix="self_attn_out"):
    foreback_performances = {}
    for folder in os.listdir(results_dir):
        if "ipynb" in folder:
            continue
        foreback_performances[folder] = {}
        pickle_filenames = os.listdir(os.path.join(results_dir, folder))
        pickle_filenames = [filename for filename in pickle_filenames if filename.endswith(".pkl")]
        for pickled_stats in pickle_filenames:
            if postfix in pickled_stats:
                block_layer = pickled_stats[len("saved_test_results_"):pickled_stats.rfind(f"_{postfix}")]
            else:
                continue
            block = block_layer[:block_layer.find("_")]
            block_ind = block_layer[block_layer.find("_")+1:block_layer.rfind("_layer")]
            layer_ind = block_layer[block_layer.rfind("_")+1:]
            with open(os.path.join(results_dir, folder, pickled_stats), "rb") as infile:
                saved_stats = pickle.load(infile)
                foreground_dice = np.mean(saved_stats[0])
                background_dice = np.mean(saved_stats[1])

            foreback_performances[folder][f"{block}_{block_ind}_layer_{layer_ind}"] = [foreground_dice, background_dice]

    layer_orders = [f"down_0_layer_{i}" for i in range(2)] + [f"down_1_layer_{i}" for i in range(2)] + [f"down_2_layer_{i}" for i in range(2)] + ["mid_0_layer_0"]
    layer_orders += [f"up_1_layer_{i}" for i in range(3)] + [f"up_2_layer_{i}" for i in range(3)] + [f"up_3_layer_{i}" for i in range(3)]
    
    for step in foreback_performances.keys():
        ordered_fore_scores = [0] * len(layer_orders)
        ordered_back_scores = [0] * len(layer_orders)
        for i in range(len(layer_orders)):
            try:
                ordered_fore_scores[i] = foreback_performances[step][layer_orders[i]][0] 
                ordered_back_scores[i] = foreback_performances[step][layer_orders[i]][1]
            except:
                pass
        foreback_performances[step]["ordered_foreground_scores"] = ordered_fore_scores
        foreback_performances[step]["ordered_background_scores"] = ordered_back_scores

    return foreback_performances
    
    
def read_continuous_probing_scores(results_dir, postfix="self_attn_out"):
    foreback_performances = {}
    for folder in os.listdir(results_dir):
        if "ipynb" in folder:
            continue
        foreback_performances[folder + "_no_bias"] = {}
        for pickled_stats in os.listdir(os.path.join(results_dir, folder)):
            if not pickled_stats.endswith(".pkl"):
                continue
            block_layer = pickled_stats[len("saved_test_results_"):pickled_stats.rfind(f"_{postfix}")]
            block = block_layer[:block_layer.find("_")]
            block_ind = block_layer[block_layer.find("_")+1:block_layer.rfind("_layer")]
            layer_ind = block_layer[block_layer.rfind("_")+1:]
            with open(os.path.join(results_dir, folder, pickled_stats), "rb") as infile:
                saved_stats = pickle.load(infile)
                rank_corr = np.mean(saved_stats[0])
                linear_corr = np.mean(saved_stats[1])
                rmse = np.mean(saved_stats[2])

            if "_linear_no_bias" in pickled_stats:
                foreback_performances[folder + "_no_bias"][f"{block}_{block_ind}_layer_{layer_ind}"] = [rank_corr, linear_corr, rmse]
            else:
                continue

    layer_orders = [f"down_0_layer_{i}" for i in range(2)] + [f"down_1_layer_{i}" for i in range(2)] + [f"down_2_layer_{i}" for i in range(2)] + ["mid_0_layer_0"]
    layer_orders += [f"up_1_layer_{i}" for i in range(3)] + [f"up_2_layer_{i}" for i in range(3)] + [f"up_3_layer_{i}" for i in range(3)]

    for step in foreback_performances.keys():
        ordered_rank_corrs = [0] * len(layer_orders)
        ordered_linear_corrs = [0] * len(layer_orders)
        ordered_rmses = [0] * len(layer_orders)
        for i in range(len(layer_orders)):
            try:
                ordered_rank_corrs[i] = foreback_performances[step][layer_orders[i]][0] 
                ordered_linear_corrs[i] = foreback_performances[step][layer_orders[i]][1]
                ordered_rmses[i] = foreback_performances[step][layer_orders[i]][2]
            except:
                pass
        foreback_performances[step]["ordered_rank_corrs"] = ordered_rank_corrs
        foreback_performances[step]["ordered_linear_corrs"] = ordered_linear_corrs
        foreback_performances[step]["ordered_rmse"] = ordered_rmses
        
    return foreback_performances
    