import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn', force=True)  # or 'forkserver'

import argparse
import os
import time
from IPython import embed

import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms

import numpy as np
import common_args
import random
from dataset import Dataset, ImageDataset
from net import Transformer, ImageTransformer
from utils import (
    build_bandit_data_filename,
    build_bandit_model_filename,
    build_linear_bandit_data_filename,
    build_linear_bandit_model_filename,
    build_darkroom_data_filename,
    build_darkroom_model_filename,
    build_miniworld_data_filename,
    build_miniworld_model_filename,
    build_maze_data_filename,
    build_maze_model_filename,
    worker_init_fn,
)

device = torch.device("mps")


if __name__ == '__main__':
    if not os.path.exists('figs/loss'):
        os.makedirs('figs/loss', exist_ok=True)
    if not os.path.exists('models'):
        os.makedirs('models', exist_ok=True)

    parser = argparse.ArgumentParser()
    common_args.add_dataset_args(parser)
    common_args.add_model_args(parser)
    common_args.add_train_args(parser)

    parser.add_argument('--seed', type=int, default=0)

    args = vars(parser.parse_args())
    print("Args: ", args)

    env = args['env']
    n_envs = args['envs']
    n_hists = args['hists']
    n_samples = args['samples']
    horizon = args['H']
    dim = args['dim']
    state_dim = dim
    action_dim = dim
    n_embd = args['embd']
    n_head = args['head']
    n_layer = args['layer']
    lr = args['lr']
    shuffle = args['shuffle']
    dropout = args['dropout']
    var = args['var']
    cov = args['cov']
    num_epochs = args['num_epochs']
    seed = args['seed']
    lin_d = args['lin_d']
    
    tmp_seed = seed
    if seed == -1:
        tmp_seed = 0


    torch.manual_seed(tmp_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(tmp_seed)
        torch.cuda.manual_seed_all(tmp_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(tmp_seed)
    random.seed(tmp_seed)

    if shuffle and env == 'linear_bandit':
        raise Exception("Are you sure you want to shuffle on the linear bandit? Data collected from an adaptive algorithm in a stochastic setting can bias the learner if shuffled.")

    dataset_config = {
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
        'dim': dim,
    }
    model_config = {
        'shuffle': shuffle,
        'lr': lr,
        'dropout': dropout,
        'n_embd': n_embd,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_envs': n_envs,
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
        'dim': dim,
        'seed': seed,
    }
    if env == 'bandit':
        state_dim = 1

        dataset_config.update({'var': var, 'cov': cov, 'type': 'uniform'})
        path_train = build_bandit_data_filename(
            env, n_envs, dataset_config, mode=0)
        path_test = build_bandit_data_filename(
            env, n_envs, dataset_config, mode=1)

        model_config.update({'var': var, 'cov': cov})
        filename = build_bandit_model_filename(env, model_config)

    elif env == 'bandit_thompson':
        state_dim = 1

        dataset_config.update({'var': var, 'cov': cov, 'type': 'bernoulli'})
        path_train = build_bandit_data_filename(
            env, n_envs, dataset_config, mode=0)
        path_test = build_bandit_data_filename(
            env, n_envs, dataset_config, mode=1)

        model_config.update({'var': var, 'cov': cov})
        filename = build_bandit_model_filename(env, model_config)

    elif env == 'linear_bandit':
        state_dim = 1

        dataset_config.update({'lin_d': lin_d, 'var': var, 'cov': cov})
        path_train = build_linear_bandit_data_filename(
            env, n_envs, dataset_config, mode=0)
        path_test = build_linear_bandit_data_filename(
            env, n_envs, dataset_config, mode=1)

        model_config.update({'lin_d': lin_d, 'var': var, 'cov': cov})
        filename = build_linear_bandit_model_filename(env, model_config)

    elif env.startswith('darkroom'):
        state_dim = 2
        action_dim = 5

        dataset_config.update({'rollin_type': 'uniform'})
        path_train = build_darkroom_data_filename(
            env, n_envs, dataset_config, mode=0)
        path_test = build_darkroom_data_filename(
            env, n_envs, dataset_config, mode=1)

        filename = build_darkroom_model_filename(env, model_config)

    elif env == "Maze":
        state_dim = 2
        action_dim = 4

        dataset_config.update({'rollin_type': 'uniform'})
        coverage_bad = {"mode": True, "size": 2}
        coverage_good = {"mode": False}
        walls = "fixed_walls"


        path_train = build_maze_data_filename(
            env, n_envs, horizon, dataset_config, mode=0, coverage=coverage_good, walls = walls)
        path_icl = build_maze_data_filename(
            env, n_envs, horizon, dataset_config, mode=1, coverage=coverage_good, walls = walls)
        path_iwl = build_maze_data_filename(
            env, n_envs, horizon, dataset_config, mode=2, coverage=coverage_bad, walls = walls)

        filename = build_maze_model_filename(env, model_config)

    elif env == 'miniworld':
        state_dim = 2   # direction vector is 2D, no position included
        action_dim = 4

        dataset_config.update({'rollin_type': 'uniform'})

        increment = 5000
        starts = np.arange(0, n_envs, increment)
        starts = np.array(starts)
        ends = starts + increment - 1

        paths_train = []
        paths_test = []
        for start_env_id, end_env_id in zip(starts, ends):
            path_train = build_miniworld_data_filename(
                env, start_env_id, end_env_id, dataset_config, mode=0)
            path_test = build_miniworld_data_filename(
                env, start_env_id, end_env_id, dataset_config, mode=1)

            paths_train.append(path_train)
            paths_test.append(path_test)

        filename = build_miniworld_model_filename(env, model_config)
        print(f"Generate filename: {filename}")

    else:
        raise NotImplementedError

    config = {
        'horizon': horizon,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'n_layer': n_layer,
        'n_embd': n_embd,
        'n_head': n_head,
        'shuffle': shuffle,
        'dropout': dropout,
        'test': False,
        'store_gpu': True,
    }
    if env == 'miniworld':
        config.update({'image_size': 25, 'store_gpu': False})
        model = ImageTransformer(config).to(device)
    else:
        model = Transformer(config).to(device)

    params = {
        'batch_size': 64,
        'shuffle': True,
    }

    log_filename = f'figs/loss/{filename}_logs.txt'
    with open(log_filename, 'w') as f:
        pass
    def printw(string):
        """
        A drop-in replacement for print that also writes to a log file.
        """
        # Use the standard print function to print to the console
        print(string)

        # Write the same output to the log file
        with open(log_filename, 'a') as f:
            print(string, file=f)




    if env == 'miniworld':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        params.update({'num_workers': 16,
                'prefetch_factor': 2,
                'persistent_workers': True,
                'pin_memory': True,
                'batch_size': 64,
                'worker_init_fn': worker_init_fn,
            })


        printw("Loading miniworld data...")
        train_dataset = ImageDataset(paths_train, config, transform)
        test_dataset = ImageDataset(paths_test, config, transform)
        printw("Done loading miniworld data")
    else:
        train_dataset = Dataset(path_train, config)
        # test_dataset = Dataset(path_test, config)

        # train_loader = torch.utils.data.DataLoader(train_dataset, **params)
        # test_loader = torch.utils.data.DataLoader(test_dataset, **params)

    if env == 'Maze':
        train_dataset = Dataset(path_train, config)
        icl_dataset = Dataset(path_icl, config)
        iwl_dataset = Dataset(path_iwl, config)

        train_loader = torch.utils.data.DataLoader(train_dataset, **params)
        icl_loader = torch.utils.data.DataLoader(icl_dataset, **params)
        iwl_loader = torch.utils.data.DataLoader(iwl_dataset, **params)

        printw("Num train batches: " + str(len(train_loader)))
        printw("Num icl batches: " + str(len(icl_loader)))
        printw("Num iwl batches: " + str(len(iwl_loader)))



    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    # test_loss = []
    # train_loss = []

    train_loss = []
    icl_loss = []
    iwl_loss = []
    icl_accuracy = []
    iwl_accuracy = []
    train_accuracy = []

    def calculate_accuracy(pred_actions, true_actions):
        _, pred_classes = torch.max(pred_actions, dim=1)
        _, true_classes = torch.max(true_actions, dim=1)
        correct_predictions = (pred_classes == true_classes).sum().item()
        return correct_predictions

    for epoch in range(num_epochs):
        printw(f"Epoch: {epoch + 1}")
        if env == "Maze":
            start_time = time.time()
            with torch.no_grad():
                epoch_icl_loss = 0.0
                total_correct_icl = 0
                total_samples_icl = 0
                for i, batch in enumerate(icl_loader):
                    print(f"Batch {i} of {len(icl_loader)}", end='\r')
                    batch = {k: v.to(device) for k, v in batch.items()}
                    true_actions = batch['optimal_actions']
                    pred_actions = model(batch)
                    true_actions = true_actions.unsqueeze(1).repeat(1, pred_actions.shape[1], 1)
                    true_actions = true_actions.reshape(-1, action_dim)
                    pred_actions = pred_actions.reshape(-1, action_dim)

                    loss = loss_fn(pred_actions, true_actions)
                    epoch_icl_loss += loss.item() / horizon

                    # Calculate accuracy
                    total_correct_icl += calculate_accuracy(pred_actions, true_actions)
                    total_samples_icl += true_actions.size(0)

                icl_loss.append(epoch_icl_loss / len(icl_dataset))
                icl_accuracy.append(total_correct_icl / total_samples_icl)

            end_time = time.time()
            print(f"\tICL loss: {icl_loss[-1]}")
            print(f"\tICL accuracy: {icl_accuracy[-1] * 100}%")
            print(f"\tEval time: {end_time - start_time}")

            start_time = time.time()
            with torch.no_grad():
                epoch_iwl_loss = 0.0
                total_correct_iwl = 0
                total_samples_iwl = 0
                for i, batch in enumerate(iwl_loader):
                    print(f"Batch {i} of {len(iwl_loader)}", end='\r')
                    batch = {k: v.to(device) for k, v in batch.items()}
                    true_actions = batch['optimal_actions']
                    pred_actions = model(batch)
                    true_actions = true_actions.unsqueeze(1).repeat(1, pred_actions.shape[1], 1)
                    true_actions = true_actions.reshape(-1, action_dim)
                    pred_actions = pred_actions.reshape(-1, action_dim)

                    loss = loss_fn(pred_actions, true_actions)
                    epoch_iwl_loss += loss.item() / horizon

                    # Calculate accuracy
                    total_correct_iwl += calculate_accuracy(pred_actions, true_actions)
                    total_samples_iwl += true_actions.size(0)

                iwl_loss.append(epoch_iwl_loss / len(iwl_dataset))
                iwl_accuracy.append(total_correct_iwl / total_samples_iwl)

            end_time = time.time()
            print(f"\tIWL loss: {iwl_loss[-1]}")
            print(f"\tIWL accuracy: {iwl_accuracy[-1] * 100}%")
            print(f"\tEval time: {end_time - start_time}")

        # TRAINING
        epoch_train_loss = 0.0
        epoch_train_correct = 0.0
        epoch_train_samples = 0.0

        start_time = time.time()

        for i, batch in enumerate(train_loader):
            print(f"Batch {i} of {len(train_loader)}", end='\r')
            batch = {k: v.to(device) for k, v in batch.items()}
            true_actions = batch['optimal_actions']
            pred_actions = model(batch)
            true_actions = true_actions.unsqueeze(
                1).repeat(1, pred_actions.shape[1], 1)
            true_actions = true_actions.reshape(-1, action_dim)
            pred_actions = pred_actions.reshape(-1, action_dim)

            optimizer.zero_grad()
            loss = loss_fn(pred_actions, true_actions)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() / horizon
            epoch_train_correct += calculate_accuracy(pred_actions, true_actions)
            epoch_train_samples += true_actions.size(0)


        train_loss.append(epoch_train_loss / len(train_dataset))
        train_accuracy.append(epoch_train_correct / epoch_train_samples)
        end_time = time.time()
        printw(f"\tTrain loss: {train_loss[-1]}")
        printw(f"\tTrain time: {end_time - start_time}")


        # LOGGING
        if (epoch + 1) % 50 == 0 or (env == 'linear_bandit' and (epoch + 1) % 10 == 0):
            torch.save(model.state_dict(),
                       f'models/{filename}_epoch{epoch+1}.pt')


        if (epoch + 1) % 10 == 0:
            printw(f"Epoch: {epoch + 1}")
            printw(f"ICL Loss:        {icl_loss[-1]}")
            printw(f"IWL Loss:        {iwl_loss[-1]}")
            printw(f"Train Loss:       {train_loss[-1]}")
            printw("\n")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            ax1.set_yscale('log')
            ax1.plot(train_loss[1:], label="Train Loss")
            ax1.plot(icl_loss[1:], label="ICL Loss")
            ax1.plot(iwl_loss[1:], label="IWL Loss")
            ax1.legend()
            ax1.set_ylabel("Cross-Entropy Loss")

            ax2.plot(train_accuracy, label="Train Accuracy")
            ax2.plot(icl_accuracy, label="ICL Accuracy")
            ax2.plot(iwl_accuracy, label="IWL Accuracy")
            ax2.legend()
            ax2.set_ylabel('Accuracy')
            ax2.set_xlabel('Epochs')

            plt.tight_layout()
            plt.savefig(f"figs/loss/{filename}_train_loss_accuracy.png")
            plt.clf()

    torch.save(model.state_dict(), f'models/{filename}.pt')
    print("Done.")
