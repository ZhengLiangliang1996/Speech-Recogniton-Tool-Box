#! /usr/bin/env python

"""
Author: LiangLiang ZHENG
Date:
File Description
"""

from glob import glob
import numpy as np
import _pickle as pickle
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style(style='white')


def draw_loss():
    all_pickles_1 = sorted(glob("768_unit_2_layer_BIGRU.pickle"))
    all_pickles_2 = sorted(glob("768_unit_2_layer_lstm_forget.pickle"))
    all_pickles = np.concatenate((all_pickles_1, all_pickles_2))

    # Extract the name of each model
    model_names = ['encoded_2_'+item[5:-7] if 'test' in item else 'encoded_1_'+item[:-7] for item in all_pickles]
    print(model_names)
    # Extract the loss history for each model
    valid_loss = [pickle.load(open(i, "rb"))['val_loss'] for i in all_pickles]
    train_loss = [pickle.load(open(i, "rb"))['loss'] for i in all_pickles]
    # Save the number of epochs used to train each model
    num_epochs = [len(valid_loss[i]) for i in range(len(valid_loss))]
    print(num_epochs)
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    fig = plt.figure(figsize=(4, 3))

    # Plot the training loss vs. epoch for each model
    ax1 = fig.add_subplot(111)

    for i in range(len(all_pickles)):
        ax1.plot(np.linspace(1, num_epochs[i], num_epochs[i]),
                 train_loss[i], label=model_names[i])
        # Clean up the plot
        leg1 = ax1.legend()
        leg1.get_frame().set_edgecolor('black')
        ax1.set_xlim([1, max(num_epochs)])
        # ax1.set_yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Training CTC Loss')

        # Plot the validation loss vs. epoch for each model
    fig = plt.figure(figsize=(4, 3))
    ax2 = fig.add_subplot(111)
    for i in range(len(all_pickles)):
        ax2.plot(np.linspace(1, num_epochs[i], num_epochs[i]),
                 valid_loss[i], label=model_names[i])
    # # Clean up the plot
    leg = ax2.legend()
    leg.get_frame().set_edgecolor('black')
    ax2.set_xlim([1, max(num_epochs)])
    # ax2.set_yscale('log')
    plt.xlabel('Epoch')
    # plt.tight_layout()
    plt.ylabel('Validation CTC Loss')
    plt.show()


def main():
    draw_loss()


if __name__ == "__main__":
    main()
