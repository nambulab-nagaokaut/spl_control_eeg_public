from multiprocessing import freeze_support

import Functions
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

if __name__ == "__main__":
    freeze_support()  # Necessary for parallelization

    matdir = "./data/exp1"  # Data directory, exp1 or exp2
    matpathlist = Functions.search_mat(matdir)  # Data search

    evoked_list = []
    epochs_allsub = []

    for matpath in matpathlist:
        userdir = "result" + matpath.replace(".mat", "").replace(matdir, "")

        data, label, group = Functions.LoadData(matpath)

        data2, label2, group2 = Functions.LoadData(matpath)

        conv_data, conv_label = Functions.Data_Formatting(
            data, label
        )  # Create data and labels for training and other purposes

        conv_data2, conv_label2 = Functions.Data_Formatting(
            data2, group2
        )  # Create data and labels for training and other purposes

        # Remove unnecessary labels (for oddball tasks, remove standard stimuli)
        indexes = [i for i, x in enumerate(conv_label) if x == 2]  # Index search
        print("Removed label indexes: " + str(indexes))
        indexes.reverse()  # Reverse the order to avoid label misalignment in deletion loop
        for index in indexes:
            conv_label.pop(index)
            conv_data = np.delete(conv_data, index, axis=0)
            data = np.delete(data, index, axis=0)

        # Task where the target changes between the first and second halves (Change labels to Non-Target=1, Target=2)
        ch_indexes = [
            i2 for i2, x2 in enumerate(conv_label[0:260]) if x2 == 3
        ]  # Index search
        for ch_index in ch_indexes:
            conv_label[ch_index] = int(2)
        ch_indexes = [
            i2 for i2, x2 in enumerate(conv_label[261:520]) if x2 == 1
        ]  # Index search
        for ch_index in ch_indexes:
            conv_label[261 + ch_index] = int(2)
        ch_indexes = [
            i2 for i2, x2 in enumerate(conv_label[261:520]) if x2 == 3
        ]  # Index search
        for ch_index in ch_indexes:
            conv_label[261 + ch_index] = int(1)

        # Create MNE arrays for visualization
        epochs_mne, evoked_mne, evoked_diff, N_label = Functions.make_mne_array(
            data, conv_label, 256
        )

        # evoked_mne[0] = Non-Target
        # evoked_mne[1] = Target
        # evoked_mne2[0] = Non-Target + Standard Stimulus
        # evoked_mne2[1] = Target

        evokeds = [evoked_mne[0], evoked_mne[1]]
        evoked_list.append(evokeds)  # For overall subject average

        epochs_allsub.append(epochs_mne)  # For statistics across all subjects

        # Visualization
        limit = None
        Functions.mne_visualize(
            epochs_mne,
            evokeds,
            evoked_diff,
            userdir,
            limit,
            False,
            ["Non-Target", "Target"],
        )

        colormap_ch = [
            "Fp1",
            "Fpz",
            "Fp2",
            "AF8",
            "AF4",
            "AFz",
            "AF3",
            "AF7",
            "F7",
            "F5",
            "F3",
            "F1",
            "Fz",
            "F2",
            "F4",
            "F6",
            "F8",
            "FT8",
            "FC6",
            "FC4",
            "FC2",
            "FCz",
            "FC1",
            "FC3",
            "FC5",
            "FT7",
            "T7",
            "C5",
            "C3",
            "C1",
            "Cz",
            "C2",
            "C4",
            "C6",
            "T8",
            "TP8",
            "CP6",
            "CP4",
            "CP2",
            "CPz",
            "CP1",
            "CP3",
            "CP5",
            "TP7",
            "P9",
            "P7",
            "P5",
            "P3",
            "P1",
            "Pz",
            "P2",
            "P4",
            "P6",
            "P8",
            "P10",
            "PO8",
            "PO4",
            "POz",
            "PO3",
            "PO7",
            "O1",
            "Oz",
            "O2",
            "Iz",
        ]  # Rearrange channels from frontal to occipital
        # colormap_ch = None  # Uncomment this line if using all channels
        colormap, colormap_timelist = Functions.mne_statistics(
            epochs_mne, evokeds, evoked_diff, userdir, colormap_ch
        )

        Functions.make_heatmap(
            evoked_mne, colormap, colormap_timelist, userdir, colormap_ch
        )

        resampled_data = Functions.Resampling(
            conv_data, 256, 32
        )  # Change the sample rate (study the Nyquist theorem for details)

        average_list = [
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [5, 5],
        ]  # Combinations for averaging

        for average in average_list:
            data_ave, label_ave = Functions.Ntime_Addition_Average(
                resampled_data, conv_label, [1, 2], average
            )
            tuned_parameters = [
                {
                    "C": [
                        0.000001,
                        0.00001,
                        0.0001,
                        0.001,
                        0.01,
                        0.1,
                        1,
                        10,
                        100,
                        1000,
                    ],
                    "kernel": ["linear"],
                }
            ]
            Functions.evaluate_cross_validation(
                data_ave,
                label_ave,
                5,
                userdir,
                tuned_parameters,
                matpath,
                None,
                "balanced_accuracy",
                average,
            )

    Functions.mne_average_visualize(evoked_list, None, False, ["Non-Target", "Target"])
    colormap, colormap_timelist = Functions.mne_statistics_allsub(
        epochs_mne, epochs_allsub, colormap_ch
    )
    Functions.make_heatmap(
        evoked_mne,
        colormap,
        colormap_timelist,
        userdir,
        colormap_ch,
        "statistics_map.pdf",
    )
