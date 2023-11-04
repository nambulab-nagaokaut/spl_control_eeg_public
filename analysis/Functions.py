import scipy.io
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn import *
from sklearn.model_selection import *
from sklearn.metrics import *
import sys
import datetime
import resampy
import mne
import glob
import os
import seaborn as sns
import pandas as pd
import multiprocessing as mp
import statsmodels.stats.multitest as multi

def LoadFilelist(dirpath):
    files = glob.glob("./*.mat")
    return files

def LoadData(path): # Extracting data and labels from MAT files.
    # Loading matfile
    mat = scipy.io.loadmat(path)
    data = mat["DataF"]
    label = mat["Pressure_Label"]
    if ("Group" in mat.keys()) == True: # Check for the presence of the 'Group' key in the MAT file and branch the processing (for handling the previous data format).
        group = mat["Group"]
    else:
        group = None # If the 'Group' key is not present, return None.
    return data, label, group


def Data_Formatting(data, label, ROI_CH=None): # Adjust the concatenation and list structure of CH data for learning purposes.
    # Data formatting
    combined_data = []  # Data storage
    for i in range(len(data)):  # Loop through the number of trials.
        buffer = []
        if ROI_CH == None:
            for t in range(64):  # 64CH concatenation
                buffer.append(data[i, :, t])  # store in buffer
            buffer = list(itertools.chain.from_iterable(buffer))  # If the line above is valid, comment out this line.
            # buffer = buffer[47] # For use with only specific channels.
        elif isinstance(ROI_CH, list):
            for CH in ROI_CH:
                buffer.append(data[i, :, CH])  # Store in a list temporarily
            buffer = list(itertools.chain.from_iterable(buffer))  # If the line above is valid, comment out this line.
        combined_data.append(buffer)
    # Label formatting
    combined_label = list(itertools.chain.from_iterable(label))  # Convert from 2D to 1D (concatenate all elements).

    return combined_data, combined_label

def Label_Select(data, label, label_num, CH): # "Extract data for a specific label and a single channel only (for plotting, etc.)
    # Label formatting
    combined_label = list(itertools.chain.from_iterable(label))  # Convert from 2D to 1D (concatenate all elements).

    # Label selection
    indexes = [i for i, x in enumerate(combined_label) if x == label_num]  # Index search
    print("Label #: " + str(indexes))

    selected_data = []
    for index in indexes:
        selected_data.append(data[index, :, CH])

    return selected_data

def Remove_Label(data_list, label_list, rem_list): # Remove data corresponding to selected labels in the list (rem_list).
    for rem_label in rem_list:
        indexes = [i for i, x in enumerate(label_list) if x == rem_label]  # Index search
        print("Removed label index: " + str(indexes))
        indexes.reverse()  # Reverse the order (from larger index numbers) and ensure labels do not shift during deletion loop
        for index in indexes:
            label_list.pop(index)
            data_list.pop(index)
    return data_list, label_list

def Addition_Average(data_list): # Calculate the mean by adding all elements in the list
    sum_data = 0
    for data in data_list:
        sum_data += data
    average_data = sum_data/len(data_list)

    return average_data

def Resampling(list, before, after): # Resample each element of the array (list, original sample rate, target sample rate). Be cautious as this operation can be particularly computationally intensive within the program

    for item in list:
        Resampled_data.append(resampy.resample(np.array(item), before, after, filter='kaiser_best')) # Usually slow, but setting filter="kaiser_fast" makes the process twice as fast (with a slight decrease in accuracy).
        #Resampled_data.append(scipy.signal.resample(np.array(item), round(len(item)/before*after))) # A version using scipy.signal (several times faster than resampy, but a bit less user-friendly due to specifying sample numbers)
    return Resampled_data

def Change_Label(label_list, change_list, after_label): # Change label numbers (label array, list of original label numbers to be changed, new label numbers)
    for chlabel in change_list:
        ch_indexes = [i2 for i2, x2 in enumerate(label_list) if x2 == chlabel]  # Index search
        for ch_index in ch_indexes:
            label_list[ch_index] = int(after_label)
    return label_list

def FFT(data_list): #
    buffer = []
    for data in data_list:
        buffer.append(np.fft.fft(data))
    return buffer

def Plot_WaveForm(data_list, total_time=None, title=None, savename=None): # Plot all time-series data from a 2D array (data list, time per data [ms] - optional, plot title - optional, image save name - optional).
    fig, ax = plt.subplots()
    if total_time == None: # If no time axis is specified, do not input the x-axis → x-axis becomes the sample numbers.
        for t in data_list:
            ax.plot(t)
        ax.set_xlabel('Samples')
    else:  # If a time axis is specified, regenerate the x-axis.
        for t in data_list:
            ax.plot(np.linspace(0, total_time, len(t)), t)
        ax.set_xlabel('Times [ms]')
    if title != None: # Output the graph title only if the argument 'title' is provided
        plt.title(str(title))
    ax.set_ylabel('Amplitue [μV]')
    if savename != None: # Save as an image only if the 'savename' argument is provided
        plt.savefig(str(savename))
    plt.show()

def Ntime_Addition_Average(data, label, index_num, N): #　Calculate the cumulative mean at any desired number of iterations.
    if all([t == 1 for t in N]) == True: # Do not perform the operation if all cumulative means are 1 (for single trial).
        return data, label
    else:
        processed_data = []
        processed_label = []
        for i in range(len(index_num)):
            # 各ラベルを平均
            indexes1 = [idx1 for idx1, xindx1 in enumerate(label) if xindx1 == index_num[i]]  # Index search
            data1 = []
            for index1 in indexes1:
                data1.append(data[index1])
            data1 = np.array(data1)
            loop1 = int(len(data1) / N[i])
            data_ave1 = []
            for a1 in range(loop1):
                data_ave1.append(np.sum(data1[a1 * N[i]:(a1 + 1) * N[i] - 1], axis=0).tolist())
            label_ave1 = [index_num[i]] * len(data_ave1)
            processed_data.append(data_ave1)
            processed_label.append(label_ave1)
        alldata = processed_data[0] # Prepare only the first element
        alllabel = processed_label[0]
        for index2 in range(len(N)-1): # Concatenate the data (from the second element onward).
            alldata += processed_data[index2+1]
            alllabel += processed_label[index2+1]

        return alldata, alllabel

def traintest_split(data, label, test_size, randomseed=None): # Split the data for training and testing: data, corresponding labels, ratio to allocate to the test data (from 0 to 1), random seed value
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=test_size, random_state=randomseed)

    return X_train, X_test, y_train, y_test # Training data, testing data, training labels, testing labels.

def mne_visualize(epochs_mne, evoked_mne, evoked_diff,userdir ,limit=None, average_ref=False, legends=None): # Visualize the input by averaging across all trials: data array, label array, specific labels to select (list), sample rate.
    if average_ref == True:
        for evoked in evoked_mne:
            evoked.set_eeg_reference('average', projection=False) # Apply average referencing directly instead of using a projector
    if legends == None: # For cases where the plot labels are not specified
        legends = []  # Buffer for plot label names
        for i in range(len(evoked_mne)):  # Create label names in a loop
            legends.append("label" + str(i + 1))

    f = open(userdir+"/Compare-Peaks.txt", "w", encoding='utf-8')  # Write standard output to a text file
    for t in range(len(evoked_mne)): # Execute for each label.
        # Search for peak amplitude values and latency of the Cz channel for each label.
        channel, latency, value = evoked_mne[t].copy().pick_channels(["Cz"]).get_peak(return_amplitude=True)
        latency = float(latency * 1e3)  # convert to milliseconds
        value = float(value * 1e6)  # convert to µV
        Peaks_str = str('Label {}: peak of {} μV at {} ms in channel {}'.format(t, value, latency, channel))
        f.write(Peaks_str + "\n")

        # プロット
        topomap = evoked_mne[t].plot_topomap(times = np.arange(0, 0.96, 0.06), nrows='auto', colorbar=True, show=False) # Topography
        topomap.savefig(userdir+"/Topomap_"+str(legends[t])+".pdf") # Save as an image
        plot_joint = evoked_mne[t].plot_joint(times=[0., 0.15, 0.3, 0.45, 0.6, 0.75, 0.9], picks="eeg", ts_args=limit, show=False) # Synthesis of time series and topography (setting plot range): ts_args = {"xlim": (0, 1), "ylim": dict(eeg=[-20, 20])})
        plot_joint.savefig(userdir+"/Plot_joint_"+str(legends[t])+".pdf") # Save as an image

    if len(evoked_mne) == 2: # Execute only if there are two labels
        plot_joint_diff = evoked_diff.plot_joint(title="Difference between Target and Non-Target",
                                                times=[0., 0.15, 0.3, 0.45, 0.6, 0.75, 0.9], picks="all", show=False)
        plot_joint_diff.savefig(userdir+"/Plot_joint_diff.pdf")

    for params in [(None, "Cz"), ("mean", "eeg")]:
        combine, picks = params
        evokeds = mne.viz.plot_compare_evokeds(dict(zip(legends, evoked_mne)), combine=combine, picks=picks, show=False, show_sensors=None,truncate_yaxis=False)  # By specifying pick="Cz", you can select a single channel. Using picks=eeg includes all channels. Legends are also available
        evokeds[0].savefig(userdir+"/Compare-"+str(picks)+".pdf") # The plot images are returned as a list. In cases where there is only 1 channel or when combine="mean", only one element is returned. Handle this by accessing the first element using [0].
    plt.close("all")  # Close operation (memory release)

def mne_average_visualize(evoked_list, limit=None, average_ref=False, legends=None):
    if len(evoked_list) > 1: # Check if there are two or more sets of data available
        diff_buffer = []
        f = open("result/[Subjects-Average]Compare-Peaks.txt", "w", encoding='utf-8')  # Write standard output to a text file
        for i in range(len(evoked_list[1])): # The length of the first dataset equals the number of labels
            buffer = []
            for evoked in evoked_list:
                buffer.append(evoked[i])

            evoked_average = mne.combine_evoked(buffer, weights="equal")
            # Search for peak amplitude values and latency of the Cz channel for each label
            channel, latency, value = evoked_average.copy().pick_channels(["Cz"]).get_peak(return_amplitude=True)
            latency = float(latency * 1e3)  # convert to milliseconds
            value = float(value * 1e6)  # convert to µV
            Peaks_str = str('Label {}: peak of {} μV at {} ms in channel {}'.format(i, value, latency, channel))
            f.write(Peaks_str+"\n")

            if average_ref == True:
                evoked_average.set_eeg_reference('average', projection=False)

            topomap = evoked_average.plot_topomap(times=np.arange(0, 0.96, 0.06), nrows='auto', colorbar=True,
                                                 show=False)  # Topography
            topomap.savefig("result/[Subjects-Average]Topomap_label"+str(i)+".pdf")
            plot_joint = evoked_average.plot_joint(times=[0., 0.15, 0.3, 0.45, 0.6, 0.75, 0.9], picks="eeg", ts_args=limit, show=False)
            #plot_joint = evoked_average.plot_joint(times=[0.351, 0.335, 0.324, 0.316,0.304], picks="eeg", ts_args=limit, show=False)
            plot_joint.savefig("result/[Subjects-Average]Plot_joint_label"+str(i)+".pdf")
            diff_buffer.append(evoked_average)

        if len(diff_buffer) == 2:  # Execute only if there are two labels
            if legends == None:  # For cases where the plot labels are not specified
                legends = []  # Buffer for plot label names
                for i in range(len(diff_buffer)):  # Create label names in a loop
                    legends.append("label" + str(i + 1))
            else:
                pass
            evoked_diff = mne.combine_evoked(diff_buffer, weights=[-1, 1])
            plot_joint_diff = evoked_diff.plot_joint(title="Difference between Target and Non-Target",
                                                     times=[0., 0.15, 0.3, 0.45, 0.6, 0.75, 0.9], picks="all",
                                                     show=False)
            plot_joint_diff.savefig("result/[Subjects-Average]Plot_joint_diff.pdf")

        else:
            pass
        for params in [(None, "Cz"), ("mean", "eeg")]:
            combine, picks = params
            plt.rcParams["font.size"] = 14

            evoked_label1_data = []
            evoked_label2_data = []
            for subn in range (len(evoked_list)):
                evoked_label1_data.append(evoked_list[subn][0])
                evoked_label2_data.append(evoked_list[subn][1])

            evoked_list_all = [evoked_label1_data, evoked_label2_data]
            evokeds = mne.viz.plot_compare_evokeds(dict(zip(legends, evoked_list_all)), combine=combine, picks=picks, show=False, show_sensors=None, truncate_yaxis=False, ci=0.95) # ci=は信頼区間の設定
            evokeds[0].savefig("result/[Subjects-Average]Compare-" + str(picks) + ".pdf")
        plt.close("all")  # Close operation (memory release)
    else:
        pass

def mann_whitney_u_test(elec, DF1, DF2, alternative="less"): # Mann-Whitney U test
    A = DF1[elec].groupby("epoch").mean()
    B = DF2[elec].groupby("epoch").mean()

    result = scipy.stats.mannwhitneyu(A, B, alternative=alternative) 
    p = result.pvalue
    statistic = result.statistic

    return p, statistic

def wilcoxon(elec, DF1, DF2, alternative="less"): # The Wilcoxon signed-rank test (used only in mne_statistics_allsub)
    A = []
    B = []
    for DF in DF1:
        A.append(DF[elec].mean())
    for DF in DF2:
        B.append(DF[elec].mean())

    result = scipy.stats.wilcoxon(A, B, alternative=alternative)
    p = result.pvalue
    statistic = result.statistic

    return p, statistic

def kruskal(elec, DF_list): # Kruskal-Wallis test (elec: electrodes, DF_list: list containing data for each label)
    DF_list_elec = []
    for df in DF_list:
        DF_list_elec.append(df[elec].groupby("epoch").mean())
    result = scipy.stats.kruskal(*DF_list_elec)
    p = result.pvalue
    statistic = result.statistic

    return p, statistic

def multiple_comparison_correction(color_map, alpha=0.05, method="fdr_bh"): # Redefine the colormap array after performing multiple comparison correction
    color_map_1d = np.reshape(color_map, (1, -1))[0]
    reject, pvals_corrected, alphacSidak, alphacBonf = multi.multipletests(color_map_1d, alpha, method)
    colormap_corrected = np.reshape(pvals_corrected, (50, -1)).tolist()  # Restore to the original shape

    return colormap_corrected

def mne_statistics(epochs_mne, evoked_mne, evoked_diff, userdir, colormap_ch=None, multiprocess=False): # Conduct a test between two labels to examine the significance
    global alternative # Keep as a global variable for later use in the file name
    alternative = "less"  # Setting assumptions
    sys.stdout = open(userdir + "/statistics-"+str(alternative)+".txt", "w")  # Write standard output to a text file
    timelist = [(0, 0.4), (0.4, 0.7), (0, 0.9)]
    print("Cz variance")
    for epochs_label in range(len(epochs_mne)): # Calculate the variance of Cz for each label
        epochs_peak = epochs_mne[epochs_label].copy().to_data_frame(index="epoch")["Cz"].groupby("epoch").max()
        var = np.var(epochs_peak)
        print("Label"+ str(epochs_label) +": "+ str(var))

    print("\ p-value for each channel")
    if len(epochs_mne) == 2: # In the case of two groups
        for time in timelist:
            tmin, tmax = time
            index = ['condition', 'epoch', 'time']
            DF1 = epochs_mne[0].copy().crop(tmin, tmax).to_data_frame(index=index)
            DF2 = epochs_mne[1].copy().crop(tmin, tmax).to_data_frame(index=index)

            #for elec in ["Cz", "Pz"]:
            for elec in evoked_mne[0].info['ch_names']:
                p, statistic = mann_whitney_u_test(elec, DF1, DF2, alternative)  # Wilcoxon signed-rank test

                # Display the results
                format_dict = dict(elec=elec, tmin=tmin, tmax=tmax,
                                   df=len(epochs_mne[0].events) - 2, p=p, statistic=statistic)
                report = "{elec}, time: {tmin}-{tmax} s;, p={p:.20f}, statistic={statistic:.10f}"
                print(report.format(**format_dict))
            print("\n----------------------------\n")

        # Colormap creation
        colormap_timelist = np.linspace(0, 1, 51)
        colormap_timelist_copy = colormap_timelist.copy()
        colormap_buffer = []
        for i in range(len(colormap_timelist)-1):
            tmin, tmax = colormap_timelist[0], colormap_timelist[1]
            colormap_timelist = np.delete(colormap_timelist, 0)
            index = ['condition', 'epoch', 'time']
            DF1 = epochs_mne[0].copy().crop(tmin, tmax).to_data_frame(index=index)
            DF2 = epochs_mne[1].copy().crop(tmin, tmax).to_data_frame(index=index)

            electrode_buffer = []
            if colormap_ch == None:
                colormap_elec = evoked_mne[0].info['ch_names']
            else:
                colormap_elec = colormap_ch

            if multiprocess == False: # When not parallelized (currently faster in this scenario)
                for elec in colormap_elec:
                    p, statistic = mann_whitney_u_test(elec, DF1, DF2, alternative) # Wilcoxon signed-rank test

                    electrode_buffer.append(p)
                colormap_buffer.append(electrode_buffer)
            elif multiprocess == True: # When parallelizing (usually discouraged as the overhead associated with parallelization often outweighs the benefits)
                pool = mp.Pool(os.cpu_count()) # Parallelize using the same number of threads as the CPU's thread count
                colormap_buffer.append(pool.starmap_async(mann_whitney_u_test, [(elec, DF1, DF2) for elec in colormap_elec]).get()) # 非同期処理(ループは内包表記)
                #pool.close()
                #pool.join()

    elif len(epochs_mne) > 2: # In the case of three groups
        alternative = "" # Redefine the filename to prevent issues.
        for time in timelist:
            DF_list = []
            tmin, tmax = time
            index = ['condition', 'epoch', 'time']
            for df in epochs_mne:
                DF_list.append(df.copy().crop(tmin, tmax).to_data_frame(index=index))

            # for elec in ["Cz", "Pz"]:
            for elec in evoked_mne[0].info['ch_names']:
                #  Kruskal-Wallis test (non-parametric, independent samples, for three or more groups)
                p, statistic = kruskal(elec, DF_list)

                # Display result
                format_dict = dict(elec=elec, tmin=tmin, tmax=tmax,
                                   df=len(epochs_mne[0].events) - 2, p=p, statistic=statistic)
                report = f"{elec}, time: {tmin}-{tmax} s;, p={p:.20f}, statistic={statistic:.10f}"
                print(report.format(**format_dict))
            print("\n----------------------------\n")

        # Color map creation
        colormap_timelist = np.linspace(0, 1, 51)
        colormap_timelist_copy = colormap_timelist.copy()
        colormap_buffer = []
        for i in range(len(colormap_timelist) - 1):
            tmin, tmax = colormap_timelist[0], colormap_timelist[1]
            colormap_timelist = np.delete(colormap_timelist, 0)
            index = ['condition', 'epoch', 'time']
            DF_list = []
            for df in epochs_mne:
                DF_list.append(df.copy().crop(tmin, tmax).to_data_frame(index=index))

            electrode_buffer = []
            if colormap_ch == None:
                colormap_elec = evoked_mne[0].info['ch_names']
            else:
                colormap_elec = colormap_ch
            # for elec in evoked_mne[0].info['ch_names']:
            for elec in colormap_elec:
                # Kruskal-Wallis test
                p, statistic = kruskal(elec, DF_list)

                electrode_buffer.append(p)
            colormap_buffer.append(electrode_buffer)

    else: # Do nothing if there are only groups and no other conditions
        pass
    sys.stdout.close()
    sys.stdout = sys.__stdout__

    return colormap_buffer, colormap_timelist_copy

def mne_statistics_allsub(epochs_mne, epochs_allsub, colormap_ch=None, multiprocess=False): # Conduct the test by averaging within-subject data
    global alternative
    alternative = "less"  # Setting assumptions
    sys.stdout = open("result/statistics-"+str(alternative)+".txt", "w")  # Write standard output to a text file
    timelist = [(0, 0.4), (0.4, 0.7), (0, 0.9)]
    for time in timelist:
        tmin, tmax = time
        DF1 = []
        DF2 = []
        for epochs in epochs_allsub:
            DF1.append(epochs[0].copy().crop(tmin, tmax).average().to_data_frame())
            DF2.append(epochs[1].copy().crop(tmin, tmax).average().to_data_frame())

        for elec in epochs_mne[0].info['ch_names']:
            p, statistic = wilcoxon(elec, DF1, DF2, alternative)  # Wilcoxon signed-rank test

            # Display the results
            format_dict = dict(elec=elec, tmin=tmin, tmax=tmax,
                               df=len(epochs_mne[0].events) - 2, p=p, statistic=statistic)
            report = "{elec}, time: {tmin}-{tmax} s;, p={p:.20f}, statistic={statistic:.10f}"
            print(report.format(**format_dict))
        print("\n----------------------------\n")

    # Color map creation
    colormap_timelist = np.linspace(0, 1, 51)
    colormap_timelist_copy = colormap_timelist.copy()
    colormap_buffer = []
    for i in range(len(colormap_timelist)-1):
        tmin, tmax = colormap_timelist[0], colormap_timelist[1]
        colormap_timelist = np.delete(colormap_timelist, 0)
        DF1 = []
        DF2 = []
        for epochs in epochs_allsub:
            DF1.append(epochs[0].copy().crop(tmin, tmax).average().to_data_frame())
            DF2.append(epochs[1].copy().crop(tmin, tmax).average().to_data_frame())

        electrode_buffer = []
        if colormap_ch == None:
            colormap_elec = epochs_mne[0].info['ch_names']
        else:
            colormap_elec = colormap_ch

        if multiprocess == False: # When not parallelized (currently faster in this scenario)
            for elec in colormap_elec:
                p, statistic = wilcoxon(elec, DF1, DF2, alternative) # ウィルコクソン符号付き順位検定

                electrode_buffer.append(p)
            colormap_buffer.append(electrode_buffer)
        elif multiprocess == True: # When parallelizing (usually discouraged as the overhead associated with parallelization often outweighs the benefits)
            pool = mp.Pool(os.cpu_count()) # Parallelize using the same number of threads as the CPU's thread count
            colormap_buffer.append(pool.starmap_async(mann_whitney_u_test, [(elec, DF1, DF2) for elec in colormap_elec]).get()) # Asynchronous processing (loops in list comprehensions)
            #pool.close()
            #pool.join()

        sys.stdout.close()
        sys.stdout = sys.__stdout__

    return colormap_buffer, colormap_timelist_copy


def make_heatmap(evoked_mne, colormap_list, colormap_timelist, userdir, colormap_ch=None, filename=None):
    # Convert to a DataFrame and then plot as a heatmap
    df = pd.DataFrame(np.array(colormap_list).T)
    df.columns = np.round(colormap_timelist[0:-1], 2)
    if colormap_ch == None:
        df.index = evoked_mne[0].info['ch_names']
    else:
        df.index = colormap_ch
    #df.index = np.arange(1, 65) # in case of channel number
    sns.heatmap(df, vmin=0, vmax=0.15, cmap="jet_r", xticklabels=5, yticklabels=4,
                cbar_kws={"label": "p-value"}) # heatmap using seaborn
    plt.xlabel("Time [s]")
    plt.ylabel("Electrode position")
    #plt.ylabel("Channel")

    if filename == None: # Usually, the filename argument is not necessary
        plt.savefig(userdir + "/statistics_map-"+str(alternative)+".pdf")
    else:
        plt.savefig("./result/"+filename)
    plt.clf()  # Initialize the plot


def make_mne_array(data_list, label_list, sfreq):
    label_set = sorted(list(set(label_list))) # Labels sorted in ascending order without duplicates
    N_label = len(label_set)
    databuffer = []
    data_reshape = np.rollaxis(data_list, 2, 1)  # Reshape into a multidimensional shape
    for label_index in label_set:
        #indexbuffer.append([i for i, x in enumerate(label_list) if x == label_index]) # Index search
        indexes = [i for i, x in enumerate(label_list) if x == label_index]  # Index search
        print("ラベル"+str(label_index)+": " + str(indexes))
        buffer = []
        for index in indexes:
            buffer.append(np.array(data_reshape[index, :, :]) / 10e5) # 10e5 is a bias
        databuffer.append(buffer)
    ch_types = ['eeg'] * 64
    montage = mne.channels.make_standard_montage("biosemi64", head_size='auto')  # Used channel information
    info = mne.create_info(ch_names=montage.ch_names, sfreq=sfreq, ch_types=ch_types)  # Generating property information
    # Creating various arrays
    epochs_mne = []
    evoked_mne = []
    for index in range(N_label):
        epochs = mne.EpochsArray(databuffer[index], info)
        epochs_mne.append(epochs.set_montage(montage))
        evoked_mne.append(epochs.average().set_montage(montage))
    if N_label == 2:
        evoked_diff = mne.combine_evoked(evoked_mne, weights=[-1, 1]).set_montage(montage)
    else:
        evoked_diff = None

    return epochs_mne, evoked_mne, evoked_diff, N_label



# Buffer for score calculation
originalclass = []
predictedclass = []
def classification_report_with_accuracy_score(y_true, y_pred): # Append data to the buffer.
    originalclass.extend(y_true)
    predictedclass.extend(y_pred)
    return accuracy_score(y_true, y_pred) # return accuracy score


def evaluate_cross_validation(x, y, K, userdir, gs_params, matpath="Unknown", rem_list="None", gs_scoring="accuracy", average = None): # Classification using Nested Cross-Validation with text output for results
    # Display and save scores
    sys.stdout = open(userdir+"/score-ave"+str(average) + ".txt", "w")  # Write standard output to a text file

    print("Matfile: " + str(matpath))
    print()
    print("Removed_label: " + str(rem_list))
    print("")
    clf = GridSearchCV( # Grid search configuration
        svm.SVC(),  # Discriminator (svm.SVC())
        gs_params,  # Set of parameters to be optimized
        cv=4,  # Number of cross-validation folds
        n_jobs=10,  # Number of parallel jobs (-1 for using all available cores)
        scoring=gs_scoring,  # Specifying the evaluation metric for the model
        verbose=4)   # or displaying progress information
    cv = KFold(K, shuffle=True, random_state=0)
    scores = cross_validate(clf, x, y, cv=cv, scoring=make_scorer(classification_report_with_accuracy_score), verbose=0) # Evaluate in alignment with the classification report
    print(scores)
    print("")
    print(classification_report(originalclass, predictedclass)) # Calculate the average score for each fold in K-Fold
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    originalclass.clear()
    predictedclass.clear()


def search_mat(matdir):
    files = glob.glob(matdir+"/*.mat")
    for file in files:
        print(file)
        userpath = "./result/" + str(file).replace('.mat', '').replace(matdir, '')
        os.makedirs(userpath, exist_ok=True)
    return files


if __name__ == '__main__': # Prevent the following from executing when the module is imported
    exit()
