# -*- coding: utf-8 -*-

#### Libs ####
import pyedflib
import numpy as np
import pandas as pd
import commands
import pywt
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt
from pywt import WaveletPacket
import pywt.data
from detect_peaks import detect_peaks
import os
from sys import platform
from scipy.signal import welch
#%matplotlib notebook

peaks_rms = [ ]
norm = [ ]
debug = False
#### Filters ####
def FilterSignal(emg, low_pass=10., sfreq=2000., high_band=2., low_band=500.):
    """
    emg: EMG data
    high: high-pass cut off frequency
    low: low-pass cut off frequency
    sfreq: sampling frequency
    """

    # normalise cut-off frequencies to sampling frequency
    high_band = high_band/(sfreq/2)
    low_band = low_band/(sfreq/2)

    # create bandpass filter for EMG
    b1, a1 = sp.signal.butter(4, [high_band,low_band], btype='bandpass')

    # process EMG signal: filter EMG
    emg_filtered = sp.signal.filtfilt(b1, a1, emg)

    # process EMG signal: rectify
    emg_rectified = emg_filtered
    #emg_rectified = abs(emg_filtered)

    # create lowpass filter and apply to rectified signal to get EMG envelope
    low_pass = low_pass/sfreq
    b2, a2 = sp.signal.butter(4, low_pass, btype='lowpass')
    emg_envelope = sp.signal.filtfilt(b2, a2, emg_rectified)

    #return emg_envelope
    return emg_filtered

#### Plot FFT ###
def PlotFft(Xc, F, fig_name=False):
    #Xc = dfn['Ch1']
    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
    X = np.abs(np.fft.fft(Xc[:, 0]))
    fs = np.linspace(0, F,Xc[:, 0].shape[0])
    ax1.plot(fs,X, linewidth=0.2, color = '#B22222')
    ax1.set_title('Sinal FFT'+'\nCanal 1')
    ax1.set_ylabel('EMG (u.v.)')
    ax1.grid()
    plt.rc('grid', linestyle="dotted", color='black')

    X = np.abs(np.fft.fft(Xc[:, 1]))
    fs = np.linspace(0, F,Xc[:, 1].shape[0])
    ax2.plot(fs,X, linewidth=0.2, color = '#008000')
    ax2.set_title('\nCanal 2')
    ax2.set_ylabel('EMG (u.v.)')
    ax2.grid()
    plt.rc('grid', linestyle="dotted", color='black')

    X = np.abs(np.fft.fft(Xc[:, 2]))
    fs = np.linspace(0, F,Xc[:, 2].shape[0])
    ax3.plot(fs,X, linewidth=0.2)
    ax3.set_title('\nCanal 3')
    ax3.set_ylabel('EMG (u.v.)')
    ax3.grid()
    plt.rc('grid', linestyle="dotted", color='black')

    X = np.abs(np.fft.fft(Xc[:, 3]))
    fs = np.linspace(0, F,Xc[:, 3].shape[0])
    ax4.plot(fs,X, linewidth=0.2, color = '#FF8C00')
    ax4.set_title('\nCanal 4')
    ax4.set_ylabel('EMG (u.v.)')
    ax4.set_xlabel('Amostras')
    ax4.grid()
    plt.rc('grid', linestyle="dotted", color='black')

    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    f.set_size_inches(w=10,h=8)

    if fig_name!=False:
        f.savefig('classificar/preproc_figs/signal_fft')

#### Plot ####
def PlotFile(dfn, fig_name=False, tSinal='Puro', fill=False, debug=False, lw=0.2):

    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
    fs = np.arange(len(dfn))

    if debug == True:
        print np.shape(fs)

    ax1.plot(dfn['Ch1'][:], linewidth=lw, color = '#B22222')
    ax1.set_title('Sinal '+tSinal+'\nCanal 1')
    ax1.set_ylabel('EMG (u.v.)')
    ax1.grid()
    plt.rc('grid', linestyle="dotted", color='black')

    ax2.plot(dfn['Ch2'][:], linewidth=lw, color = '#008000')
    ax2.set_title('\nCanal 2')
    ax2.set_ylabel('EMG (u.v.)')
    ax2.grid()
    plt.rc('grid', linestyle="dotted", color='black')

    ax3.plot(dfn['Ch3'][:], linewidth=lw)
    ax3.set_title('\nCanal 3')
    ax3.set_ylabel('EMG (u.v.)')
    ax3.grid()
    plt.rc('grid', linestyle="dotted", color='black')

    ax4.plot(dfn['Ch4'][:], linewidth=lw, color = '#FF8C00')
    ax4.set_title('\nCanal 4')
    ax4.set_ylabel('EMG (u.v.)')
    ax4.set_xlabel('Amostras')
    ax4.grid()
    plt.rc('grid', linestyle="dotted", color='black')
    if debug == True:
        print (np.shape(dfn))

    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    f.set_size_inches(w=10,h=8)

    if fill==True:
        ax1.fill_between(fs,0,dfn['Ch1'][:], color = '#B22222', alpha=0.3)
        ax2.fill_between(fs,0,dfn['Ch2'][:], color = '#008000', alpha=0.3)
        ax3.fill_between(fs,0,dfn['Ch3'][:], alpha=0.3)
        ax4.fill_between(fs,0,dfn['Ch4'][:], color = '#FF8C00', alpha=0.3)

    if fig_name!=False:
        f.savefig('classificar/preproc_figs/'+fig_name)

#### RMS ####
def Rms(file_):

    #Calcula RMS
    def rms(a, window_size):
        def subrms(a, window_size):
            a2 = np.power(a,2)
            window = np.ones(window_size)/float(window_size)
            return np.sqrt(np.convolve(a2, window, 'valid'))

        for i in range(a.shape[1]):
            if i == 0:
                Xc = subrms(a[:, i], window_size)
            else:
                Xcp = subrms(a[:, i], window_size)
                Xc = np.vstack((Xc,Xcp))
        return np.transpose(Xc)

    Xc = np.load('classificar/partitioned/'+file_)

    df = pd.DataFrame(Xc)

    #Nomeia cada coluna "canal"
    df = df.rename(columns={0: 'Ch1'})
    df = df.rename(columns={1: 'Ch2'})
    df = df.rename(columns={2: 'Ch3'})
    df = df.rename(columns={3: 'Ch4'})

    dfa = df.as_matrix()

    #Aplica RMS em cada canal
    rms = rms(dfa,50)
    rms = pd.DataFrame(data=rms, columns=['Ch1','Ch2','Ch3','Ch4'])

    return rms

#### Find, select and split files on "classificar" folder ####
def sectioner(file_):

    #Lê dados dos arquivos .edf
    file_name = pyedflib.EdfReader('classificar/pure/'+file_)
    n = file_name.signals_in_file
    signal_labels = file_name.getSignalLabels()
    sigbufs = np.zeros((n, file_name.getNSamples()[0]))
    for j in np.arange(n):
         sigbufs[j, :] = file_name.readSignal(j)
    file_name._close()

    suf = 0
    Xc = np.transpose(sigbufs)
    Xc = Xc[4000:14000]
    qtd_div = 3
    brk_div = len(Xc)/qtd_div

    for i in range(0,qtd_div*brk_div,brk_div):
        suf += 1
        np.save('classificar/partitioned/'+file_[:-3]+str(suf), Xc[i:i+brk_div], allow_pickle=False)

#### Calucula a Norma de cada canal ####
def Norm(file_):

    #Calcula RMS
    def rms(a, window_size):
        def subrms(a, window_size):
            a2 = np.power(a,2)
            window = np.ones(window_size)/float(window_size)
            return np.sqrt(np.convolve(a2, window, 'valid'))

        for i in range(a.shape[1]):
            if i == 0:
                Xc = subrms(a[:, i], window_size)
            else:
                Xcp = subrms(a[:, i], window_size)
                Xc = np.vstack((Xc,Xcp))
        return np.transpose(Xc)

    Xc = np.load('classificar/partitioned/'+file_)

    df = pd.DataFrame(Xc)

    #Nomeia cada coluna "canal"
    df = df.rename(columns={0: 'Ch1'})
    df = df.rename(columns={1: 'Ch2'})
    df = df.rename(columns={2: 'Ch3'})
    df = df.rename(columns={3: 'Ch4'})

    #dfa = df.as_matrix(columns=df.columns[1:])
    dfa = df.as_matrix()

    #Aplica RMS em cada canal
    rms = rms(dfa,50)
    rms = pd.DataFrame(data=rms, columns=['Ch1','Ch2','Ch3','Ch4'])

    #Calcula o pico RMS máximo de cada canal em cada arquivo e adiciona a lista peaks_rmslll
    return ([max(rms['Ch1']),max(rms['Ch2']),max(rms['Ch3']),max(rms['Ch4']),])

#### Aplica a norma em cada amostra e salva ####
def Std(file_, norm, debug=False):
    Xc = np.load('classificar/partitioned/'+file_)

    df = pd.DataFrame(Xc)
    dfa = df.as_matrix()
    if debug == True:
        print np.shape(norm)
    #Aplica a norma em cada canal
    Xc=np.array([dfa[:,0]/norm[0],dfa[:,1]/norm[1],dfa[:,2]/norm[2],dfa[:,3]/norm[3]])
    Xc = np.transpose(Xc)
    #Salva o arquivo normalizado na pasta <preproc>
    np.save('classificar/preproc/'+file_[:-4]+'_std', Xc, allow_pickle=False)
    if debug == True:
        print file_[:-4]+'_std.npy'

    return(pd.DataFrame(data=Xc, columns=['Ch1','Ch2','Ch3','Ch4']))

def preProcessing(debug=False):
    #########################
    if platform == "linux" or platform == "linux2":
        files = os.listdir('classificar/pure')
    elif platform == "win32":
        files = os.listdir('C:\Users\\'+os.getenv('username')+'\Documents\GitHub\PKS_ML\classificar\pure')

    for i in files:
        sectioner(i)
    #########################

    #########################
    if platform == "linux" or platform == "linux2":
        files = os.listdir('classificar/partitioned')
    elif platform == "win32":
        files = os.listdir('C:\Users\\'+os.getenv('username')+'\Documents\GitHub\PKS_ML\classificar\partitioned')

    c_pure = np.load('classificar/partitioned/'+files[0][:-4]+'.npy')
    dfn_pure = pd.DataFrame(data=c_pure, columns=['Ch1','Ch2','Ch3','Ch4'])
    PlotFile(dfn_pure,'signal_zero','Puro')
    #########################

    #########################
    peaks_rms = [ ]
    norm = [ ]

    #Cria a lista de picos RMS máximo
    for i in files:
        peaks_rms.append(Norm(i))

    #Calcula o máximo global
    peaks_rms = pd.DataFrame(data=peaks_rms, columns=['Ch1','Ch2','Ch3','Ch4'])
    norm = [max(peaks_rms['Ch1']),max(peaks_rms['Ch2']),max(peaks_rms['Ch3']),max(peaks_rms['Ch4'])]

    if debug == True:
        print 'Norma de cada canal: \n',norm
    #########################

    #########################
    for i in files:
        Std(i,norm,debug)

    c_std = np.load('classificar/preproc/'+files[0][:-4]+'_std.npy')
    df_std = pd.DataFrame(data=c_std, columns=['Ch1','Ch2','Ch3','Ch4'])
    PlotFile(df_std,'signal_std','Normalizado')
    #########################

    #########################
    for i in files:
        Xc = Rms(i)
        np.save('classificar/preproc/'+i[:-4]+'_rms', Xc, allow_pickle=False)
        if debug == True:
            print i[:-4]+'_rms.npy'

    c_rms = np.load('classificar/preproc/'+files[0][:-4]+'_rms.npy')
    df_rms = pd.DataFrame(data=c_rms, columns=['Ch1','Ch2','Ch3','Ch4'])
    PlotFile(df_rms,'signal_rms','RMS',True,0.5)
    #########################

    #########################
    for i in files:
        c = np.load('classificar/preproc/'+i[:-4]+'_std.npy')
        df = pd.DataFrame(data=c, columns=['Ch1','Ch2','Ch3','Ch4'])
        #Nomeia cada coluna "canal"
        df = df.rename(columns={0: 'Ch1'})
        df = df.rename(columns={1: 'Ch2'})
        df = df.rename(columns={2: 'Ch3'})
        df = df.rename(columns={3: 'Ch4'})

        Xc = df.copy()
        Xc['Ch1'] = FilterSignal(df['Ch1'], low_pass=2)
        Xc['Ch2'] = FilterSignal(df['Ch2'], low_pass=2)
        Xc['Ch3'] = FilterSignal(df['Ch3'], low_pass=2)
        Xc['Ch4'] = FilterSignal(df['Ch4'], low_pass=2)

        np.save('classificar/preproc/'+i[:-4]+'_filt', Xc[:-49], allow_pickle=False)
        if debug == True:
            print i[:-4]+'_filt.npy'

    c_filt = np.load('classificar/preproc/'+files[0][:-4]+'_filt.npy')
    df_filt = pd.DataFrame(data=c_filt, columns=['Ch1','Ch2','Ch3','Ch4'])
    PlotFile(df_filt,'signal_filt','Filtrado')
    #########################

    #########################
    for i in files:
        c_fft = np.load('classificar/preproc/'+i[:-4]+'_std.npy')
        #X = np.abs(np.fft.fft(Xc[:, 0]))
        df_fft = pd.DataFrame(data=c_fft, columns=['Ch1','Ch2','Ch3','Ch4'])
        #Nomeia cada coluna "canal"
        df_fft = df_fft.rename(columns={0: 'Ch1'})
        df_fft = df_fft.rename(columns={1: 'Ch2'})
        df_fft = df_fft.rename(columns={2: 'Ch3'})
        df_fft = df_fft.rename(columns={3: 'Ch4'})

        Xc_fft = df_fft.copy()
        Xc_fft['Ch1'] = np.abs(np.fft.fft(df_fft['Ch1']))
        Xc_fft['Ch2'] = np.abs(np.fft.fft(df_fft['Ch2']))
        Xc_fft['Ch3'] = np.abs(np.fft.fft(df_fft['Ch3']))
        Xc_fft['Ch4'] = np.abs(np.fft.fft(df_fft['Ch4']))

        np.save('classificar/preproc/'+i[:-4]+'_fft', Xc_fft[:-49], allow_pickle=False)
        if debug == True:
            print i[:-4]+'_fft.npy'

    c_fft = np.load('classificar/preproc/'+files[0][:-4]+'_std.npy')
    PlotFft(c, 50)
    #########################

    #########################
