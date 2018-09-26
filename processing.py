# -*- coding: utf-8 -*-

#### Libs ####
import os
import pandas as pd
import numpy as np
import scipy as sp
import itertools
import heapq
import matplotlib.pyplot as plt
import ml_time_series as mls
import timeit
from scipy import signal
from datetime import datetime
from sys import platform
from itertools import cycle
from scipy import interp
from sklearn import preprocessing
from sklearn import utils
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.externals import joblib
DATAv_hs = []
DATAv = []
y_validate = []
debug = False

#### Stack Horizontal das features ####
def Stacker(files_,t):

    #Lê dados do arquivo e remove janela
    Xc = np.load('classificar/preproc/'+files_[len(files_)-1])
    Xc = Xc[:-49]

    #Aplica Hstack nos dados
    for j in range(0,len(files_)-1):
        Xci = np.load('classificar/preproc/'+files_[j])
        Xc = np.hstack((Xc.reshape(Xc.shape),Xci.reshape(Xci.shape)))

    np.save('classificar/preproc/stacked_v/'+t+'_hs', Xc, allow_pickle=False)
    return (t+'_hs.npy')


#### Função ProccessData(): Aplica PCA, rotula e salva a amostra ####
def ProccessData(x,DATA,LABEL,debug=False):

    Xc = np.load("classificar/preproc/stacked_v/"+x)

    if debug == True:
        print '\nXc initial shape ', Xc.shape

    #Principal component analysis
    pca = PCA(n_components=8)
    pca.fit(Xc)
    Xc = pca.transform(Xc)
    if debug == True:
        print pca.explained_variance_ratio_
    #entropy.append(sum(pca.explained_variance_ratio_))
    if debug == True:
        print 'Xc PCA shape ', Xc.shape

    #Labeling the PKS level
    C = (np.ones(len(Xc))*LABEL).reshape((len(Xc),1))
    Xc = np.hstack((Xc.reshape(Xc.shape),C.reshape((len(Xc),1))))
    if debug == True:
        print 'Xc labeled shape ', Xc.shape

    # Salving in file on the folder </preproc/labeled>
    np.save('classificar/preproc/labeled_v/'+x[:9]+'_tsl', Xc, allow_pickle=False)
    if debug == True:
        print '\n'+x[:9]+'_tsl'
    return (x[:9]+'_tsl.npy')



#### Calcula e plota a curva ROC e AUC ####
def ROCCurve(y_pred, y_real, clf):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for k in range(1,5,1):

        yy_pred = np.array([0 if int(el)!=k else 1 for el in y_pred])
        yy_valid = np.array([0 if int(el)!=k else 1 for el in y_real])

        fpr[k], tpr[k], _ = roc_curve(yy_pred, yy_valid)
        roc_auc[k] = auc(fpr[k], tpr[k])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(yy_pred.ravel(), yy_valid.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    n_classes = len(np.unique(y_validate))
    lw = 2
    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(1,5,1)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(1,5,1):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves

    def plot_roc_c():
        params = {'text.usetex': True,'text.latex.unicode': True}
        plt.rcParams.update(params)
        plt.plot(fpr["micro"], tpr["micro"],
                 label=u'micro-média da curva ROC (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='m', linestyle=':', linewidth=6)

        plt.plot(fpr["macro"], tpr["macro"],
                 label=u'macro-média da curva ROC (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=6)

        colors = cycle(['aqua', 'firebrick', 'darkorange', 'chartreuse'])

        for i, color in zip(range(1,5,1), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=u'curva ROC da classe N-{0} (área = {1:0.2f})'
                     ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel(u'Taxa de Falsos Positivos')
        plt.ylabel(u'Taxa de Verdadeiros Positivos')
        #plt.title('Some extension of Receiver operating characteristic to multi-class')
        #plt.title(clf)
        plt.legend(loc="lower right")

    fig = plt.figure()
    plot_roc_c()
    fig.set_size_inches(w=7,h=6)
    fig_name = 'classificar/CF_figs/ROC_'+clf+'.png'
    fig.savefig(fig_name)
    plt.show()


#### Calcula matriz de confusão e métricas ####
def Metrics(y_pred, y_real, clf, nivel):

    class_names = np.array(['N-1', 'N-2', 'N-3', 'N-4'])

    yvalid = y_real

    from sklearn.metrics import confusion_matrix

    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion de matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        params = {'text.usetex': True,'text.latex.unicode': True}
        plt.rcParams.update(params)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        #plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)


        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Matriz de confusão normalizada")
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, ('%.3f' % cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

        else:
            print('Matriz de confusão, sem normalização')
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

        print(cm)

        plt.tight_layout()
        plt.ylabel('Valores reais')
        plt.xlabel('Valores preditos')


    # Compute confusion matrix
    cm = confusion_matrix(yvalid, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    #fig = plt.figure()
    #plot_confusion_matrix(cm, classes=class_names)#, title=u'Matriz de confusão -'+clf)
    #fig.set_size_inches(w=7,h=6)

    # Plot normalized confusion matrix
    #fig = plt.figure()
    #plot_confusion_matrix(cm, classes=class_names, normalize=True)#,
                          #title=u'Matriz de confusão - '+clf)
    #fig_name = 'classificar/CF_figs/cm_'+clf+'.png'
    #fig.set_size_inches(w=7,h=6)
    #fig.savefig(fig_name)

    #plt.show()

    #Plota Probabilidade
    def plotProbLevel(cm, classes, title='Confusion de matrix', cmap=plt.cm.Blues):

        fig = plt.figure()
        params = {'text.usetex': True,'text.latex.unicode': True}
        cmap=plt.cm.Blues
        plt.rcParams.update(params)
        plt.imshow(cm,  cmap=cmap)
        #plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        y = np.array([''])
        plt.yticks(range(len(y)), y)
        #plt.yticks(tick_marks, 1)
        thresh = cm.max() / 2.

        for i in (range(cm.shape[1])):
            plt.text(i,0.05, ('%.2f' % cm[0][i]), horizontalalignment="center", color="white" if cm[0][i] > thresh else "black")

        plt.tight_layout()
        #plt.ylabel('Valores reais')
        plt.xlabel(u'Probabilidade de Níveis')

        fig_name = 'classificar/CF_figs/PL.png'
        fig.set_size_inches(w=10,h=8)
        fig.savefig(fig_name)

    np.seterr(divide='ignore', invalid='ignore')
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = [i * 100 for i in cm]
    pl = np.array([cm[nivel -1]])
    plotProbLevel(pl, classes=class_names)
###################

###################
def processing(nivel,debug=False):
    # Processamento dos dados
    ## Cria array com nome dos arquivos de amostra
    if platform == "linux" or platform == "linux2":
        files = os.listdir('classificar/partitioned')
    elif platform == "win32":
        files = os.listdir('C:\Users\\'+os.getenv('username')+'\Documents\GitHub\PKS_ML\classificar\partitioned')

    files_ = []
    for i in files:
        files_.append([i[:-4]+'_filt.npy',i[:-4]+'_rms.npy',i[:-4]+'_fft.npy',i[:-4]+'_std.npy'])

    list.sort(files)
    ###################

    ###################
    ## Cria array com conjunto de  total (treinamento e teste) e conjunto de validação
    DATAv_hs = []

    for i in range(len(files_)):
        DATAv_hs.append(Stacker(files_[i],files_[i][0][:9]))

    ## Ordena os arrays DATA_hs e DATAv_hs de forma crescente
    list.sort(DATAv_hs)
    DATAv_hs

    ## Mostra a forma da amostra para simples conferência
    Xc = np.load('classificar/preproc/stacked_v/'+DATAv_hs[0])
    dfn = pd.DataFrame(data=Xc)
    dfn.head()
    Xc.shape
    ###################

    ###################
    ## Aplica função ProccessData() no conjunto de validação
    entropy = []
    l = [1,1,1]
    l = [i * nivel for i in l]
    #l=[2,2,2, 3,3,3, 4,4,4, 1,1,1, 1,1,1, 2,2,2, 3,3,3, 4,4,4, 1,1,1, 1,1,1]
    if debug == True:
        print ('vetor de rotulos',l)
    DATAv = []

    for x, y in zip(DATAv_hs, l):
        DATAv.append(ProccessData(x,DATAv,y,debug))
    ###################

    ###################
    ## Vertical Stack dos dados e separa os dados como matriz X e rótulos como vetor y do conjuto de validação
    cdata=0
    #Separando set de dados X, e set de labels y - Validação
    Xc = np.load("classificar/preproc/labeled_v/"+DATAv[0])
    for i in DATAv[1:]:
        Xc = np.vstack((Xc,np.load('classificar/preproc/labeled_v/'+i)))
        cdata += 1
        if debug == True:
            print ("Vstacking DATA test on X: %i de %i" % (cdata, len(DATAv)))

    X_validate = Xc[:,0:(Xc.shape[1]-1)]

    if debug == True:
        print "Creating labels vector ..."

    yz = Xc[:,[(Xc.shape[1]-1)]]
    y_validate = np.array([])
    for i in range(len(yz)):
        y_validate = np.hstack((y_validate,yz[i]))

    X_validate.shape, y_validate.shape
    np.unique(y_validate)
    ###################

    ###################
    ## Carrega variáveis
    X_train = np.load('model/X_train.npy')
    X_train_std = np.load('model/X_train_std.npy')
    X_test_std = np.load('model/X_test_std.npy')
    X_validate_std0 = np.load('model/X_validate_std.npy')
    y_train = np.load('model/y_train.npy')
    y_test = np.load('model/y_test.npy')
    y_validate0 = np.load('model/y_validate.npy')

    ## Aplica normalizador e retorna o Shape de cada um
    sc = StandardScaler()
    sc.fit(X_train)
    X_validate_std = sc.transform(X_validate)
    if debug == True:
        print X_validate_std.shape, y_validate.shape
    ###################

    ###################
    ## Carrega classificador de um arquivo
    rfc = joblib.load('model/rfc.pkl')
    rfc_y_pred_v0 = rfc.predict(X_validate_std0)
    rfc_y_pred_v = rfc.predict(X_validate_std)

    if debug == True:
        print ('ClassifyRF accuracy:---------->%.2f %%' % (accuracy_score(rfc_y_pred_v0, y_validate0)*100))
        print''
        print ('ClassifyRF accuracy:---------->%.2f %%' % (accuracy_score(rfc_y_pred_v, y_validate)*100))
    ###################

    ###################
    ## Calcula e plota a curva ROC e AUC
    #ROCCurve(rfc_y_pred_v, y_validate, 'RF')
    ###################

    ###################
    ## Calcula e plota a martriz de confusão e métricas
    print ('A probabilidade de cada nível é:')
    Metrics(rfc_y_pred_v, y_validate, 'RF', nivel)

    ## Apaga arquivos gerados
    if platform == "linux" or platform == "linux2":
        dFiles = os.listdir('classificar\partitioned')
        for i in dFiles:
            os.remove('classificar\partitioned\\'+i)
    elif platform == "win32":
        dFiles = os.listdir('C:\Users\\'+os.getenv('username')+'\Documents\GitHub\PKS_ML\classificar\partitioned')
        for i in dFiles:
            os.remove('C:\Users\\'+os.getenv('username')+'\Documents\GitHub\PKS_ML\classificar\partitioned\\'+i)
