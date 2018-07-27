import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal as sg


def temps_index_patient(data, patient, date=None):
    if date is None:
        date = '2014-01-13 19:40:00' # date dedébut pour p03
    t1 = (str(data.index[1])[-5:-3])
    t0 = (str(data.index[0])[-5:-3])
    time_step = int(t1)- int(t0) # date pour eeg
    if time_step == 0:
        time_step = int(str(data.index[1])[-2:]) - int(str(data.index[0])[-2:])  # date pour eeg2
    freq = str(time_step) + 'min'
    periode = len(data.index)
    dtindex = pd.date_range(start=date, periods=periode, freq=freq)
    dtindex = pd.DataFrame(dtindex)
    dtindex.rename(columns={0: 'Temps'}, inplace=True)
    return dtindex


def patient(data, numero_patient):
    # Permet de sélectionner les données d'un patient
    col_concernees = []
    for i in range(len(data.iloc[0, :])):
        verif = data.columns[i].find(numero_patient)
        if verif != -1:
            col_concernees.append(data.columns[i])
    data_return = data[col_concernees]
    temps = 'Time_' + numero_patient
    data_return.set_index(temps, inplace=True)
    data_return = data_return.dropna()
    dtindex = temps_index_patient(data_return, numero_patient)
    data_return0 = pd.concat([data_return, dtindex], axis=1)
    for i in range(len(dtindex)):
        if data_return0.iloc[i, -1] is pd.NaT:
            data_return0.iloc[i, -1] = pd.Timestamp(2014, 1, 13, 19, 40, 0) + pd.Timedelta(minutes=10*i)
    N = int(len(data_return0))
    data_return0 = data_return0.iloc[:N//2, :]
    data_return0.set_index('Temps', inplace=True)
    return data_return0


def parametre_columns(data, nom_parametre):
    # retourne les données d'une catégorie
    col_concernees0 = []
    col_concernees = []
    # on enleve les colonnes avec le nan parce que nan != -1 renvoie True
    # ce qui induit une erreur lorsqu'on fait columns(i) et que i = nan
    for i in range(len(data.iloc[0, :])):
        verif0 = data.columns[i]
        if verif0 == verif0:
            col_concernees0.append(data.columns[i])
    # on cherche dans dataframe data modifié le nom de la catégorie demandée
    for i in range(len(data[col_concernees0].iloc[0, :])):
        verif = data[col_concernees0].columns[i].find(nom_parametre)
        if verif != -1:
            col_concernees.append(data[col_concernees0].columns[i])
    # return col_concernees
    return data[col_concernees]


def nom_objet(obj):
    nom = set(varname for varname, varval in globals().items() if varval is obj)
    nom = str(nom)
    nom = nom.split("'")
    nom = nom[1]
    return nom


def affichage_signal_data(signal_data, figure=1):
    plt.figure(figure)
    signal_data.plot()
    # Attention cette ligne est  à utiliser si la premiere ligne de signal_data n'est pas composée de nombres
    #signal_data.iloc[1:].plot()
    #nom_data = nom_objet(signal_data)
    plt.xlabel('Date')
    #ylab = 'Amplitude des signaux de ' + nom_data
    ylab = 'Amplitude du signal de P03 delta'
    plt.ylabel(ylab)
    # title = 'Amplitude de ' + nom_data + ' en fonction du temps'
    title = ylab = 'Amplitude du sgianl de P03 delta en fonction du temps'
    plt.title(title)
    plt.show()


def affichage_signal_data_subplot(signal_data, figure=1):
    plt.figure(figure)
    signal_data.plot(subplots=True)
    # Attention cette ligne est  à utiliser si la premiere ligne de signal_data n'est pas composée de nombres
    #signal_data.iloc[1:].plot(subplots=True)
    nom_data = nom_objet(signal_data)
    plt.xlabel('Date')
    ylab = 'Amplitude des signaux de ' + nom_data
    plt.ylabel(ylab)
    title = 'Amplitude de ' + nom_data + ' en fonction du temps'
    plt.title(title)
    plt.show()


def conversion_data_obj_float(data, l0=0, l_fin=None, col0=0, col_fin=None):
    if len(data.shape) == 1:
        data.iloc[l0:l_fin] = data.iloc[l0:l_fin].astype(np.float64)
        return data
    elif len(data.shape) == 2:
        # Si data est une dataframe avec au moins 2 colonnes
        data.iloc[l0:l_fin, col0:col_fin] = data.iloc[l0:l_fin, col0:col_fin].astype(np.float64)
        return data
    else:
        # Si data n'est pas une dataframe
        return "Erreur : ", data, " n'est pas une dataframe"


def enleve_date_data(data):
    data_to_return = pd.DataFrame()
    for i in range(len(data.iloc[0, :])):
        verif = data.iloc[0, i].find('Time')
        if verif == -1:
            data_to_return = pd.concat([data_to_return, data.iloc[:, i]], axis=1, sort=False)
    return data_to_return


def signal_data(data):
    # Attention à l'ordre des différentes opérations sur data !
    # Peut changer selon data ! Fonction construitre pour pour la variable eeg
    nom = nom_objet(data)
    print('Attention ! Quelques données ont pu être supprimées dû aux opérations sur ', nom, '!')
    data_sans_NaN = data.dropna(axis=1, how='all')
    data_sans_NaN = data_sans_NaN.dropna()
    #data_sans_date = enleve_date_data(data_sans_NaN)
    #data_sans_nom_col = data_sans_date.iloc[1:, :]
    data_sans_nom_col = data_sans_NaN.iloc[0:, :]
    data_en_float = conversion_data_obj_float(data_sans_nom_col)
    return data_en_float


def densite_spectrale(s):
    densite = np.abs(np.fft.fft(s))**2
    time_step = float(str(s.index[2] - s.index[1])[-5:-3])
    time_step *= 60  # car time_step est actuellement en minute
    if time_step == 0.0:
        time_step = float(str(s.index[2] - s.index[1])[-2:])
    freqs = np.fft.fftfreq(s.size, time_step)
    idx = np.argsort(freqs)
    return freqs[idx], densite[idx]


def spectres_data_dataframe2(data):
    #possèdes des lignes potentiellement inutiles (celles liées à la freq)
    spectres = []
    # on fait les spectres de signaux 1 par 1 (col par col)
    for i in range(len(data.iloc[0, :])):
        si = data.iloc[:, i]
        freqs, sp = densite_spectrale(si)
        spectres.append(sp)
    spectres = pd.DataFrame(spectres)
    spectres = spectres.transpose()
    spectres.index = data.index
    spectres.columns = data.columns

    return spectres


def affichage_spectre_dataframe(spectre, figure=1):
    N = int(len(spectre)/2)
    spectre.iloc[N+1:].plot()
    nom_spectre = nom_objet(spectre)
    plt.xlabel('Duree(en heure)')
    labelY = 'Densité spectrale de ' + nom_spectre
    plt.ylabel(labelY)
    title = 'Densité spectrale de ' + nom_spectre + ' en fonction de la durée'
    plt.title(title)
    plt.show()


def affichage_spectre_dataframe0_subplot(spectre, figure=1):
    N = int(len(spectre)/2)
    spectre.iloc[N+1:].plot(subplots=True)
    nom_spectre = nom_objet(spectre)
    plt.xlabel('Durée (en heure)')
    labelY = 'Densité spectrale de ' + nom_spectre
    plt.ylabel(labelY)
    title = 'Densité spectrale de ' + nom_spectre + ' en fonction de la durée'
    plt.title(title)
    plt.show()


def affichage_spectrogramme(data, fe, figure=1, freq_dtxlabel=None, cmap=None, nperseg=256, noverlap=None, nfft=None):
    signal = np.array(data.iloc[:])
    nb_signal = signal.shape[1]
    if nb_signal == 1:
        s = np.array(data.iloc[:, 0])
        plt.figure(figure)
        freq, temps, spectro = sg.spectrogram(s, fe, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
        temps_h = temps / 3600
        if freq_dtxlabel is None:
            duree = data.index[-1] - data.index[0]
            duree_heure = duree.total_seconds() / 3600
            freqx = round(duree_heure / len(temps_h), 4)
            freq_dtxlabel = str(freqx) + 'H'
        dtxlabel = pd.date_range(start=data.index[0], periods=len(temps_h), freq=freq_dtxlabel)
        plt.pcolormesh(dtxlabel, freq, spectro, cmap=cmap)
        #plt.pcolormesh(temps_h, freq, spectro, cmap=cmap)
        plt.grid()
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [h]')
        titre = 'Spectrogramme avec nperseg = ' + str(nperseg) + ' noverlap = ' + str(noverlap) + ' et nfft = ' + str(nfft)
        plt.title(titre)
        plt.show()
        return freq, dtxlabel, spectro
    if nb_signal > 1:
        spectro = []
        for i in range(nb_signal):
            s = np.array(data.iloc[:, i])
            plt.figure(figure)
            freq, temps, Sxx = sg.spectrogram(s, fe, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
            temps_h = temps / 3600
            if freq_dtxlabel is None:
                duree = data.index[-1] - data.index[0]
                duree_heure = duree.total_seconds() / 3600
                freqx = round(duree_heure / len(temps_h), 4)
                freq_dtxlabel = str(freqx) + 'H'
            dtxlabel = pd.date_range(start=data.index[0], periods=len(temps_h), freq=freq_dtxlabel)
            plt.pcolormesh(dtxlabel, freq, Sxx, cmap=cmap)
            #plt.pcolormesh(temps_h, freq, Sxx, cmap=cmap)
            plt.grid()
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            titre = 'Spectrogramme avec nperseg = ' + str(nperseg) + ' noverlap = ' + str(noverlap) + ' et nfft = ' + str(nfft)
            plt.title(titre)
            plt.show()
            spectro.append(Sxx)
        return freq, temps_h, spectro
    else:
        return "Erreur : il n'y a pas de signal à afficher"


def zoom_spectro(spectro, freq, temps, freq_lim, cmap=None):
    spectro_return = []
    freq_return = []
    for i in range(len(freq)):
        if freq[i] <= freq_lim:
            spectro_return.append(spectro[i])
            freq_return.append(freq[i])
    plt.pcolormesh(temps, freq_return, spectro_return, cmap=cmap)
    plt.grid()
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [h]')
    titre = 'Spectrogramme'
    plt.title(titre)
    plt.show()
    #return spectro_return, freq_return


# fonction main()

if __name__ == '__main__':
    sns.set()

    eeg = pd.read_excel('Copie de Moyenne_10Min_Fabien_TempsFrequence.xlsx')
    eeg.index = eeg.iloc[:, 1]

    eeg.columns = eeg.iloc[0, :]
    eeg0 = eeg.iloc[1:, :]
    eeg1 = eeg0.dropna(axis=1, how='all')
    eeg1

    # p03 + Delta
    p03 = patient(eeg1, 'P03')
    p03
    p03delta = parametre_columns(p03, "Delta")
    p03delta
    p03delta0 = signal_data(p03delta)
    p03delta0
    affichage_signal_data(p03delta0)
    spectre_p03delta = spectres_data_dataframe2(p03delta0)
    spectre_p03delta
    affichage_spectre_dataframe(spectre_p03delta)

    fe = 1/600
    D = 32
    freq, temps, spectro = affichage_spectrogramme(p03delta0, fe, figure=1, cmap='spring', nperseg=D, noverlap=31, nfft=64)

    zoom_spectro(spectro, freq, temps, 0.0006)  # avec données de eeg (données moyennes)

    ###################################
    # Etude de l'autre document excel

    #eeg2 = pd.read_excel('P21_feat_diff_ArtfexclFalse_ExtrexclFalse.xlsx')
    eeg2 = pd.read_excel('/home/david/Documents/David/Informatique/formation/Notebooks/P03_feat_diff_filt_ArtfEmpty1_ExtrExclTrue.xlsx')
    eeg2
    param_plus = parametre_columns(eeg2, 'Fz-Cz')
    param_plus
    affichage_signal_data(param_plus)

    p03_ok = signal_data(param_plus)
    param_plus0 = p03_ok
    param_plus0
    sp_param_plus = spectres_data_dataframe2(param_plus0)
    sp_param_plus

    affichage_spectre_dataframe(sp_param_plus)


    # Etude avec Spectrogramme

    fe2 = 1 / 15
    D2 = 4 * 320
    freq2, temps_h2, spectro2 = affichage_spectrogramme(param_plus0, fe2, cmap = 'spring', nperseg=D2,noverlap = D2-1, nfft=D2)

    zoom_spectro(spectro2, freq2, temps_h2, freq_lim=0.0006)  # avec données de eeg (données brutes)
