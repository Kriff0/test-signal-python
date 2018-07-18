iimport numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal as sg


def patient(data, numero_patient):  ## commit: changement eeg par data
    # retourne les données d'1 seul patient
    col_concernees = []
    for i in range(len(data.iloc[0, :])):
        verif = data.columns[i].find(numero_patient)
        if verif != -1:
            col_concernees.append(data.columns[i])
    # return col_concernees
    return data[col_concernees]


def parametre(data, nom_parametre):   ## commit: changement eeg par data
    # retourne les données d'une catégorie
    col_concernees0 = []
    col_concernees = []
    # on enleve les colonnes avec le nan parce que nan != -1 renvoie True
    # ce qui induit une erreur lorsqu'on fait columns(i) et que i = nan
    for i in range(len(data.iloc[0, :])):
        verif0 = data.iloc[0, i]
        if verif0 == verif0:
            col_concernees0.append(data.columns[i])
    # on cherche dans dataframe data modifié le nom de la catégorie demandée
    for i in range(len(data[col_concernees0].iloc[0, :])):
        verif = data[col_concernees0].iloc[0, i].find(nom_parametre)
        if verif != -1:
            col_concernees.append(data[col_concernees0].columns[i])
    # return col_concernees
    return data[col_concernees]


def parametre_columns(data, nom_parametre):   ## commit: changement eeg par data
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


def spectre(s):  ## commit: à enlever
    tfd = np.fft.fft(s)
    N = len(s)
    return np.absolute(tfd)*2/N


def affiche_signal(data):  ## commit: à enlever
    plt.figure
    data.plot()
    plt.show()


def affichage_temps(t, s, figure=1,  sign='S'):  ## commit: à enlever
    plt.figure(figure)
    plt.plot(t, s)
    plt.xlabel('Temps (en secondes)')
    labelY = 'Amplitude de ' + sign
    plt.ylabel(labelY)
    title = 'Amplitude de ' + sign + ' en fonction du temps'
    plt.title(title)


def conversion_data_obj_float(data, l0=0, l_fin=None, col0=0, col_fin=None):   ## commit on modifie les print
    print('Ne fonctionne que si les données à changer sont des nombres')
    if len(data.shape) == 1:
        print(data, ' possède 1 colonne')
        data.iloc[l0:l_fin] = data.iloc[l0:l_fin].astype(np.float64)
        return data
    elif len(data.shape) == 2:
        print(data, ' possède plusieurs colonnes')
        # Si data est une dataframe avec au moins 2 colonnes
        data.iloc[l0:l_fin, col0:col_fin] = data.iloc[l0:l_fin, col0:col_fin].astype(np.float64)
        return data
    else:
        # Si data n'est pas une dataframe
        print("len.shape =", len(list(data.iloc[0, :].shape)))
        return "Erreur : ", data, " n'est pas une dataframe"


def affichage_spectre(t, spectre, figure=1):  ## commit modification   ## fonction mauvaise ?
    plt.figure(figure)
    freq = np.arange(len(spectre))*1.0/max(t)
    plt.figure(figure)
    plt.plot(freq, spectre)
    nom_spectre = nom_objet(spectre)
    plt.xlabel('Fréquence (en Hz)')
    labelY = 'Amplitude de ' + nom_spectre
    plt.ylabel(labelY)
    title = 'Amplitude de ' + nom_spectre + ' en fonction de la fréquence'
    plt.title(title)
    plt.show()


def affichage_spectre_dataframe(spectre, figure=1):  ## commit modification
    spectre.plot()
    nom_spectre = nom_objet(spectre)
    plt.xlabel('Fréquence (en Hz)')
    labelY = 'Amplitude de ' + nom_spectre
    plt.ylabel(labelY)     ## PB d'abscisse
    title = 'Amplitude de ' + nom_spectre + ' en fonction de la fréquence'
    plt.title(title)
    plt.show()


def conversion_secondes(unite, t):
    t = np.array(t)
    if unite == 'heure':
        t *= 3600
        return t
    elif unite == 'minute':
        t *= 60
        return t
    elif unite == 'secondes':
        return t
    else:
        return "Erreur : le paramètre d'entrée unite est soit 'heure', soit 'minute', soit 'seconde'"


def enleve_date_data(data):   ## commit
    data_to_return = pd.DataFrame()
    print(data_to_return)
    for i in range(len(data.iloc[0, :])):
        verif = data.iloc[0, i].find('Time')
        if verif == -1:
            data_to_return = pd.concat([data_to_return, data.iloc[:, i]], axis=1, sort=False)
    return data_to_return


def nom_objet(obj):
    nom = set(varname for varname, varval in globals().items() if varval is obj)
    nom = str(nom)
    nom = nom.split("'")
    nom = nom[1]
    return nom


def signal_data(data):
    # Attention à l'ordre des différentes opérations sur data !
    # Peut changer selon data ! Fonction construitre pour pour la variable eeg
    print('Attention ! Quelques données ont pu être supprimées dû aux opérations sur ', data)
    data_sans_NaN = data.dropna(axis=1, how='all')
    data_sans_NaN = data_sans_NaN.dropna()
    data_sans_date = enleve_date_data(data_sans_NaN)
    data_sans_nom_col = data_sans_date.iloc[1:, :]
    data_en_float = conversion_data_obj_float(data_sans_nom_col)
    return data_en_float


def affichage_signal_data(signal_data, figure=1):  ## commit
    plt.figure(figure)
    signal_data.plot()
    nom_data = nom_objet(signal_data)
    plt.xlabel('Temps (en minute)')
    ylab = 'Amplitude des signaux de ' + nom_data
    plt.ylabel(ylab)
    title = 'Amplitude de ' + nom_data + ' en fonction du temps'
    print(title)
    plt.title(title)
    plt.show()


def spectres_data0(data):  ## fonction mauvaise ?
    spectres = []
    for i in range(len(data.iloc[0, :])):
        si = data.iloc[:, i]
        sp = spectre(si)
        spectres.append(sp)
    spectres = np.array(spectres).transpose()
    return spectres


def frequence_periode(data_depart):  ## commit : nouvelle fonction
    # Attention ! Cette fonction est construite pour la data de départ eeg
    freq = [i/600 for i in range(len(data_depart.iloc[:, 0])-1)]
    freq_index = [10*(i+1) for i in range(len(data_depart)-1)]
    freq = pd.DataFrame(freq, index=freq_index)
    freq.rename(columns={0: 'Fréquences'}, inplace=True)
    return freq


def frequence_date(data_depart):  ## commit : nouvelle fonction
    # a ete construit pour un fichier excel similaire à eeg2
    T0 = data_depart.index[0]
    T1 = data_depart.index[1]
    Te = (T1-T0).total_seconds()
    freq = np.arange(0, len(data_depart.index)/Te - 1/Te, 1/Te)
    freq_data = pd.DataFrame(freq, index=data_depart.index)
    freq_data.rename(columns={0: 'Fréquences'}, inplace=True)
    return freq_data


def spectres_data_dataframe(data, data_depart): ## commit : nouvelle fonction
    # ici normalement, data_depart  = eeg
    spectres = []
    # on fait les spectres de signaux 1 par 1 (col par col)
    for i in range(len(data.iloc[0, :])):
        si = data.iloc[:, i]
        sp = spectre(si)
        spectres.append(sp)
    spectres = pd.DataFrame(spectres)
    spectres = spectres.transpose()
    spectres.index = data.index
    spectres.columns = data.columns
    # on construit la colonne freq afin de remplacer l'index temporelle par
    # un index fréquentiel
    if type(data_depart.index[0]) == pd._libs.tslibs.timestamps.Timestamp:
        freq = frequence_date(data_depart)
    else:
        freq = frequence_periode(data_depart)
    spectres_f = pd.merge(spectres, freq, right_index=True, left_index=True)
    spectres_f.set_index('Fréquences', inplace=True)
    return spectres_f


def affichage_spectrogramme(s, fe, figure=1, cmap=None):  ## commit: nouvelle fonction
    signal = np.array(s.iloc[:])
    nb_signal = signal.shape[1]
    if nb_signal == 1:
        s = np.array(param_plus.iloc[:, 0])
        plt.figure(figure)
        freq, temps, spectro = sg.spectrogram(s, fe)
        plt.pcolormesh(temps, freq, spectro, cmap=cmap)
        plt.grid()
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title('Spectrogramme')
        plt.show()
        return freq, temps, spectro
    if nb_signal > 1:
        spectro = []
        for i in range(nb_signal):
            s = np.array(param_plus.iloc[:, i])
            plt.figure(figure)
            freq, temps, Sxx = sg.spectrogram(s, fe)
            plt.pcolormesh(temps, freq, Sxx, cmap=cmap)
            plt.grid()
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.title('Spectrogramme')
            plt.show()
            spectro.append(Sxx)
        return freq, temps, spectro
    else:
        return "Erreur : il n'y a pas de signal à afficher"


# fonction main()
if __name__ == '__main__':
    eeg = pd.read_excel('Copie de Moyenne_10Min_Fabien_TempsFrequence.xlsx')
    eeg.index = eeg.iloc[:, 1] * 10
    t = eeg.index[1:]
    sns.set()
    # personne (p05)+ param (delta)

    p05 = patient(eeg, 'P05')
    p05d = parametre(p05, 'Delta')
    p05d0 = signal_data(p05d)
    #affichage_signal_data(p05d0)
    #sp05d00 = spectres_data0(p05d0)
    sp05d01 = spectres_data_dataframe(p05d0, eeg)
    #affichage_spectre(t, sp05d00, figure=5)
    #affichage_spectre_dataframe(sp05d01)

    # delta + p05

    d = parametre(eeg, 'Delta')
    d05 = patient(d, 'P05')
    d050 = signal_data(d05)
    #affiche_signal(d050)
    #spd0500 = spectres_data0(d050)
    spd0501 = spectres_data_dataframe(d050, eeg)
    #affichage_spectre(t, spd050)
    #affichage_spectre(t, spd0500, figure=6)
    #affichage_spectre_dataframe(spd0501)

    # delta
    d00 = signal_data(d)
    #affiche_signal(d00)
    #spd00 = spectres_data0(d00)
    Spd01 = spectres_data_dataframe(d00, eeg)
    #affichage_spectre(t, spd)
    #affichage_spectre_dataframe(Spd01)


    #p05
    p0500 = signal_data(p05)
    #affiche_signal(p0500)
    #sp5000 = spectres_data0(p0500)
    sp5001 = spectres_data_dataframe(p0500, eeg)
    #affichage_spectre(t, sp500)
    #affichage_spectre(t, sp5000)
    #affichage_spectre_dataframe(sp5001)


    ###################################
    # Etude de l'autre document excel

    eeg2 = pd.read_excel('P21_feat_diff_ArtfexclFalse_ExtrexclFalse.xlsx')
    param_plus = parametre_columns(eeg2, 'heog+')
    affichage_signal_data(param_plus)
    sp_param_plus = spectres_data_dataframe(param_plus, eeg2)
    affichage_spectre_dataframe(sp_param_plus)

    # Etude avec Spectrogramme
    fe = 1/150
    freq, temps, spectro = affichage_spectrogramme(param_plus, fe)

# CWT (tau, S) = somme (de - inf à +inf) de[x(t) * psi conj ((t-tau)/S)]dt
# pr morlet, psi(t) = exp(i*a*t)*exp((-t²)/(2*rho²)
# a = param de modulatio, et rho = param d'étallonage
