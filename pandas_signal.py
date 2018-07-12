import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  ## Commit 1 : Création d'un main + affichage spectre 
eeg = pd.read_excel('Copie de Moyenne_10Min_Fabien_TempsFrequence.xlsx')


def patient(numero_patient):
    # retourne les données d'1 seul patient
    col_concernees = []
    for i in range(len(eeg.iloc[0, :])):
        verif = eeg.columns[i].find(numero_patient)
        if verif != -1:
            col_concernees.append(eeg.columns[i])
    # return col_concernees
    return eeg[col_concernees]


def categorie(nom_categorie):
    # retourne les données d'une catégorie
    col_concernees0 = []
    col_concernees = []
    # on enleve les colonnes avec le nan parce que nan != -1 renvoie True
    # ce qui induit une erreur lorsqu'on fait columns(i) et que i = nan
    for i in range(len(eeg.iloc[0, :])):
        verif0 = eeg.iloc[0, i]
        if verif0 == verif0:
            col_concernees0.append(eeg.columns[i])
    # on cherche dans dataframe eeg modifié le nom de la catégorie demandée
    for i in range(len(eeg[col_concernees0].iloc[0, :])):
        verif = eeg[col_concernees0].iloc[0, i].find(nom_categorie)
        if verif != -1:
            col_concernees.append(eeg[col_concernees0].columns[i])
    # return col_concernees
    return eeg[col_concernees]


def affiche_signal(data):
    plt.figure
    data.plot()
    plt.show()


def conversion_obj_float(data):
    data.iloc[1:, :-2] = data.iloc[1:, :-2].astype(np.float64)
    return data


def spectre(s, t):
    tfd = np.fft.fft(s)
    N = len(s)
    return np.absolute(tfd)*2/N


def affichage_spectre(t, spectre, figure=4, sp='Sp', couleur='b'):  ## Commit 1 : Création d'un main + affichage spectre
    plt.figure(figure)
    sns.set()
    freq = np.arange(len(spectre))*1.0/max(t)
    plt.figure(figure)
    plt.plot(freq, spectre, couleur)
    plt.xlabel('Fréquence (en Hz)')
    labelY = 'Amplitude de ' + sp
    plt.ylabel(labelY)
    title = 'Amplitude de ' + sp + ' en fonction de la fréquence #fft by numpy'
    plt.title(title)
    plt.show()


def conversion_secondes(unite, t):  ## Commit 1 : Création d'un main + affichage spectre
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
    
    
# fonction main()  ## Commit 1 : Création d'un main + affichage spectre
eeg = pd.read_excel('Copie de Moyenne_10Min_Fabien_TempsFrequence.xlsx')
    # affiche_signal(eeg)
    # on prend en exemple la catégorie delta
    delta = categorie('Delta')
    delta = delta.dropna()  # enleve toutes les lignes avec des NaN
    # delta.iloc[1:, :-2] = conversion_obj_float(delta)
    t = delta.index[1:]
    t = conversion_secondes('minute', t)
    Te = (t[1] - t[0])
    fe = 1 / Te
    spectre_delta = spectre(delta.iloc[1:, :])
    spectre_delta
    affichage_spectre(t, spectre_delta)


#CWT (tau, S) = somme (de - inf à +inf) de[x(t) * psi conj ((t-tau)/S)]dt
# pr morlet, psi(t) = exp(i*a*t)*exp((-t²)/(2*rho²)
#a = param de modulatio, et rho = param d'étallonage
