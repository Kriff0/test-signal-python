import numpy as np
import matplotlib
from scipy import signal as sg
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import datetime

# fonction pour construire des signaux
def signal(f, t):
    return np.sin(2*np.pi*f*t)


def spectre(s):
    tfd = np.fft.fft(s)
    N = len(s)
    return np.absolute(tfd)*2/N


# fonctions affichage d'un signal en fonction du temps
def affichage_temps(t, s, figure=1,  sign='S'):
    plt.figure(figure)
    plt.plot(t, s)
    plt.xlabel('Temps (en secondes)')
    labelY = 'Amplitude de ' + sign
    plt.ylabel(labelY)
    title = 'Amplitude de ' + sign + ' en fonction du temps'
    plt.title(title)



def affichage_temps2(t, s, figure=1,  sign='S'):
    plt.figure(figure)
    sns.set()
    plt.plot(t, s)
    plt.xlabel('Temps en secondes')
    labelY = 'Amplitude de ' + sign
    plt.ylabel(labelY)
    title = 'Amplitude de ' + sign + ' en fonction du temps'
    plt.title(title)


# fonction affichage de signaux (14 maximum)
def affichage_signaux(t, *args, color='w'):
    nb = len(args)
    # si la fonction ne prend pas de paramètres à part le temps en entrée
    if nb == 0:
        print("Mettez un ou des paramètres")
    if nb > 14:
        print("Mettez moins de 14 signaux en paramètres")
    # liste de couleurs pour le tracé de courbes
    couleur = ['blue', 'red', 'green', 'yellow', 'orange', 'pink', 'grey', 'black', 'purple', 'cyan', 'lime', 'brown', 'gold', 'chocolate']
    compteur = 0
    for i in args:
        # si les arguments (sans compter le temps) ne sont pas des signaux
        if type(i) != np.ndarray:
            print("Mettez un ou des listes en paramètres")
            return
        longueur_signal = len(i)
        plt.figure(4)
        for j in range(longueur_signal):
            if type(i[j]) != np.float64:
                print("Mettez un ou des signaux en paramètres")
                return
        # affichage des courbes
        plt.plot(t, i, couleur[compteur])
        legende = 's' + str(compteur + 1)
        plt.legend(legende)
        compteur += 1
    plt.xlabel('Temps (en secondes)')
    plt.ylabel('Amplitude des différents signaux')
    plt.title('Amplitude des différents signaux en fonction du temps')
    plt.show()


# fonction affichage d'un spectre en fonction de la fréquence
def affichage_spectre(t, spectre, figure=4, sp='Sp', couleur='b'):
    plt.figure(figure)
    sns.set()
    freq = np.arange(len(spectre))*1.0/max(t)
    plt.figure(figure)
    plt.plot(freq, spectre, couleur)
    plt.xlabel('Fréquence (en Hz)')
    labelY = 'Amplitude de ' + sp
    plt.ylabel(labelY)
    title = 'Amplitude de ' + sp + ' en fonction de la fréquence  #fft by numpy'
    plt.title(title)
    plt.show()


# Spectrogramme
def affichage_spectrogramme(s, fe, figure=1):
    plt.figure(figure)
    freq, temps, Sxx = sg.spectrogram(s, fe)
    plt.pcolormesh(temps, freq, Sxx)
    plt.grid()
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogramme')
    plt.show()
    return freq, temps, Sxx


def affichage_spectrogramme_3D(freq, temps, spectre, figure=1):
    fig = plt.figure(figure)
    ax = Axes3D(fig)
    temps, freq = np.meshgrid(temps, freq)
    surf = ax.plot_surface(temps, freq, Sxx, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(matplotlib.ticker.LinearLocator(10))
    ax.zaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('Représentation 3D du spectrogramme')
    plt.xlabel('Temps (en secondes)')
    plt.ylabel('Fréquence (en Hertz)')
    plt.show()


# fonction main()
if __name__ == '__main__':
    """Permet de lance ce qu'il y a en dessous uniquement quand il s'agit du
    script principal et non quand il est imorté"""
    Te = 1 / 512  # pas en temps
    t = np.arange(0, 10, Te)
    # construction des signaux s1, s2 et s3
    f1 = 10
    s1 = signal(f1, t)
    f2 = 30
    s2 = signal(f2, t)
    s3 = s1 + s2

    # construction des spectres
    S1 = spectre(s1)
    S2 = spectre(s2)
    S3 = spectre(s3)

    # affichage des signaux en fonction du temps

    affichage_temps(t, s1, 1, 's1')
    affichage_temps(t, s2, 2, 's2')
    affichage_temps(t, s3, 3, 's3')

    """
    # affichage de plusieurs signaux
    affichage_signaux(t, *[s1, 0.5*s2, 0.3*s1])


    # affichage des spectres en fréquence
    affichage_spectre(t, S1, 4, 'S1', 'b')
    affichage_spectre(t, S2, 5, 'S2', 'r')
    affichage_spectre(t, S3, 6, 'S3', 'g')


    # sprectrogramme
    affichage_spectrogramme(s1, 1/Te, 10)
    affichage_spectrogramme(s2, 1/Te, 11)
    freq, temps, Sxx = affichage_spectrogramme(s3, 1/Te, 12)
    """

    # représentaton 3D
    affichage_spectrogramme_3D(freq, temps, Sxx)

    # On construit le vecteur time grâce à l'index de eeg
    def temps_data(data):
        time = []
        for t in data.index:
            time.append(t)
        return time
    # on change le vecteur time en vecteur temps car time est composée de string
    # et on veut essayer d'avoir des durées
    def date_devient_temps(time):
        t0 = str(time[0])
        t1 = str(time[1])
        nouveau_temps0 = datetime.datetime.strptime(t0, '%Y-%m-%d %H:%M:%S')
        nouveau_temps1 = datetime.datetime.strptime(t1, '%Y-%m-%d %H:%M:%S')
        temps = [0]
        for t in range(len(time) - 1):
            t0 = str(time[t])
            t1 = str(time[t + 1])
            nouveau_temps0 = datetime.datetime.strptime(t0, '%Y-%m-%d %H:%M:%S')
            nouveau_temps1 = datetime.datetime.strptime(t1, '%Y-%m-%d %H:%M:%S')
            duree = nouveau_temps1 - nouveau_temps0
            duree = str(duree.seconds)
            duree = int(duree) + temps[t]
            temps.append(duree)
        return temps


    def spectres_data(data):
        spectres = []
        for i in range(len(data.iloc[0, :])):
            si = data.iloc[:, i]
            sp = spectre(si)
            spectres.append(sp)
        return spectres


    def affichage_spectro_data(data):
        time = temps_data(data)
        temps = date_devient_temps(time)
        spectres = spectres_data(data)
        fe = 20 / (max(temps))
        for i in range(len(spectres)):
            freq, temps, Sxx = affichage_spectrogramme(spectres[i], fe)
            affichage_spectrogramme_3D(freq, temps, Sxx, i+20)


    eeg = pd.read_excel('/home/david/Documents/David/Informatique/signaux_test.xlsx')
    affichage_spectro_data(eeg)
