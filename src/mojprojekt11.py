""""""""""""""""""""
### Meno a priezvysko: Jozef Makis
### Login: xmakis00
""""""""""""""""""""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import soundfile as sf
import copy as copy

""""""""""""""""""""
## Uloha 11
""""""""""""""""""""

fsOn, dataOn = wavfile.read('../audio/maskon_tone.wav')
fsoff, dataOff = wavfile.read('../audio/maskoff_tone.wav')
#vybranie si 2. sekundy zo signalu s ruskom
dataOn_2 = dataOn[16000:32000]
#vybranie si 1.a 2. sekundy zo signalu bez ruska
dataOff_2 = dataOff[31000:47000]
# ustrednenie
dataOn_2 = dataOn_2 - np.mean(dataOn_2)
dataOff_2 = dataOff_2 - np.mean(dataOff_2)
# normalizacia
dataOn = dataOn_2 / np.abs(dataOn_2).max()
dataOff_2 = dataOff_2 / np.abs(dataOff_2).max()
#####################
# rozdelenie na ramce
#####################
#vytvorenie si okna
window = np.hanning(320)

# vykreslenie grafu v spektralnej oblasti
windowFrek = np.fft.fft(window, 1024)/160
mag = np.abs(np.fft.fftshift(windowFrek))
osx = np.linspace(-0.5, 0.5, len(windowFrek))

with np.errstate(divide='ignore', invalid='ignore'):
    responsHan = 10 * np.log10(mag**2)

responsHan = np.clip(responsHan, -200, 100)
plt.plot(osx, responsHan)
plt.gca().set_title("Spektralna oblast hanning okienkovej funkcie")
plt.gca().set_xlabel("frekvencia")
plt.gca().set_ylabel("dB")
plt.savefig("hanningfrek")
plt.show()

# vykreslenie grafu v casovej oblasti
osx = np.linspace(0, 0.02, len(window))
plt.plot(osx, window)
plt.gca().set_xlabel("t [ms]")
plt.gca().set_ylabel("y")
plt.gca().set_title("Graf v casovej oblasti hanning window")
plt.savefig("hanningCas")
plt.show()

prekrytie = int(0.01*fsOn)
pocetVzorkov = int(0.02*fsOn)
pocetRamcov = int((16000/prekrytie))

# vsetky ramce pre nasadenu rusku
ramceOn = np.ndarray((pocetRamcov, pocetVzorkov))

for i in range(0, pocetRamcov-1):
    for j in range(0, pocetVzorkov):
        if((i*prekrytie+j) < 16000):
            ramceOn[i][j] = dataOn[i*prekrytie+j]

# vsetky ramce pre nahravku kde je ruska dole
ramceOff = np.ndarray((pocetRamcov, pocetVzorkov))

for i in range(0, pocetRamcov-1):
    for j in range(0, pocetVzorkov):
        if((i*prekrytie+j) < 16000):
            ramceOff[i][j] = dataOff_2[i*prekrytie+j]

# aplikacia okna na kazdy ramec
for i in range(0, 99):
    ramceOn[i] = ramceOn[i] * window

for i in range(0, 99):
    ramceOff[i] = ramceOff[i] * window

##########################################################
# FFT
##########################################################
nonImpArray = np.ndarray((99, 1024), dtype=np.complex_)
test = np.ndarray((99, 1024))

for i in range(0, 99):
    nonImpArray[i] = np.fft.fft(ramceOn[i], 1024)

for i in range(0, 99):
    for j in range(0, 1024):
        test[i][j] = 10*np.log10((np.abs(nonImpArray[i][j])**2))

nonImpArrayOff = np.ndarray((99, 1024), dtype=np.complex_)
testOff = np.ndarray((99, 1024))

for i in range(0, 99):
    nonImpArrayOff[i] = np.fft.fft(ramceOff[i], 1024)

for i in range(0, 99):
    for j in range(0, 1024):
        testOff[i][j] = 10*np.log10((np.abs(nonImpArrayOff[i][j])**2))

nonImpArrayFrek = np.ndarray((99, 1024), dtype=np.complex_)
nonComFrekS = np.zeros((1024))

for i in range(0, 99):
    for j in range(0, 1024):
        nonImpArrayFrek[i][j] = nonImpArrayOff[i][j]/nonImpArray[i][j]

for i in range(0, 1024):
    for j in range(0, 99):
        nonComFrekS[i] = nonComFrekS[i] + np.abs(nonImpArrayFrek[j][i]) * 1/99

impulgraf = copy.deepcopy(nonComFrekS)
for i in range(0, 1024):
    nonComFrekS[i] = 10 * np.log10(np.abs(nonComFrekS[i])**2)

osx = np.linspace(0, 8000, len(nonComFrekS.transpose()[:512]))
plt.figure(figsize=(9, 3))
plt.plot(osx, nonComFrekS.transpose()[:512])
plt.gca().set_xlabel("Frekvencia")
plt.gca().set_ylabel("y")
plt.gca().set_title("Frekvencna charakteristika rusky po aplikovani okienkovej funkcie")
plt.savefig("frekCharWindowOn")
plt.show()

response = np.fft.ifft(impulgraf, 1024)

# pole pre ulozenie si koeficientu a
a_0 = np.ndarray((1))
a_0[0] = 1
# nacitanie dat z nahravok viet
fsOffsen, dataSentenceOff = wavfile.read('../audio/maskoff_sentence.wav')
fsOnsen, dataSentenceOn = wavfile.read('../audio/maskon_sentence.wav')
# filtrovanie nahravok
filterOutToneOff = signal.lfilter(response[:512], a_0, dataOff)
filterOutSentenceOff = signal.lfilter(response[:512], a_0, dataSentenceOff)
filterOutToneOff = filterOutToneOff.real
filterOutSentenceOff = filterOutSentenceOff.real
filterOutToneOff = filterOutToneOff.astype('int16')
filterOutSentenceOff = filterOutSentenceOff.astype('int16')
sf.write("sim_maskon_tone_window.wav", filterOutToneOff, 16000)
sf.write("sim_maskon_sentence_window.wav", filterOutSentenceOff,  16000)