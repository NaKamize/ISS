""""""""""""""""""""
### Meno a priezvysko: Jozef Makis
### Login: xmakis00
""""""""""""""""""""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import copy as copy
import soundfile as sf

"""""""""""""""""""""""""""""""""""""""
### Uloha 1,2,3
"""""""""""""""""""""""""""""""""""""""
fsOn, dataOn = wavfile.read('../audio/maskon_tone.wav')
fsoff, dataOff = wavfile.read('../audio/maskoff_tone.wav')
# vybranie si 2. sekundy zo signalu s ruskom
dataOn = dataOn[3000:19000]
# vybranie si sekundy ktora najviac vyhovovala frekvencii v podla nasledujucej ulohy
dataOff_2 = dataOff[31000:47000]
# ustrednenie
dataOn = dataOn - np.mean(dataOn)
dataOff_2 = dataOff_2 - np.mean(dataOff_2)
# normalizacia
dataOn = dataOn / np.abs(dataOn).max()
dataOff_2 = dataOff_2 / np.abs(dataOff_2).max()

"""""""""""""""""""""""""""""""""""""""
### Rozdelenie na ramce
"""""""""""""""""""""""""""""""""""""""
prekrytie = int(0.01*fsOn)
pocetVzorkov = int(0.025*fsOn)
pocetRamcov = int((16000/prekrytie))

### vsetky ramce pre nasadenu rusku
ramceOn = np.ndarray((pocetRamcov-1, pocetVzorkov))

for i in range(0, pocetRamcov-1):
    for j in range(0, pocetVzorkov):
        if((i*prekrytie+j) < 16000):
            ramceOn[i][j] = dataOn[i*prekrytie+j]

### vsetky ramce pre nahravku kde je ruska dole
ramceOff = np.ndarray((pocetRamcov-1, pocetVzorkov))

for i in range(0, pocetRamcov-1):
    for j in range(0, pocetVzorkov):
        if((i*prekrytie+j) < 16000):
            ramceOff[i][j] = dataOff_2[i*prekrytie+j]

plt.plot(ramceOn[0])
plt.plot(ramceOff[0])
plt.gca().set_xlabel("Pocet vzorkov v ramci")
plt.gca().set_ylabel("y")
plt.gca().set_title("Piaty ramec z nahravky bez ruska")
plt.savefig("prekryteRamce")
plt.show()

## maskOn clipping
uloha4On = copy.deepcopy(ramceOn)
for i in range(0, pocetRamcov-1):
    kladMax = np.abs(ramceOn[i]).max() * 0.7
    zapMax = np.abs(ramceOn[i]).max() * -0.7
    for j in range(0, pocetVzorkov):
        if uloha4On[i][j] > 0 and uloha4On[i][j] > kladMax:
            uloha4On[i][j] = 1
        elif uloha4On[i][j] < 0 and uloha4On[i][j] < zapMax:
            uloha4On[i][j] = -1
        else:
            uloha4On[i][j] = 0

## maskOff clipping
uloha4Off = copy.deepcopy(ramceOff)
for i in range(0, pocetRamcov-1):
    kladMax = np.abs(ramceOff[i]).max() * 0.7
    zapMax = np.abs(ramceOff[i]).max() * -0.7
    for j in range(0, pocetVzorkov):
        if uloha4Off[i][j] > 0 and uloha4Off[i][j] > kladMax:
            uloha4Off[i][j] = 1
        elif uloha4Off[i][j] < 0 and uloha4Off[i][j] < zapMax:
            uloha4Off[i][j] = -1
        else:
            uloha4Off[i][j] = 0

corrArr = np.ndarray((99, pocetVzorkov*2-1))
for i in range(0, 99):
        corrArr[i] = np.correlate(uloha4Off[i], uloha4On[i], "full")

corrArr2 = np.ndarray((99, pocetVzorkov*2-1))
for i in range(0, 99):
        corrArr2[i] = np.correlate(uloha4On[i], uloha4Off[i], "full")

fazPosunOn = np.array([])
fazPosunOff = np.array([])

for i in range(0, 99):
    fazPosunOn = np.append(fazPosunOn, np.argmax(corrArr[i][399:799]))

for i in range(0, 99):
    fazPosunOff = np.append(fazPosunOff, np.argmax(corrArr2[i][399:799]))

fazPosSpat = 0
fazPosVpred = 0
reRamOn = np.ndarray((99, 320))
reRamOff = np.ndarray((99, 320))
tempResArray = np.array(320)

for i in range(0, pocetRamcov-1):
        if fazPosunOff[i] < fazPosunOn[i]:
            fazPosSpat = int(fazPosunOff[i])
            fazPosVpred = int(fazPosunOff[i] + 320)
            for j in range(fazPosSpat, fazPosVpred):
                 tempResArray = np.append(tempResArray, ramceOn[i][j])
            reRamOn[i] = tempResArray[1:321]
            tempResArray = 0
            reRamOff[i] = ramceOff[i][:320]
        else:
            fazPosSpat = int(fazPosunOn[i])
            fazPosVpred = int(fazPosunOn[i] + 320)
            for j in range(fazPosSpat, fazPosVpred):
                 tempResArray = np.append(tempResArray, ramceOff[i][j])
            ramceOff[i] = tempResArray[1:321]
            tempResArray = 0
            reRamOn[i] = ramceOn[i][:320]

plt.plot(reRamOn[0])
plt.plot(reRamOff[0])
plt.gca().set_xlabel("Pocet vzorkov v ramci")
plt.gca().set_ylabel("y")
plt.gca().set_title("Prve ramce nahravok")
plt.savefig("hotovyPosun")
plt.show()
            
##########################################################
# FFT
##########################################################
nonImpArray = np.ndarray((99, 1024), dtype=np.complex_)
test = np.ndarray((99, 1024))

for i in range(0, 99):
    nonImpArray[i] = np.fft.fft(reRamOn[i], 1024)

for i in range(0, 99):
    for j in range(0, 1024):
        test[i][j] = 10*np.log10((np.abs(nonImpArray[i][j])**2))

nonImpArrayOff = np.ndarray((99, 1024), dtype=np.complex_)
testOff = np.ndarray((99, 1024))

for i in range(0, 99):
    nonImpArrayOff[i] = np.fft.fft(reRamOff[i], 1024)

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

response = np.fft.ifft(impulgraf, 1024)

a_0 = np.ndarray((1))
a_0[0] = 1
# nacitanie dat z nahravok viet
fsOffsen, dataSentenceOff = wavfile.read('../audio/maskoff_sentence.wav')

filterOutToneOff = signal.lfilter(response[:512], a_0, dataOff)
filterOutSentenceOff = signal.lfilter(response[:512], a_0, dataSentenceOff)

filterOutToneOff = filterOutToneOff.real
filterOutSentenceOff = filterOutSentenceOff.real
filterOutToneOff = filterOutToneOff.astype('int16')
filterOutSentenceOff = filterOutSentenceOff.astype('int16')

sf.write("sim_maskon_tone_phase.wav", filterOutToneOff, 16000)
sf.write("sim_maskon_sentence_phase.wav", filterOutSentenceOff, 16000)