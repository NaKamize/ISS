""""""""""""""""""""
### Meno a priezvysko: Jozef Makis
### Login: xmakis00
""""""""""""""""""""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import copy as copy

"""""""""""""""""""""""""""""""""""""""
### Uloha 1,2,3
"""""""""""""""""""""""""""""""""""""""
fsOn, dataOn = wavfile.read('../audio/maskon_tone.wav')
fsoff, dataOff = wavfile.read('../audio/maskoff_tone.wav')
# vybranie si 2. sekundy zo signalu s ruskom
dataOn = dataOn[16000:32000]
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
pocetVzorkov = int(0.02*fsOn)
pocetRamcov = int((16000/prekrytie))

### vsetky ramce pre nasadenu rusku
ramceOn = np.ndarray((pocetRamcov, pocetVzorkov))

for i in range(0, pocetRamcov-1):
    for j in range(0, pocetVzorkov):
        if((i*prekrytie+j) < 16000):
            ramceOn[i][j] = dataOn[i*prekrytie+j]


### vsetky ramce pre nahravku kde je ruska dole
ramceOff = np.ndarray((pocetRamcov, pocetVzorkov))

for i in range(0, pocetRamcov-1):
    for j in range(0, pocetVzorkov):
        if((i*prekrytie+j) < 16000):
            ramceOff[i][j] = dataOff_2[i*prekrytie+j]

""""""""""""""""""""
## Uloha 4
""""""""""""""""""""
## maskOn clipping
uloha4On = ramceOn
#lag 74, 0.865
for i in range(0, pocetRamcov-1):
    kladMax = np.abs(ramceOn[i]).max() * 0.865
    zapMax = -np.abs(ramceOn[i]).max() * 0.865
    for j in range(0, pocetVzorkov):
        if uloha4On[i][j] > 0 and uloha4On[i][j] > kladMax:
            uloha4On[i][j] = 1
        elif uloha4On[i][j] < 0 and uloha4On[i][j] < zapMax:
            uloha4On[i][j] = -1
        else:
            uloha4On[i][j] = 0

## maskOff clipping
uloha4Off = ramceOff
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

## Autokorelacia maskOn
uloha4Acorr = copy.deepcopy(uloha4On)
arrCorr = np.ndarray((320, 320))
tempArray = np.array([])
interShift = 0

# v prvom cykle sa vyberie ramec
# v druhom cykle sa vyberie kazdy prvok, ktory sa postupne posuva po jednom prvku
# v tretom cykle sa ramec nasobi sam so sebou pricom sa upravuje prekrytie ramca samim so sebou
for i in range(0, pocetRamcov-1):
    for j in range(0, pocetVzorkov):
        posunRamca = pocetVzorkov - j
        for g in range(0, posunRamca):
            interShift = pocetVzorkov - posunRamca
            tempArray = np.append(tempArray, uloha4Acorr[i][g] * uloha4Acorr[i][interShift + g])
        arrCorr[i][j] = np.sum(tempArray)
        tempArray = 0
    uloha4Acorr[i] = arrCorr[i]

plt.plot(uloha4Acorr[74])
plt.gca().set_xlabel("Pocet vzorkov v ramci")
plt.gca().set_ylabel("y")
plt.gca().set_title("75. ramec z nahravky maskOn")
plt.savefig("maskOnCorrLag")
plt.show()

## Autokorelacia maskOff
uloha4AcorrOff = copy.deepcopy(uloha4Off)
arrCorrOff = np.ndarray((320, 320))
tempArrayOff = np.array([])
interShiftOff = 0

for i in range(0, pocetRamcov-1):
    for j in range(0, pocetVzorkov):
        posunRamcaOff = pocetVzorkov - j
        for g in range(0, posunRamcaOff):
            interShiftOff = pocetVzorkov - posunRamcaOff
            tempArrayOff = np.append(tempArrayOff, uloha4AcorrOff[i][g] * uloha4AcorrOff[i][interShiftOff + g])
        arrCorrOff[i][j] = np.sum(tempArrayOff)
        tempArrayOff = 0
    uloha4AcorrOff[i] = arrCorrOff[i]

### lag segment
# maskOn
uloha4ALags = uloha4Acorr
prahVzorka = int(16000/500)
lagArray = np.array([])
a = 0

for j in range(0, 99):
    lagArray = np.append(lagArray, (16000/(np.argmax(uloha4ALags[j][32:320]) + 32)))

# maskOf
uloha4ALagsOff = uloha4AcorrOff
lagArrayOff = np.array([])

for j in range(0, 99):
    lagArrayOff = np.append(lagArrayOff, (16000/(np.argmax(uloha4ALagsOff[j][32:320]) + 32)))

plt.plot(lagArray)
plt.plot(lagArrayOff)
plt.gca().set_xlabel("ramce")
plt.gca().set_ylabel("f0")
plt.gca().set_title("Zakladna frekvencia ramcov")
plt.savefig("maskOffCorrFullLag")
plt.show()
mean = np.mean(lagArray)
meanOff = np.mean(lagArrayOff)
print("Stredna hodnota tonu nahravky maskon je ", mean)
print("Stredna hodnota tonu nahravky maskon je", meanOff)

doubleLagArr = copy.deepcopy(lagArray)
for i in range(0, 99):
    if lagArray[i] < mean - (1/3*mean):
        lagArray[i] = np.median(doubleLagArr)

doubleLagArrOff = copy.deepcopy(lagArrayOff)
for i in range(0, 99):
    if lagArrayOff[i] < meanOff - (1/3*meanOff):
        lagArrayOff[i] = np.median(doubleLagArrOff)

plt.plot(lagArray)
plt.plot(lagArrayOff)
plt.gca().set_xlabel("ramce")
plt.gca().set_ylabel("f0")
plt.gca().set_title("Zakladna frekvencia ramcov")
plt.savefig("maskOffCorrFullLagFix")
plt.show()
