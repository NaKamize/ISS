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
osxramce = np.linspace(0, 0.02)

### vsetky ramce pre nasadenu rusku
ramceOn = np.ndarray((pocetRamcov, pocetVzorkov))

for i in range(0, pocetRamcov-1):
    for j in range(0, pocetVzorkov):
        if((i*prekrytie+j) < 16000):
            ramceOn[i][j] = dataOn[i*prekrytie+j]

plt.plot(ramceOn[0])
plt.gca().set_xlabel("Pocet vzorkov v ramci")
plt.gca().set_ylabel("y")
plt.gca().set_title("Druhy ramec z nahravky s ruskom")
plt.savefig("maskOn_ramec")
plt.show()

### vsetky ramce pre nahravku kde je ruska dole
ramceOff = np.ndarray((pocetRamcov, pocetVzorkov))

for i in range(0, pocetRamcov-1):
    for j in range(0, pocetVzorkov):
        if((i*prekrytie+j) < 16000):
            ramceOff[i][j] = dataOff_2[i*prekrytie+j]

plt.plot(ramceOff[0])
plt.gca().set_xlabel("Pocet vzorkov v ramci")
plt.gca().set_ylabel("y")
plt.gca().set_title("Prve ramce nahravok")
plt.savefig("maskoff_ramec")
plt.show()

""""""""""""""""""""
## Uloha 4
""""""""""""""""""""
## maskOn clipping
uloha4On = ramceOn
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

plt.plot(uloha4On[0])
plt.gca().set_xlabel("Pocet vzorkov v ramci")
plt.gca().set_ylabel("y")
plt.gca().set_title("Centralne klipovanie s 70% nahravky s ruskom")
plt.savefig("maskOnclip")
plt.show()

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

plt.plot(uloha4Off[0])
plt.gca().set_xlabel("Pocet vzorkov v ramci")
plt.gca().set_ylabel("y")
plt.gca().set_title("Centralne klipovanie s 70% piateho ramca nahravky bez ruska")
plt.savefig("maskOffclip")
plt.show()

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
plt.gca().set_title("Autokorelacia z nahravky maskOn")
plt.savefig("maskOnCorr")
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

plt.plot(uloha4AcorrOff[0])
plt.axvline(x=15, color='b')
plt.axvline(x=117, ymin=0, ymax=10, color='r')
plt.gca().set_xlabel("Pocet vzorkov v ramci")
plt.gca().set_ylabel("y")
plt.gca().set_title("Autokorelacia prveho ramca z nahravky maskOff")
plt.savefig("maskOffCorr")
plt.show()

### lag segment
# maskOn
uloha4ALags = uloha4Acorr
lagArray = np.array([])

for j in range(0, pocetRamcov-1):
    lagArray = np.append(lagArray, (16000/(np.argmax(uloha4ALags[j][32:320]) + 32)))

# maskOf
uloha4ALagsOff = uloha4AcorrOff
lagArrayOff = np.array([])

for j in range(0, pocetRamcov-1):
    lagArrayOff = np.append(lagArrayOff, (16000/(np.argmax(uloha4ALagsOff[j][32:320]) + 32)))

print("Stredna hodnota oboch nahravok je", np.mean([lagArray, lagArrayOff]))
print("Stredna hodnota nahravky maskon je", np.mean(lagArray))
print("Stredna hodnota nahravky maskof je", np.mean(lagArrayOff))
print("Rozptyl oboch nahravok je ", np.var([lagArray, lagArrayOff]))
print("Rozptyl nahravky maskon je", np.var(lagArray))
print("Rozptyl nahravky maskof je", np.var(lagArrayOff))
plt.plot(lagArray)
plt.plot(lagArrayOff)
plt.gca().set_xlabel("ramce")
plt.gca().set_ylabel("f0")
plt.gca().set_title("Zakladna frekvencia ramcov")
plt.savefig("maskOffCorrFull")
plt.show()
