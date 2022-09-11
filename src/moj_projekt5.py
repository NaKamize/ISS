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

fsOn, dataOn = wavfile.read('../audio/maskon_tone.wav')
fsoff, dataOff = wavfile.read('../audio/maskoff_tone.wav')
#vybranie si 2. sekundy zo signalu s ruskom
dataOn = dataOn[16000:32000]
#vybranie si 1.a 2. sekundy zo signalu bez ruska
dataOff_2 = dataOff[31000:47000]
# ustrednenie
dataOn = dataOn - np.mean(dataOn)
dataOff_2 = dataOff_2 - np.mean(dataOff_2)
# normalizacia
dataOn = dataOn / np.abs(dataOn).max()
dataOff_2 = dataOff_2 / np.abs(dataOff_2).max()
# ramce
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

""""""""""""""""""""
## Uloha 5
""""""""""""""""""""
#############
# DFT
#############
padMaskOn = copy.deepcopy(ramceOn)
impArray = np.ndarray((99, 1024), dtype=np.complex_)
resultOn = np.ndarray((99, 1024), dtype=np.complex_)
resultOnFin = np.ndarray((99, 1024))

for i in range(0, pocetRamcov-1):
    impArray[i] = np.pad(ramceOn[i], (0, 1024 - len(ramceOn[i]) % 1024), 'constant')


def dft(dx):
    dftsum = 0j
    resultOnfunc = np.ndarray((99, 1024), dtype=np.complex_)
    for i in range(0, 99):
        for f in range(0, 1024):
            for k in range(0, 1024):
                dftsum = dftsum + (dx[i][k] * (
                            np.cos((2 * np.pi * k * f) / 1024) - 1j * np.sin((2 * np.pi * k * f) / 1024)))
            resultOnfunc[i][f] = dftsum
            dftsum = 0j
    return resultOnfunc


resultOn = dft(impArray)

for i in range(0, 99):
    for j in range(0, 1024):
        resultOnFin[i][j] = 10 * np.log10((np.abs(resultOn[i][j])**2))

resultOnFin = resultOnFin.transpose()
plt.imshow(resultOnFin[:][:512], origin='lower', extent=(0, 1, 0, 8000), aspect='auto')
plt.gca().set_xlabel("Cas")
plt.gca().set_ylabel("Frekvencia")
plt.gca().set_title("Spectrogram s ruskou DFT")
plt.colorbar()
plt.savefig("spectrogramDFT")
plt.show()

##############
# FFT
##############
nonImpArray = np.ndarray((99, 1024), dtype=np.complex_)
test = np.ndarray((99, 1024))

for i in range(0, 99):
    nonImpArray[i] = np.fft.fft(ramceOn[i], 1024)

for i in range(0, 99):
    for j in range(0, 1024):
        test[i][j] = 10*np.log10((np.abs(nonImpArray[i][j])**2))

fftNonImpOn = test.transpose()
plt.imshow(fftNonImpOn[:][:512],  origin='lower', extent=(0, 1, 0, 8000), aspect='auto')
plt.gca().set_xlabel("Cas")
plt.gca().set_ylabel("Frekvencia")
plt.gca().set_title("Spectrogram s ruskou")
plt.colorbar()
plt.savefig("spectrogramFFT")
plt.show()

### Paeed maskOff
padMaskOff = copy.deepcopy(ramceOff)
impArray1 = np.ndarray((99, 1024), dtype=np.complex_)
impArrayOff = np.ndarray((99, 1024), dtype=np.complex_)
hustotaOff = np.ndarray((99, 1024))

for i in range(0, pocetRamcov-1):
    impArray1[i] = np.pad(ramceOff[i], (0, 1024 - len(ramceOff[i]) % 1024), 'constant')

impArrayOff = dft(impArray1)

for i in range(0, 99):
    for j in range(0, 1024):
        hustotaOff[i][j] = 10*np.log10((np.abs(impArrayOff[i][j])**2))

hustotaOff = hustotaOff.transpose()
plt.imshow(hustotaOff[:][:512],  origin='lower', extent=(0, 1, 0, 8000), aspect='auto')
plt.gca().set_xlabel("Cas")
plt.gca().set_ylabel("Frekvencia")
plt.gca().set_title("Spectrogram bez rusky")
plt.colorbar()
plt.show()

nonImpArrayOff = np.ndarray((99, 1024), dtype=np.complex_)
testOff = np.ndarray((99, 1024))

for i in range(0, 99):
    nonImpArrayOff[i] = np.fft.fft(ramceOff[i], 1024)

for i in range(0, 99):
    for j in range(0, 1024):
        testOff[i][j] = 10*np.log10((np.abs(nonImpArrayOff[i][j])**2))

fftNonImp = testOff.transpose()
plt.imshow(fftNonImp[:][:512],  origin='lower', extent=(0, 1, 0, 8000), aspect='auto')
plt.gca().set_xlabel("Cas")
plt.gca().set_ylabel("Frekvencia")
plt.gca().set_title("Spectrogram bez rusky")
plt.colorbar()
plt.savefig("spectrogramBez")
plt.show()
""""""""""""""""""""
## Uloha 6
""""""""""""""""""""
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
plt.gca().set_title("Frekvencna charakteristika rusky")
plt.savefig("frekChar")
plt.show()

""""""""""""""""""""
## Uloha 7
""""""""""""""""""""
response = np.fft.ifft(impulgraf, 1024)

plt.plot(osx, response[:512].real)
plt.gca().set_xlabel("Frekvencia")
plt.gca().set_ylabel("y")
plt.gca().set_title("Impulzna odozva rusky")
plt.savefig("impOdozv")
plt.show()

impIdftRes = np.ndarray((1024), dtype=np.complex_)


def idft(ix):
    impIdft = np.ndarray((1024), dtype=np.complex_)
    idftsum = 0j
    for k in range(0, 1024):
        for f in range(0, 1024):
            idftsum = idftsum + 1/1024 * ix[f] * (np.cos((2*np.pi*k*f)/1024) + 1j * np.sin((2*np.pi*k*f)/1024))
        impIdft[k] = idftsum
        idftsum = 0j
    return impIdft


impIdftRes = idft(impulgraf)
impIdftRes = impIdftRes.transpose()

plt.plot(osx, impIdftRes[:512].real)
plt.gca().set_xlabel("Frekvencia")
plt.gca().set_ylabel("y")
plt.gca().set_title("Impulzna odozva rusky")
plt.show()

""""""""""""""""""""
## Uloha 8
""""""""""""""""""""
filterOut = np.array([])
a_0 = np.ndarray((1))
a_0[0] = 1
osxnahravka = np.linspace(0, 1, len(dataOff))
filterOutToneOff = signal.lfilter(response[:512], a_0, dataOff)

plt.plot(osxnahravka, filterOutToneOff.real)
plt.gca().set_xlabel("Frekvencia")
plt.gca().set_ylabel("y")
plt.gca().set_title("Nahravka tonu so simulovanou ruskou")
plt.savefig("nahravkaTonSoSimRuskou")
plt.show()

fsOffsen, dataSentenceOff = wavfile.read('../audio/maskoff_sentence.wav')
fsOnsen, dataSentenceOn = wavfile.read('../audio/maskon_sentence.wav')

filterOutSentenceOff = signal.lfilter(response[:512], a_0, dataSentenceOff)

dlzkaNahr = len(dataSentenceOn)/16000
osxn = np.linspace(0, dlzkaNahr, len(dataSentenceOff))
osxnOn = np.linspace(0, dlzkaNahr, len(dataSentenceOn))
# maskof sentence
plt.plot(osxn, filterOutSentenceOff.real)
plt.gca().set_xlabel("Frekvencia")
plt.gca().set_ylabel("y")
plt.gca().set_title("Nahravka vety so simulovanou ruskou")
plt.savefig("nahravkaSoSimRuskou")
plt.show()

plt.plot(osxn, dataSentenceOff)
plt.gca().set_xlabel("Frekvencia")
plt.gca().set_ylabel("y")
plt.gca().set_title("Nahravka vety bez ruska")
plt.savefig("nahravkabezRuska")
plt.show()

# maskon sentence
plt.plot(osxnOn, dataSentenceOn)
plt.gca().set_xlabel("Frekvencia")
plt.gca().set_ylabel("y")
plt.gca().set_title("Nahravka vety s ruskou a simulovanou ruskou")
plt.savefig("Porovnanie")
plt.show()

filterOutToneOff = filterOutToneOff.real
filterOutSentenceOff = filterOutSentenceOff.real
filterOutToneOff = filterOutToneOff.astype('int16')
filterOutSentenceOff = filterOutSentenceOff.astype('int16')

sf.write("sim_maskon_tone.wav", filterOutToneOff, 16000)
sf.write("sim_maskon_sentence.wav", filterOutSentenceOff, 16000)