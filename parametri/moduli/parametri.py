import os
import librosa
import librosa.display
import pypianoroll
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

def sine(frq=1,amp=1):
    '''
    visualizza una sinusoide di 200 punti
    '''
    t = np.arange(200) / 200             # Periodo in samples
    out = amp * np.sin(2 * np.pi * (frq * t))
    x = np.arange(0., 1., 1/frq)
    x = np.append(x,[1])
    plt.figure(figsize=(10, 2))
    plt.plot(t, out, 'b-')    
    plt.xlim([0, 1])                     # Limiti plot x                   
    plt.ylim([-1, 1])                    # Limiti plot y
    plt.xticks(x)                        # Griglia x
    plt.yticks([-1,0,1])                 # Griglia y
    plt.grid(True)                       # Grigla
    plt.xlabel('Tempo')                  # Testo x
    plt.ylabel('Ampiezza')               # Testo y
    plt.tight_layout()                   # Stile del Plot
    return plt.show()

def vsine(frq=1,amp=1):
    '''
    visualizza una sinusoide di 200 punti
    '''
    t = np.arange(200) / 200             # Periodo in samples
    out = amp * np.sin(2 * np.pi * (frq * t))
    x = np.arange(0., 1., 1/frq)
    x = np.append(x,[1])
    plt.figure(figsize=(1.5, 2))
    plt.plot(out, t, 'b-')    
    plt.xlim([-1, 1])              # Limiti plot x                   
    plt.ylim([0, 1])                     # Limiti plot y
    plt.xticks([-1,0,1])                 # Griglia x
    plt.yticks([])                       # Griglia y
    plt.grid(True)                       # Grigla
    plt.xlabel('Ampiezza')               # Testo x
    plt.ylabel('Tempo')                  # Testo y
    plt.tight_layout()                   # Stile del Plot
    return plt.show()

def img(nsd=12):
    xa = np.arange(int(125)) / 100        # Genera lista x analogico
    outa = np.sin(2*np.pi * xa )
    outa = outa + np.sin(2*np.pi * 1.4 * xa )
    outa = outa + np.sin(2*np.pi * 3.2 * xa )
    outa = outa / 2.53
    
    xd = np.arange(int(nsd)+1) / nsd       # Genera lista x digitale
    outd = np.sin(2*np.pi * xd )
    outd = outd + np.sin(2*np.pi * 1.4 * xd )
    outd = outd + np.sin(2*np.pi * 3.2 * xd )
    outd = outd / 2.53

    plt.figure(figsize=(10, 2))                 # Genera la figura size x y
    plt.plot(xa, outa,'-', color="blue")        # Plot analogico
    plt.plot(xd, outd,'o',  color="red")        # Plot punti digitali
    
    plt.plot(xa,np.full(125,0.9),'-', label="picco", color="red", linewidth=0.5) # Plot linea picco
    plt.plot(xa,np.full(125,0.57),'-', label="rms", color="green",linewidth=0.5) # Plot linea rms

    plt.xlim([0, 1])                  # Limiti plot x                   
    plt.ylim([-1, 1])                    # Limiti plot y
    plt.xticks([])                       # Griglia x
    plt.yticks([0,0.57,0.9])             # Griglia y
    plt.grid(True)                       # Grigla
#    plt.xlabel('Tempo (secondi)')       # Testo x
    plt.ylabel('Ampiezza')               # Testo y
    plt.legend(loc='upper right')
    plt.tight_layout()                   # Stile del Plot
    return plt.show()

def amp(nsd=12):
    xd   = np.arange(int(nsd)) / nsd   # Genera lista x digitale
    outd = np.sin(2*np.pi * xd )
    outd = outd + np.sin(2*np.pi * 1.4 * xd )
    outd = outd + np.sin(2*np.pi * 3.2 * xd )
    outd = outd / 2.53
    return np.round_(outd,2)

def sf(r_path):
    fc = 11025                        # Frequenza di campionamento
    path = os.path.abspath(r_path)
    x, sr = librosa.load(path, sr=fc) # Assegna le ampiezze relative a x
    t = np.arange(x.shape[0]) / fc    # Recupera il numero di righe (numero di campioni)

    a = np.round_(np.amax(np.abs(x)),2)          # Ampiezza di picco
    print('Ampiezza di picco: ' + str(a))

    rms = np.round_(np.sqrt(np.mean(x**2)), 2)   # Root mean square
    print('Root Mean Square: ' + str(rms))

    plt.figure(figsize=(10, 1.9))
    plt.plot(t, x, color='gray')

    plt.plot(t, np.full(t.shape[0],a),'-', label="picco", color="red", linewidth=0.5) # Plot linea picco
    plt.plot(t, np.full(t.shape[0],rms),'-', label="rms", color="blue",linewidth=0.5) # Plot linea rms
    
    plt.xlabel('Tempo')
    plt.ylabel('Ampiezza')
    plt.xlim([t[0], t[-1]])
    plt.ylim([-1, 1])
    plt.tick_params(direction='in')
    plt.legend(loc='lower right')
    plt.tight_layout()
    return plt.show()

def fourier():
    fc = 35                   # Rata di campionamento
    ts = 1.0/fc                  # Passo di campionamento
    t = np.arange(0,1,ts)        # Crea una lista di x

    a = 0.4*np.sin(2*np.pi*1*t+0) # Array con valori di sinusoide
    b = 0.1*np.sin(2*np.pi*3*t+0)
    c = 0.3*np.sin(2*np.pi*4*t+0)
    d = 0.15*np.sin(2*np.pi*5*t+0)
    e = a+b+c+d
# ---------------------------------------------
# Computa FFT e IFFT

    X = fft(a)        # Computa FFT (Tempo ---> Frequenza)
    N = len(X)        # Numero di Bins
    n = np.arange(N)  # Genera Array di interi pari al numero di Bins
    T = N/fc          # Delta tra le frequenze dei Bins

# ----- Arrays FFT
    freq = n/T                # Array frequenze Bins riscalato
    magn = 4/fc * np.abs(X)   # Array magnitudine dei parziali riscalato

# ----- Arrays IFFT
    dur = np.arange(len(e))            # Array indici (x) del sound file
    amp = ifft(X)                      # Computa FFT inversa (Frequenza ---> Tempo) 
                                       # Array ampiezze istantanee
# ---------------------------------------------
# Plots

    plt.figure(figsize = (12, 7))    # Il size della finestra intera
    
# ---------------------------------------------
    plt.subplot(521)
    plt.plot(dur, amp, 'o-', color="red", linewidth=0.5,markersize=4) # Linea
    plt.xlim([0, len(e)])                          # Limiti plot x                   
    plt.ylim([-1, 1])                              # Limiti plot y
    plt.xticks([0,len(e)])                         # Griglia x
    plt.yticks([-1,0,1])                           # Griglia y
    plt.grid(True)                                 # Grigla
    plt.xlabel('+')
    plt.ylabel('Ampiezza')
# ---------------------------------------------    
    plt.subplot(522)                 # Per mettere più plots nella finestra (linee, colonne, plot_attuale)
    markerline, stemlines, baseline = plt.stem(freq, magn, linefmt='-', markerfmt="C0 ", basefmt="C0") # barre 
    plt.setp(stemlines, 'linewidth', 7)
    plt.xlim([0, 10])                # Limiti plot x                   
    plt.ylim([0, 1])                 # Limiti plot y
    plt.xticks([0,1,2,3,4,5,6,7,8,9,10])               # Griglia x
    plt.yticks([0.0,1.0])            # Griglia y
    plt.grid(True)                   # Grigla
    plt.xlabel('+')
    plt.ylabel('Magnitudine')
# ---------------------------------------------
# ---------------------------------------------
    X = fft(b)        # Computa FFT (Tempo ---> Frequenza)
    N = len(X)        # Numero di Bins
    n = np.arange(N)  # Genera Array di interi pari al numero di Bins
    T = N/fc          # Delta tra le frequenze dei Bins

# ----- Arrays FFT
    freq = n/T                # Array frequenze Bins riscalato
    magn = 4/fc * np.abs(X)   # Array magnitudine dei parziali riscalato

# ----- Arrays IFFT
    dur = np.arange(len(e))            # Array indici (x) del sound file
    amp = ifft(X)                      # Computa FFT inversa (Frequenza ---> Tempo) 
                                       # Array ampiezze istantanee
        
    plt.subplot(523)
    plt.plot(dur, amp, 'o-', color="red", linewidth=0.5,markersize=4) # Linea
    plt.xlim([0, len(e)])                          # Limiti plot x                   
    plt.ylim([-1, 1])                              # Limiti plot y
    plt.xticks([0,len(e)])                         # Griglia x
    plt.yticks([-1,0,1])                           # Griglia y
    plt.grid(True)                                 # Grigla
    plt.xlabel('+')
    plt.ylabel('Ampiezza')           
# ---------------------------------------------
    plt.subplot(524)                 # Per mettere più plots nella finestra (linee, colonne, plot_attuale)
    markerline, stemlines, baseline = plt.stem(freq, magn, linefmt='-', markerfmt="C0 ", basefmt="C0") # barre 
    plt.setp(stemlines, 'linewidth', 7)
    plt.xlim([0, 10])                # Limiti plot x                   
    plt.ylim([0, 1])                 # Limiti plot y
    plt.xticks([0,1,2,3,4,5,6,7,8,9,10])               # Griglia x
    plt.yticks([0.0,1.0])            # Griglia y
    plt.grid(True)                   # Grigla
    plt.xlabel('+')
    plt.ylabel('Magnitudine')
# ---------------------------------------------   
# ---------------------------------------------
    X = fft(c)        # Computa FFT (Tempo ---> Frequenza)
    N = len(X)        # Numero di Bins
    n = np.arange(N)  # Genera Array di interi pari al numero di Bins
    T = N/fc          # Delta tra le frequenze dei Bins

# ----- Arrays FFT
    freq = n/T                # Array frequenze Bins riscalato
    magn = 4/fc * np.abs(X)   # Array magnitudine dei parziali riscalato

# ----- Arrays IFFT
    dur = np.arange(len(e))            # Array indici (x) del sound file
    amp = ifft(X)                      # Computa FFT inversa (Frequenza ---> Tempo) 
                                       # Array ampiezze istantanee
        
    plt.subplot(525)
    plt.plot(dur, amp, 'o-', color="red", linewidth=0.5,markersize=4) # Linea
    plt.xlim([0, len(e)])                          # Limiti plot x                   
    plt.ylim([-1, 1])                              # Limiti plot y
    plt.xticks([0,len(e)])                         # Griglia x
    plt.yticks([-1,0,1])                           # Griglia y
    plt.grid(True)                                 # Grigla
    plt.xlabel('+')
    plt.ylabel('Ampiezza')           
# ---------------------------------------------
    plt.subplot(526)                 # Per mettere più plots nella finestra (linee, colonne, plot_attuale)
    markerline, stemlines, baseline = plt.stem(freq, magn, linefmt='-', markerfmt="C0 ", basefmt="C0") # barre 
    plt.setp(stemlines, 'linewidth', 7)
    plt.xlim([0, 10])                # Limiti plot x                   
    plt.ylim([0, 1])                 # Limiti plot y
    plt.xticks([0,1,2,3,4,5,6,7,8,9,10])               # Griglia x
    plt.yticks([0.0,1.0])            # Griglia y
    plt.grid(True)                   # Grigla
    plt.xlabel('+')
    plt.ylabel('Magnitudine')
    
# ---------------------------------------------   
# ---------------------------------------------
    X = fft(d)        # Computa FFT (Tempo ---> Frequenza)
    N = len(X)        # Numero di Bins
    n = np.arange(N)  # Genera Array di interi pari al numero di Bins
    T = N/fc          # Delta tra le frequenze dei Bins

# ----- Arrays FFT
    freq = n/T                # Array frequenze Bins riscalato
    magn = 4/fc * np.abs(X)   # Array magnitudine dei parziali riscalato

# ----- Arrays IFFT
    dur = np.arange(len(e))            # Array indici (x) del sound file
    amp = ifft(X)                      # Computa FFT inversa (Frequenza ---> Tempo) 
                                       # Array ampiezze istantanee
        
    plt.subplot(527)
    plt.plot(dur, amp, 'o-', color="red", linewidth=0.5,markersize=4) # Linea
    plt.xlim([0, len(e)])                          # Limiti plot x                   
    plt.ylim([-1, 1])                              # Limiti plot y
    plt.xticks([0,len(e)])                         # Griglia x
    plt.yticks([-1,0,1])                           # Griglia y
    plt.grid(True)                                 # Grigla
    plt.xlabel('=')
    plt.ylabel('Ampiezza')           
# ---------------------------------------------
    plt.subplot(528)                 # Per mettere più plots nella finestra (linee, colonne, plot_attuale)
    markerline, stemlines, baseline = plt.stem(freq, magn, linefmt='-', markerfmt="C0 ", basefmt="C0") # barre 
    plt.setp(stemlines, 'linewidth', 7)
    plt.xlim([0, 10])                # Limiti plot x                   
    plt.ylim([0, 1])                 # Limiti plot y
    plt.xticks([0,1,2,3,4,5,6,7,8,9,10])               # Griglia x
    plt.yticks([0.0,1.0])            # Griglia y
    plt.grid(True)                   # Grigla
    plt.xlabel('=')
    plt.ylabel('Magnitudine')
    
# ---------------------------------------------   
# ---------------------------------------------
    X = fft(e)        # Computa FFT (Tempo ---> Frequenza)
    N = len(X)        # Numero di Bins
    n = np.arange(N)  # Genera Array di interi pari al numero di Bins
    T = N/fc          # Delta tra le frequenze dei Bins

# ----- Arrays FFT
    freq = n/T                # Array frequenze Bins riscalato
    magn = 4/fc * np.abs(X)   # Array magnitudine dei parziali riscalato

# ----- Arrays IFFT
    dur = np.arange(len(e))            # Array indici (x) del sound file
    amp = ifft(X)                      # Computa FFT inversa (Frequenza ---> Tempo) 
                                       # Array ampiezze istantanee
        
    plt.subplot(529)
    plt.plot(dur, amp, 'o-', color="red", linewidth=0.5,markersize=4) # Linea
    plt.xlim([0, len(e)])                          # Limiti plot x                   
    plt.ylim([-1, 1])                              # Limiti plot y
    plt.xticks([0,len(e)])                         # Griglia x
    plt.yticks([-1,0,1])                           # Griglia y
    plt.grid(True)                                 # Grigla
    plt.xlabel('Tempo')
    plt.ylabel('Ampiezza')           
# ---------------------------------------------
    plt.subplot(5,2,10)                 # Per mettere più plots nella finestra (linee, colonne, plot_attuale)
    markerline, stemlines, baseline = plt.stem(freq, magn, linefmt='-', markerfmt="C0 ", basefmt="C0") # barre 
    plt.setp(stemlines, 'linewidth', 7)
    plt.xlim([0, 10])                # Limiti plot x                   
    plt.ylim([0, 1])                 # Limiti plot y
    plt.xticks([0,1,2,3,4,5,6,7,8,9,10])               # Griglia x
    plt.yticks([0.0,1.0])            # Griglia y
    plt.grid(True)                   # Grigla
    plt.xlabel('Freq')
    plt.ylabel('Magnitudine')

    plt.tight_layout()
    return plt.show()

def plotFFT(y,fc,scala):
    X = fft(y)        # Computa FFT (Tempo ---> Frequenza)
    N = len(X)        # Numero di Bins
    n = np.arange(N)  # Genera Array di interi pari al numero di Bins
    T = N/fc          # Delta tra le frequenze dei Bins

# ----- FFT
    freq = n/T                # Array frequenze Bins riscalato
    magn = scala/fc * np.abs(X) # Array magnitudine dei parziali riscalato

# ----- IFFT
    dur = np.arange(len(y))   # Array con tutte le x del sound file
    amp = ifft(X)             # Computa FFT inversa (Frequenza ---> Tempo) 

# ---------------------------------------------
# Plots

    plt.figure(figsize = (12, 2))      # Il size della finestra intera

    plt.subplot(121)                   # Primo subplot della figura (linee, colonne, plot_attuale)

    markerline, stemlines, baseline = plt.stem(freq, magn, linefmt='-', markerfmt="C0 ", basefmt="C0") # barre 
    plt.setp(stemlines, 'linewidth', 1)

    plt.xlim([0, fc/2])                # Limiti plot x (diviso due per nascondere lo specchio FFT)                  
    plt.ylim([0, 1])                   # Limiti plot y
    plt.xticks([0,fc/2])               # Griglia x
    plt.yticks([0.0,1.0])              # Griglia y
    plt.grid(True)                     # Grigla
    plt.xlabel('Freq (Hz)')
    plt.ylabel('Magnitudine')

# ---------------------------------------------

    plt.subplot(122)                               # Secondo subplot 
    plt.plot(dur, amp, color="red", linewidth=0.5) # Linea
    plt.xlim([0, len(y)])                          # Limiti plot x                   
    plt.ylim([-1, 1])                              # Limiti plot y
    plt.xticks([0, len(y)])                        # Griglia x
    plt.yticks([-1,0,1])                           # Griglia y
    plt.grid(True)                                 # Grigla
    plt.xlabel('Tempo (campioni)')
    plt.ylabel('Ampiezza')

    plt.tight_layout()
    return plt.show()

def puro():
    fc = 11050                   # Rata di campionamento
    ts = 1.0/fc                  # Passo di campionamento
    t = np.arange(0,1,ts)        # Crea una lista di x

    y = 0.8*np.sin(2*np.pi*2*t+0) # Array con valori di sinusoide
#    y += 0.1*np.sin(2*np.pi*4*t+0)
#    y += 0.15*np.sin(2*np.pi*7*t+0)
#    y += 0.3*np.sin(2*np.pi*6*t+0)
#    y += 0.05*np.sin(2*np.pi*9*t+0)

# ---------------------------------------------
# Computa FFT e IFFT

    X = fft(y)        # Computa FFT (Tempo ---> Frequenza)
    N = len(X)        # Numero di Bins
    n = np.arange(N)  # Genera Array di interi pari al numero di Bins
    T = N/fc          # Delta tra le frequenze dei Bins

# ----- Arrays FFT
    freq = n/T                # Array frequenze Bins riscalato
    magn = 2/fc * np.abs(X)   # Array magnitudine dei parziali riscalato

# ----- Arrays IFFT
    dur = np.arange(len(y))            # Array indici (x) del sound file
    amp = ifft(X)                      # Computa FFT inversa (Frequenza ---> Tempo) 
                                       # Array ampiezze istantanee
# ---------------------------------------------
# Plots

    plt.figure(figsize = (12, 2))    # Il size della finestra intera

    plt.subplot(121)                 # Per mettere più plots nella finestra (linee, colonne, plot_attuale)
    markerline, stemlines, baseline = plt.stem(freq, magn, linefmt='-', markerfmt="C0 ", basefmt="C0") # barre 
    plt.setp(stemlines, 'linewidth', 3)
    plt.xlim([0, 10])                # Limiti plot x                   
    plt.ylim([0, 1])                 # Limiti plot y
    plt.xticks([0,1,2,3,4,5,6,7,8,9,10])               # Griglia x
    plt.yticks([0.0,1.0])            # Griglia y
    plt.grid(True)                   # Grigla
    plt.xlabel('Freq (Hz)')
    plt.ylabel('Magnitudine')

# ---------------------------------------------

    plt.subplot(122)
    plt.plot(dur, amp, color="red", linewidth=0.5) # Linea
    plt.xlim([0, len(y)])                          # Limiti plot x                   
    plt.ylim([-1, 1])                              # Limiti plot y
    plt.xticks([0,len(y)])                         # Griglia x
    plt.yticks([-1,0,1])                           # Griglia y
    plt.grid(True)                                 # Grigla
    plt.xlabel('Tempo (campioni)')
    plt.ylabel('Ampiezza')

    plt.tight_layout()
    return plt.show()

def periodico():
    fc = 11050                   # Rata di campionamento
    ts = 1.0/fc                  # Passo di campionamento
    t = np.arange(0,1,ts)        # Crea una lista di x

    y = 0.4*np.sin(2*np.pi*2*t+0) # Array con valori di sinusoide
    y += 0.1*np.sin(2*np.pi*4*t+0)
    y += 0.3*np.sin(2*np.pi*6*t+0)
    y += 0.15*np.sin(2*np.pi*7*t+0)
    y += 0.05*np.sin(2*np.pi*9*t+0)

# ---------------------------------------------
# Computa FFT e IFFT

    X = fft(y)        # Computa FFT (Tempo ---> Frequenza)
    N = len(X)        # Numero di Bins
    n = np.arange(N)  # Genera Array di interi pari al numero di Bins
    T = N/fc          # Delta tra le frequenze dei Bins

# ----- Arrays FFT
    freq = n/T                # Array frequenze Bins riscalato
    magn = 4/fc * np.abs(X)   # Array magnitudine dei parziali riscalato

# ----- Arrays IFFT
    dur = np.arange(len(y))            # Array indici (x) del sound file
    amp = ifft(X)                      # Computa FFT inversa (Frequenza ---> Tempo) 
                                       # Array ampiezze istantanee
# ---------------------------------------------
# Plots

    plt.figure(figsize = (12, 2))    # Il size della finestra intera

    plt.subplot(121)                 # Per mettere più plots nella finestra (linee, colonne, plot_attuale)
    markerline, stemlines, baseline = plt.stem(freq, magn, linefmt='-', markerfmt="C0 ", basefmt="C0") # barre 
    plt.setp(stemlines, 'linewidth', 3)
    plt.xlim([0, 10])                # Limiti plot x                   
    plt.ylim([0, 1])                 # Limiti plot y
    plt.xticks([0,1,2,3,4,5,6,7,8,9,10])               # Griglia x
    plt.yticks([0.0,1.0])            # Griglia y
    plt.grid(True)                   # Grigla
    plt.xlabel('Freq (Hz)')
    plt.ylabel('Magnitudine')

# ---------------------------------------------

    plt.subplot(122)
    plt.plot(dur, amp, color="red", linewidth=0.5) # Linea
    plt.xlim([0, len(y)])                          # Limiti plot x                   
    plt.ylim([-1, 1])                              # Limiti plot y
    plt.xticks([0,len(y)])                         # Griglia x
    plt.yticks([-1,0,1])                           # Griglia y
    plt.grid(True)                                 # Grigla
    plt.xlabel('Tempo (campioni)')
    plt.ylabel('Ampiezza')

    plt.tight_layout()
    return plt.show()

def noise():
    fc = 100                   # Rata di campionamento
    ts = 2.0/fc                  # Passo di campionamento
    t = np.arange(0,1,ts)        # Crea una lista di x

    y = 0.8*np.random.uniform(low=-1.0, high=1.0, size=(fc,)) # Array con valori random
#    y += 0.1*np.sin(2*np.pi*4*t+0)
#    y += 0.15*np.sin(2*np.pi*7*t+0)
#    y += 0.3*np.sin(2*np.pi*6*t+0)
#    y += 0.05*np.sin(2*np.pi*9*t+0)

# ---------------------------------------------
# Computa FFT e IFFT

    X = fft(y)        # Computa FFT (Tempo ---> Frequenza)
    N = len(X)        # Numero di Bins
    n = np.arange(N)  # Genera Array di interi pari al numero di Bins
    T = N/fc          # Delta tra le frequenze dei Bins

# ----- Arrays FFT
    freq = n/T                # Array frequenze Bins riscalato
    magn = 12/fc * np.abs(X)   # Array magnitudine dei parziali riscalato

# ----- Arrays IFFT
    dur = np.arange(len(y))            # Array indici (x) del sound file
    amp = ifft(X)                      # Computa FFT inversa (Frequenza ---> Tempo) 
                                       # Array ampiezze istantanee
# ---------------------------------------------
# Plots

    plt.figure(figsize = (12, 2))    # Il size della finestra intera

    plt.subplot(121)                 # Per mettere più plots nella finestra (linee, colonne, plot_attuale)
    markerline, stemlines, baseline = plt.stem(freq, magn, linefmt='-', markerfmt="C0 ", basefmt="C0") # barre 
    plt.setp(stemlines, 'linewidth', 3)
    plt.xlim([0, fc])                # Limiti plot x                   
    plt.ylim([0, 1])                 # Limiti plot y
    plt.xticks([0,fc])               # Griglia x
    plt.yticks([0.0,1.0])            # Griglia y
    plt.grid(True)                   # Grigla
    plt.xlabel('Freq (Hz)')
    plt.ylabel('Magnitudine')

# ---------------------------------------------

    plt.subplot(122)
    plt.plot(dur, amp, color="red", linewidth=0.5) # Linea
    plt.xlim([0, len(y)])                          # Limiti plot x                   
    plt.ylim([-1, 1])                              # Limiti plot y
    plt.xticks([0,len(y)])                         # Griglia x
    plt.yticks([-1,0,1])                           # Griglia y
    plt.grid(True)                                 # Grigla
    plt.xlabel('Tempo (campioni)')
    plt.ylabel('Ampiezza')

    plt.tight_layout()
    return plt.show()

def spettrogramma(y,sr):
 
    N, H = 1024, 512
    X = librosa.stft(y, n_fft=N, hop_length=H, win_length=N, window='hanning')
    Y = np.abs(X)

    plt.figure(figsize=(10, 3))
    librosa.display.specshow(librosa.amplitude_to_db(Y, ref=np.max), 
                             y_axis='linear', x_axis='time', sr=sr, hop_length=H)
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Tempo')
    plt.ylabel('Frequenze (Hz)')
    plt.tight_layout()
    return plt.show()

def sinepar(frq, amp, fas):
    
    xa = np.arange(1000) / 1000   # Genera lista x analogico 
        
    out = amp * np.sin(2*np.pi * (frq * xa ) + fas)
    
    fig, ax = plt.subplots(figsize=(12, 2.5))

    ax.plot(xa, out,'-', color="red",linewidth=0.5) # Plot analogico

    ax.set_xlim(-0.1,1.1)               # Range x
    ax.set_ylim(-1.5,1.5)               # Range y
    ax.grid(True)                   # Griglia
    ax.set_xticks([0,1/frq])               # Valori griglia x
    ax.set_yticks([0,amp])         # Valori griglia y

    ax.annotate('Periodo o ciclo', xy=(0.25, 0.25), xytext=(0.02, 1.1), color='blue',size=15)
    ax.annotate('Ampiezza', xy=(0.25, 0.25), xytext=(0.26, amp*0.41), color='green',size=15)
    ax.annotate('Fase', xy=(0, fas*amp-0.1), xytext=(0.1, -0.7), color='red',size=15,
             arrowprops=dict(arrowstyle="->", color="red",connectionstyle="angle3,angleA=0,angleB=90"))

    plt.plot([0, 1/frq], [1, 1], color='blue',linewidth=1.5, linestyle="--")
    plt.plot([0, 1/frq], [1, 1], 'o', color='blue')
    plt.plot([0.25, 0.25], [0, amp], color='green',linewidth=1.5, linestyle="--")
    plt.plot([0.25, 0.25], [0, amp], 'o', color='green')

    plt.plot([0,1/frq], [fas*amp,fas*amp], 'o', color='red')
    plt.tight_layout()                   # Stile del Plot
    return plt.show()

def prsf(m_path, s_path):

    plt.figure(figsize=(18, 2))
  
    plt.subplot(121)
  
    fc = 11025                        # Frequenza di campionamento
    path = os.path.abspath(s_path)
    x, sr = librosa.load(path, sr=fc) # Assegna le ampiezze relative a x
    t = np.arange(x.shape[0]) / fc    # Recupera il numero di righe (numero di campioni)
    plt.plot(t, x, color='blue')      
    plt.xlabel('Sound file - Tempo (secondi)')
    plt.xlim([t[0], t[-1]])
    plt.ylim([-1, 1])
    plt.grid(True)
    plt.yticks([])
    plt.tick_params(direction='in')
        
    multitrack = pypianoroll.read(m_path)
    axs = multitrack.plot()
    plt.gcf().set_size_inches((6.8, 2))
    plt.ylim([50, 84])                    
    plt.ylabel('') 
    plt.xlabel('MIDI file - Tempo (beat)')  
    plt.xlim([0, 237])  
    
    plt.tight_layout()
    return plt.show()

def proll(path):
          
    multitrack = pypianoroll.read(path)
    axs = multitrack.plot()
    plt.gcf().set_size_inches((12, 2.5))
    plt.ylim([50, 84])                    
    plt.ylabel('') 
    plt.xlabel('Tempo (beat)')  
    plt.xlim([0, 237])  
    
    plt.tight_layout()
    return plt.show()

def curve():
    
    a = np.arange(0.001, 1, 0.001)   
    
    plt.figure(figsize=(10, 7))

    plt.subplot(411)
    plt.plot(a, a, '-')
    plt.xlim([0, 1])
    plt.grid([0, 1])
    plt.ylabel('Ampiezza lineare')
    
    plt.subplot(412)
    plt.plot(a, a*127, '-')
    plt.xlim([0, 1])
    plt.grid([0, 1])
    plt.ylabel('Velocity')
 
    plt.subplot(413)
    plt.plot(a, a**4, '-')
    plt.xlim([0, 1])
    plt.grid([0, 1])
    plt.ylabel('Ampiezza quartica')

    plt.subplot(414)
    plt.plot(a, 20*np.log10(a), '-')
    plt.xlim([0, 1])
    plt.grid([0, 1])
    plt.ylabel('Decibels')
    
    plt.tight_layout()
    return plt.show()