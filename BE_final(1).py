import numpy as np
import librosa 
import soundfile as sf
import scipy.signal as sps
import scipy.linalg as la
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

# II.1 Décomposition en trames (OLA)
##II.1.1 Creation des trames
def CreateTrame(x, w, R=0.5):
    """ Découpe un signal x en trames avec une fenêtre w et un recouvrement R """
    n = len(x)
    nw = len(w)
    step = int(nw * (1 - R))
    nb = max(1, (n - nw) // step + 1)
    
    B = np.zeros((nb, nw))
    for j in range(nb):
        deb = j * step
        B[j, :] = x[deb:deb + nw] * w
    
    return B
##Test CreateTrame signal x echantilloné à 8kHz, fenetre de Hann, nw =240

def test_CreateTrame():
    fs = 8000  # Fréquence d'échantillonnage
    duration = 1.0  # Durée en secondes
    n = int(fs * duration)
    x = np.random.normal(0, 1, n)  # Bruit blanc
    nw = 240  # Taille de la fenêtre
    w = np.hanning(nw)  # Fenêtre de Hann
    R = 0.5  # Taux de recouvrement

    B = CreateTrame(x, w, R)
    
    plt.figure()
    plt.imshow(B, aspect='auto', origin='lower')
    plt.title("Matrice des trames (CreateTrame)")
    plt.xlabel("Échantillons")
    plt.ylabel("Trames")
    plt.savefig("Create_trames.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("Test CreateTrame terminé.")

#test_CreateTrame()

## II.1.2 Reconsitution d'un signal à partir des trames 
def AddTrame(B, w, R=0.5):
    """ Reconstruit un signal à partir des trames B en utilisant un recouvrement R """
    nb, nw = B.shape
    step = int(nw * (1 - R))
    n = (nb - 1) * step + nw
    x = np.zeros(n)
    wsum = np.zeros(n)
    
    for j in range(nb):
        deb = j * step
        x[deb:deb + nw] += B[j, :]
        wsum[deb:deb + nw] += w
    
    x /= np.maximum(wsum, 1e-10)
    return x

##Test AddTrame pour vérifié si on a le même signal

def test_AddTrame():
    fs = 8000
    duration = 1.0
    n = int(fs * duration)
    x = np.random.normal(0, 1, n)
    nw = 240
    w = np.hanning(nw)
    R = 0.5

    B = CreateTrame(x, w, R)
    x_reconstructed = AddTrame(B, w, R)
    
    plt.figure()
    plt.plot(x, label='Original')
    plt.plot(x_reconstructed, label='Reconstruit', linestyle='dashed')
    plt.legend()
    plt.title("Comparaison signal original vs reconstruit")
    plt.savefig("comparaison_original_reconstruit.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("Test AddTrame terminé.")

#test_AddTrame() 

# II.2 Encodage LPC
def CreateX(x, p):
    """ Construit la matrice X pour le modèle LPC """
    nw = len(x)
    X = np.zeros((nw - 1, p))
    for j in range(nw - 1):
        X[j, :min(p, j+1)] = x[j:j - p:-1] if j >= p else x[j::-1]
    return X

def SolveLPC(x, p):
    """ Résout les coefficients LPC par moindres carrés """
    X = CreateX(x, p)
    b = x[1:]
    a, _, _, _ = la.lstsq(X, b)
    e = b - X @ a
    sigma2 = np.var(e)
    return a, sigma2

def EncodeLPC(x, p, w):
    """ Encode un signal x en utilisant un modèle LPC de paramètre p """
    B = CreateTrame(x, w)
    nb = B.shape[0]
    A = np.zeros((p, nb))
    G = np.zeros(nb)
    for j in range(nb):
        A[:, j], G[j] = SolveLPC(B[j, :], p)
    return A, G

# 3. Décodage LPC
def RunAR(a, g, m):
    """ Génère un signal à partir d'un modèle AR """
    ##F0 = 100  # Fréquence fondamentale
    ##fs = 8000  # Fréquence d'échantillonnage
    p = len(a)
    e = np.sqrt(g) * np.random.randn(m)
    ##e = np.sin(2 * np.pi * F0 * np.arange(m) / fs)
    x = np.zeros(m)
    for n in range(p, m):
        x[n] = np.dot(a, x[n - p:n][::-1]) + e[n]
    return x

def DecodeLPC(A, G, w):
    """ Reconstruit un signal LPC à partir des coefficients """
    nb = A.shape[1]
    nw = len(w)
    B = np.zeros((nb, nw))
    for j in range(nb):
        B[j, :] = RunAR(A[:, j], G[j], nw)
    return AddTrame(B, w)

##III Tests
def compute_L1_L2_errors(original, reconstructed):
    """Calcule les erreurs L1 (absolue) et L2 (quadratique) entre l'original et le reconstruit."""
    min_length = min(len(original), len(reconstructed))
    original = original[:min_length]
    reconstructed = reconstructed[:min_length]
    
    error_L1 = np.mean(np.abs(original - reconstructed))
    error_L2 = np.sqrt(np.mean((original - reconstructed) ** 2))
    return error_L1, error_L2

def test_partie_3():
    
    # Charger un fichier test
    x, sr = librosa.load("speech.wav", sr=None)
    x = 0.9 * x / np.max(np.abs(x))  # Normalisation
    x = librosa.resample(x, orig_sr=sr, target_sr=8000)  # Rééchantillonnage
    w = np.hanning(240)  # Fenêtre de Hann
    
    p_values = [6,50, 24]
    for p in p_values:
        print(f"Test avec p = {p}")
        A, G = EncodeLPC(x, p, w)
        G = np.clip(G, 1e-6, 0.05)
        x_decoded = DecodeLPC(A, G, w)
        x_decoded = np.clip(x_decoded, -1.0, 1.0)  # Hard clipping
        x_decoded = x_decoded / np.max(np.abs(x_decoded)) * 0.9  # Normalisation douce
        error_L1, error_L2 = compute_L1_L2_errors(x, x_decoded)
        print(f"Erreur L1 : {error_L1}")
        print(f"Erreur L2 : {error_L2}")

        # Sauvegarde du fichier
        sf.write(f"speechLPC_p{p}.wav", x_decoded, 8000)
        
        plt.figure()
        plt.plot(x, label='Original')
        plt.plot(x_decoded, label='Décodé', linestyle='dashed')
        plt.legend()
        plt.title(f"Comparaison signal original vs décodé (p={p})")
        plt.savefig("comparaison_original_decode.png", dpi=300, bbox_inches='tight')
        plt.show()
    print("Test de la partie 3 terminé.")

#test_partie_3()


def synthese_croisee_secure():

    # Paramètres
    p = 40
    w = np.hanning(240)
    nw = len(w)
    signal_modulation = "speech.wav"
    porteurs = ["creak.wav", "bubbles.wav", "racing.wav"]

    # Charger le signal de modulation
    x_m, sr_m = librosa.load(signal_modulation, sr=None)
    x_m = 0.9 * x_m / np.max(np.abs(x_m))
    x_m = librosa.resample(x_m, orig_sr=sr_m, target_sr=8000)

    for fichier in porteurs:
        print(f"\n🔁 Synthèse croisée avec {fichier}")
        x_c, sr_c = librosa.load(fichier, sr=None)
        x_c = 0.9 * x_c / np.max(np.abs(x_c))
        x_c = librosa.resample(x_c, orig_sr=sr_c, target_sr=8000)

        # Encode les deux
        A_m, _ = EncodeLPC(x_m, p, w)
        _, G_c = EncodeLPC(x_c, p, w)

        # On prend le min nb de trames pour éviter mismatch
        nb_min = min(A_m.shape[1], len(G_c))

        A_m = A_m[:, :nb_min]
        G_c = G_c[:nb_min]

        # 🔄 Reconstruction manuelle trame par trame
        B_cross = np.zeros((nb_min, nw))
        for j in range(nb_min):
            B_cross[j, :] = RunAR(A_m[:, j], G_c[j], nw)

        x_cross = AddTrame(B_cross, w)

        # 🔈 Normalisation
        x_cross = np.clip(x_cross, -1.0, 1.0)
        x_cross = x_cross / np.max(np.abs(x_cross)) * 0.9

        # Sauvegarde
        name = f"synthese_cross_{fichier.split('.')[0]}.wav"
        sf.write(name, x_cross, 8000)

        # Affichage
        plt.figure(figsize=(10, 4))
        plt.plot(x_m, label="Signal Modulation", alpha=0.5)
        plt.plot(x_c, label="Signal Porteur", alpha=0.5)
        plt.plot(x_cross, label="Synthèse Croisée", linestyle='--')
        plt.title(f"Synthèse croisée avec {fichier}")
        plt.legend()
        plt.tight_layout()
        plt.savefig("Synthese_croisee.png", dpi=300, bbox_inches='tight')
        plt.show()

#synthese_croisee_secure()

#Deuxième Partie de TP
# 2. Visualisation des coefficients LPC
def Visu(Coefs, nfenetres, fe=8000, nb_formants=3):
    """Affiche le spectre du filtre LPC en Hertz et annote les formants détectés."""
    import matplotlib.pyplot as plt
    from scipy.signal import freqz, find_peaks
    import numpy as np

    for i in range(nfenetres):
        w, h = freqz(1, Coefs[i, :], worN=512)
        freqs = w * fe / (2 * np.pi)  # Conversion en Hz
        magnitude_db = 20 * np.log10(np.abs(h))

        # Tracer la courbe
        plt.plot(freqs, magnitude_db, label=f'Trame {i}')

        # Détection des pics (formants)
        peaks, _ = find_peaks(np.abs(h), distance=20)
        top_peaks = peaks[np.argsort(np.abs(h[peaks]))[::-1][:nb_formants]]

        # Annotations
        for peak in top_peaks:
            f = freqs[peak]
            amp = magnitude_db[peak]
            plt.plot(f, amp, 'ko')  # point noir
            plt.text(f + 30, amp, f'{int(f)} Hz', fontsize=8, color='black')

    plt.title("Spectre du filtre LPC")
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# 4. Détection du voisement
def Voisement(cor):
    """Détecte si une trame est voisée ou non en utilisant l'autocorrélation."""
    peak = np.max(cor[int(len(cor) / 10):])  # Éviter le pic à zéro
    threshold = 0.3 * np.max(cor)  # Seuil empirique
    return 1 if peak > threshold else 0

# 6. Estimation du pitch
def EstimationPitch(cor, fs=8000):
    """Estime la fréquence fondamentale (pitch) d'une trame voisée."""
    peaks, _ = sps.find_peaks(cor[int(len(cor) / 10):])
    if len(peaks) > 0:
        pitch_period = peaks[0] + len(cor) // 10  # Décalage compensé
        return fs / pitch_period
    return 0  # Si non voisée

# 7. Correction du voisement
def CorrectionVoisement(vois, pitch, nfenetres):
    """Corrige le voisement en utilisant les fenêtres voisines."""
    for i in range(1, nfenetres - 1):
        if vois[i] == 0 and vois[i - 1] == 1 and vois[i + 1] == 1:
            vois[i] = 1
            pitch[i] = (pitch[i - 1] + pitch[i + 1]) / 2
    return vois, pitch

# 8. Construction de l'excitation
def ConstructionExcitation(gain, pitch, vois, taillefenetre, fs=8000):
    """Construit l'excitation en fonction du voisement."""
    E = np.zeros(taillefenetre)
    for i in range(len(vois)):
        if vois[i]:  # Voisé -> train de Dirac
            periode = int(fs / pitch[i])
            E[i::periode] = gain[i]
        else:  # Non voisé -> Bruit blanc
            E[i] = gain[i] * np.random.randn()
    return E

# 9. Synthèse LPC
def SyntheseLPC(E, Coefs, taillefenetre, p):
    """Reconstruit un son à partir de l'excitation et des coefficients LPC."""
    nb = Coefs.shape[0]
    x = np.zeros(nb * taillefenetre)
    for i in range(nb):
        x[i * taillefenetre:(i + 1) * taillefenetre] = sps.lfilter([1], Coefs[i, :], E[i * taillefenetre:(i + 1) * taillefenetre])
    return x

# 13. Extraction des paramètres des filtres
def ParametresFiltres(Coef, N, n):
    """Extrait les positions et amplitudes des pics du filtre LPC."""
    w, h = sps.freqz(1, Coef, worN=N)
    peaks, _ = sps.find_peaks(abs(h), distance=10)
    positions = w[peaks][:n]
    valeurs = abs(h[peaks])[:n]
    return positions, valeurs

# 14. Comparaison des coefficients
def ComparaisonCoefs(P1, V1, P2, V2):
    """Calcule une distance entre deux jeux de coefficients LPC."""
    distances = [np.abs(p1 - p2) + np.abs(v1 - v2) for p1, v1, p2, v2 in zip(P1, V1, P2, V2)]
    return sum(sorted(distances)[:4])

# 15. Création du dictionnaire
def Dictionnaire():
    """Charge tous les fichiers .wav du dossier et construit un dictionnaire de sons."""
    wav_files = [f for f in os.listdir() if f.endswith(".wav") and f != "speech_french.wav"]  # Trouver tous les fichiers .wav
    print(f"📂 Chargement du dictionnaire avec {len(wav_files)} fichiers audio...")

    Pos, Val = [], []
    for file in wav_files:
        x, sr = librosa.load(file, sr=8000)
        A, _ = EncodeLPC(x, 12, np.hanning(240))
        pos, val = ParametresFiltres(A[:, 0], 512, 5)
        Pos.append(pos)
        Val.append(val)

    print("✅ Dictionnaire construit avec succès !")
    return Pos, Val

# 16. Reconnaissance du son
def DeterminationSon(Pos1, Val1, Pos2, Val2, nb):
    """Détermine le son le plus proche dans le dictionnaire."""
    distances = [ComparaisonCoefs(Pos1, Val1, Pos2[i], Val2[i]) for i in range(nb)]
    return np.argmin(distances)

# 17. Estimation du son
def EstimationSon(name, Pos2, Val2):
    """Estime le son dans chaque fenêtre et retourne les résultats."""
    x, sr = librosa.load(name, sr=8000)
    A, _ = EncodeLPC(x, 12, np.hanning(240))
    results = [DeterminationSon(*ParametresFiltres(A[:, i], 512, 5), Pos2, Val2, len(Pos2)) for i in range(A.shape[1])]
    return results

# 18. Correction du son
def CorrectionSon(Son):
    """Corrige la reconnaissance vocale en exploitant les fenêtres voisines."""
    for i in range(1, len(Son) - 1):
        if Son[i] != Son[i - 1] and Son[i] != Son[i + 1]:
            Son[i] = Son[i - 1]
    return Son


def test_speech_lpc():
    """Test de l'encodage et de la synthèse LPC sur speech.wav."""
    file = "speech_french.wav"
    x, sr = librosa.load(file, sr=8000)
    w = np.hanning(240)  # Fenêtre de Hann
    p = 40  # Ordre du modèle LPC
    
    # Encodage LPC
    A, G = EncodeLPC(x, p, w)
    print("Encodage LPC terminé.")
    
    # Visualisation des coefficients
    Visu(A.T, min(5, A.shape[1]))
    
    # Reconstruction du signal
    x_reconstructed = DecodeLPC(A, G, w)
    x_reconstructed = np.clip(x_reconstructed, -1.0, 1.0)
    x_reconstructed = x_reconstructed / np.max(np.abs(x_reconstructed)) * 0.9
    # Sauvegarde
    sf.write("speech_reconstructed_french.wav", x_reconstructed, 8000)
    
    # Affichage
    plt.figure()
    plt.plot(x, label='Original')
    plt.plot(x_reconstructed, label='Reconstruit', linestyle='dashed')
    plt.legend()
    plt.title("Comparaison signal original vs reconstruit")
    plt.show()
    print("Test de synthèse terminé.")

def test_voisement_pitch():
    """Test de la détection du voisement et estimation du pitch sur speech.wav."""
    file = "speech.wav"
    x, sr = librosa.load(file, sr=8000)
    w = np.hanning(240)  # Fenêtre de Hann
    p = 40  # Ordre du modèle LPC
    B = CreateTrame(x, w)
    
    voisements = []
    pitchs = []
    for trame in B:
        cor = np.correlate(trame, trame, mode='full')
        cor = cor[len(cor) // 2:]
        voisements.append(Voisement(cor))
        pitchs.append(EstimationPitch(cor))
    
    plt.figure()
    plt.plot(voisements, label="Voisement (1=voisé, 0=non voisé)")
    plt.legend()
    plt.title("Détection du voisement")
    plt.show()
    
    plt.figure()
    plt.plot(pitchs, label="Pitch estimé")
    plt.legend()
    plt.title("Estimation du pitch")
    plt.show()
    print("Test voisement et pitch terminé.")

def test_dictionnaire():
    """Test de la reconnaissance LPC avec dictionnaire."""
    Pos, Val = Dictionnaire(["bubbles.wav", "creak.wav", "racing.wav"])
    print("Dictionnaire construit avec 3 sons.")
    
    son_reconnu = EstimationSon("speech2.wav", Pos, Val)
    print("Reconnaissance terminée :", son_reconnu)
    
    son_corrige = CorrectionSon(son_reconnu)
    print("Reconnaissance corrigée :", son_corrige)

# Lancer les tests
test_speech_lpc()
#test_voisement_pitch()
#test_dictionnaire()


def reconstruct_speech_with_lpc():
    """Reconstruit speech.wav en utilisant la méthode LPC avec dictionnaire."""
    
    Pos, Val = Dictionnaire()
    print("Dictionnaire construit avec tous les sons du dictionnaire.")
    
    son_reconnu = EstimationSon("speech_french.wav", Pos, Val)
    son_corrige = CorrectionSon(son_reconnu)
    
    # Reconstruire le signal basé sur la reconnaissance
    x, sr = librosa.load("speech_french.wav", sr=8000)
    w = np.hanning(240)
    p = 50
    A, G = EncodeLPC(x, p, w)
    G = np.clip(G, 1e-6, 0.05)
    
    x_reconstructed = DecodeLPC(A, G, w)
    x_reconstructed = np.clip(x_reconstructed, -1.0, 1.0)
    x_reconstructed = x_reconstructed / np.max(np.abs(x_reconstructed)) * 0.9
    sf.write("speechLPC_french.wav", x_reconstructed, 8000)

    # Calcul des erreurs
    error_L1, error_L2 = compute_L1_L2_errors(x, x_reconstructed)
    print(f"Erreur L1 : {error_L1}")
    print(f"Erreur L2 : {error_L2}")
    
    print("Fichier speechLPC.wav généré avec la méthode LPC améliorée.")
    
    # Affichage comparaison
    plt.figure()
    plt.plot(x, label='Original')
    plt.plot(x_reconstructed, label='Reconstruit LPC', linestyle='dashed')
    plt.legend()
    plt.title("Comparaison signal original vs reconstruit avec dictionnaire")
    plt.show()

# Lancer la reconstruction
#reconstruct_speech_with_lpc()