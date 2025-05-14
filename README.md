# Modélisation de la parole par LPC

Ce projet a été réalisé dans le cadre d’un BE à l’INSA Toulouse (3MIC). Il porte sur la modélisation, la synthèse et la reconnaissance de signaux vocaux à l’aide du codage linéaire prédictif (LPC).

## 📁 Organisation du projet

```
LPC-Voice-Modeling/
├── main.py              # Script principal
├── wav/                 # Fichiers audio (.wav)
├── figures/             # Graphiques exportés depuis matplotlib
├── rapport/             # Rapport LaTeX et PDF
├── README.md            # Ce fichier
├── requirements.txt     # Liste des dépendances Python
```

## 🚀 Fonctionnalités

- Découpage en trames et fenêtrage du signal
- Estimation des coefficients LPC
- Synthèse du signal à partir des coefficients
- Synthèse croisée entre deux signaux vocaux
- Extraction des formants (pics spectre LPC)
- Détection du voisement et estimation du pitch
- Construction d’un dictionnaire pour reconnaissance automatique
- Reconstruction vocale à partir de l’identification

## ▶️ Utilisation

Installe les dépendances :

```bash
pip install -r requirements.txt
```

Lance le script principal :

```bash
python main.py
```

Tu peux aussi tester les fonctions de traitement audio, reconnaissance, et visualisation individuellement dans `main.py`.

## 🎓 Références

- Maitine Bergounioux : *Mathématiques pour le Traitement du Signal*
- Hyung-Suk Kim : *Linear Predictive Coding is All-Pole Resonance Modeling*
