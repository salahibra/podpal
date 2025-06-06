﻿# 🚀 Podpal

Un assistant simple et rapide basé sur l'IA.

---

## 📦 Installation

### 1. Créer un environnement virtuel

```bash
py -3.12 -m venv podpal_env
```
## 1.1 Activater environnement virtuel
```bash
podpal_env\Scripts\activate
```
### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

> ⚠️ Assure-toi que le nom du fichier est bien `requirements.txt` (et non `requirements.text`).

### 3. Ajouter le jeton Hugging Face

Crée un fichier `.env` à la racine du projet et ajoute :

```env
HUGGINGFACE_HUB_TOKEN=hf_************
```

> 🔐 Remplace `************` par ton **token d'accès Hugging Face**.
### 4. python build_podcast_vectorstore.py 
```bash
python build_podcast_vectorstore.py 
```
### 5. Lancer l'application

```bash
python app.py
```
### 6. Lancer l'application web
```bash
python main.py
```

---

## 📁 Structure du projet

```
podpal/
├── podpal_env
├── app.py
├── requirements.txt
├── .env
└── ...
```

---
