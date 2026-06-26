# AI-Powered Phishing Detection System

Projet academique complet permettant de detecter des attaques de phishing a partir de deux types d'informations :

- le contenu d'un email ;
- une URL suspecte.

L'application ne donne pas seulement une reponse comme "phishing" ou "legitime". Elle explique aussi pourquoi elle a pris cette decision, par exemple en montrant les mots suspects dans un email ou les caracteristiques dangereuses d'une URL.

Ce projet a ete concu pour une demonstration universitaire. L'objectif est donc d'avoir un code clair, une architecture propre, une interface utilisable et une documentation comprehensible meme pour une personne qui decouvre le machine learning.

---

## 1. Idee Generale Du Projet

Le phishing est une technique utilisee par des attaquants pour tromper une personne. Le but est souvent de voler :

- un mot de passe ;
- des informations bancaires ;
- un compte email ;
- une identite numerique ;
- des donnees personnelles.

Exemple d'email suspect :

```text
Your account has been blocked. Please verify your password immediately.
Click here: http://secure-account-update.example.com/login
```

Dans cet exemple, plusieurs signaux sont dangereux :

- le message cree un sentiment d'urgence ;
- il demande une verification de compte ;
- il parle de mot de passe ;
- il contient une URL qui semble imiter un service officiel.

Le projet analyse donc les textes et les URLs pour reperer ce type de signaux.

---

## 2. Ce Que Fait L'application

L'application permet de :

- charger des datasets d'emails ;
- nettoyer le texte des emails ;
- entrainer des modeles de machine learning ;
- analyser une URL avec des caracteristiques de securite ;
- calculer un score de risque ;
- afficher une prediction claire ;
- expliquer les raisons de la prediction ;
- sauvegarder l'historique des analyses ;
- afficher des statistiques et graphiques ;
- tester le code avec `pytest`.

L'application contient une interface web construite avec Streamlit. Cela permet de tester le projet directement dans le navigateur.

---

## 3. Technologies Utilisees

### Python

Python est le langage principal du projet. Il est tres utilise en data science, intelligence artificielle, cybersecurite et developpement d'outils rapides.

Version recommandee :

```text
Python 3.12+
```

### Streamlit

Streamlit sert a creer l'interface web.

Dans ce projet, Streamlit permet :

- d'afficher une page d'accueil ;
- de saisir un email ;
- de saisir une URL ;
- de lancer une analyse ;
- d'afficher le score de risque ;
- d'afficher les raisons de la decision ;
- de voir l'historique ;
- de consulter les statistiques.

Commande pour lancer l'interface :

```bash
streamlit run app/streamlit_app.py
```

### scikit-learn

`scikit-learn` est la bibliotheque utilisee pour le machine learning.

Elle sert ici a :

- convertir les textes en nombres avec TF-IDF ;
- entrainer des modeles ;
- faire des predictions ;
- calculer les metriques de performance.

Modeles utilises :

- Logistic Regression ;
- Multinomial Naive Bayes.

### pandas

`pandas` sert a lire et manipuler les donnees tabulaires.

Dans ce projet, il permet de :

- lire les fichiers CSV ;
- nettoyer les colonnes ;
- fusionner plusieurs datasets ;
- sauvegarder l'historique ;
- afficher des tableaux dans Streamlit.

### numpy

`numpy` sert aux calculs numeriques. Il est utilise indirectement par les modeles et directement pour certaines operations de scoring.

### nltk

`nltk` sert au traitement du langage naturel.

Dans ce projet, il peut fournir :

- une liste de stopwords ;
- un lemmatizer WordNet.

Si les corpus NLTK ne sont pas installes, l'application continue de fonctionner avec des listes de secours.

Messages possibles :

```text
NLTK stopwords corpus is not installed; using fallback list.
NLTK WordNet corpus is not installed; using fallback lemmatizer.
```

Ces messages ne sont pas bloquants.

Pour les supprimer, on peut installer les corpus :

```bash
python -m nltk.downloader wordnet stopwords omw-1.4
```

### matplotlib

`matplotlib` sert a generer des graphiques :

- matrices de confusion ;
- courbes ROC ;
- importance des features.

### wordcloud

`wordcloud` sert a generer un nuage de mots a partir du dataset d'emails. Les mots les plus frequents apparaissent plus grands.

### joblib

`joblib` sert a sauvegarder les modeles entraines.

Exemples de fichiers generes :

- `models/email_model.joblib` ;
- `models/email_vectorizer.joblib` ;
- `models/url_model.joblib`.

### pytest

`pytest` sert a lancer les tests automatiques.

Commande :

```bash
pytest
```

---

## 4. Architecture Du Projet

```text
Hackathon26/
  app/
  data/
    raw/
    processed/
  docs/
    screenshots/
  models/
  notebooks/
  reports/
    figures/
  src/
    email/
    explainability/
    ml/
    preprocessing/
    url/
    utils/
  tests/
  config.py
  requirements.txt
  README.md
  .gitignore
  LICENSE
```

### `app/`

Contient l'interface Streamlit.

Fichier principal :

```text
app/streamlit_app.py
```

Ce fichier gere les pages :

- Home ;
- Analyse ;
- History ;
- Statistics ;
- About.

### `data/`

Contient les donnees du projet.

```text
data/raw/
```

Ce dossier est prevu pour recevoir les datasets bruts.

```text
data/processed/
```

Ce dossier peut recevoir des donnees nettoyees ou transformees.

### `models/`

Contient les modeles entraines.

Ces fichiers sont generes apres l'entrainement :

```text
models/email_model.joblib
models/email_vectorizer.joblib
models/url_model.joblib
models/email_metrics.json
models/url_metrics.json
```

### `reports/`

Contient les rapports et figures.

Exemples :

```text
reports/figures/email_confusion_matrix.png
reports/figures/url_confusion_matrix.png
reports/figures/email_roc_curve.png
reports/figures/url_roc_curve.png
reports/figures/email_wordcloud.png
```

### `src/`

Contient le code principal du projet.

Chaque sous-dossier a une responsabilite precise.

### `tests/`

Contient les tests automatiques.

Les tests verifient notamment :

- le nettoyage du texte ;
- l'extraction des features URL ;
- le calcul du risque ;
- les predictions ;
- le chargement des donnees ;
- certains helpers de l'interface.

### `config.py`

Centralise les chemins, constantes et seuils de risque.

Cela evite d'avoir des chemins ecrits partout dans le code.

---

## 5. Installation

Depuis le dossier du projet :

```bash
cd Hackathon26
```

Creer un environnement virtuel :

```bash
python -m venv .venv
```

Activer l'environnement virtuel sur Windows :

```bash
.venv\Scripts\activate
```

Installer les dependances :

```bash
pip install -r requirements.txt
```

---

## 6. Datasets

Le projet supporte les fichiers CSV deja presents dans le dossier :

```text
SpamAssasin.csv
CEAS_08.csv
phishing_email_detection_2026_dataset.csv
```

Le code cherche aussi les datasets dans :

```text
data/raw/
```

Cela permet de garder une organisation propre si on ajoute d'autres fichiers.

### Colonnes attendues pour les emails

Le loader accepte plusieurs formats.

Colonnes possibles pour le texte :

- `subject` ;
- `body` ;
- `sender` ;
- `sender_email` ;
- `receiver` ;
- `urls`.

Colonnes possibles pour le label :

- `label` ;
- `is_phishing` ;
- `target` ;
- `class`.

Le label doit representer :

- `0` pour un email legitime ;
- `1` pour un email de phishing.

### Dataset URL

Pour un dataset URL dedie, les fichiers possibles sont :

```text
urls.csv
phishtank.csv
PhishTank.csv
```

Colonnes attendues :

- `url` ou `urls` ;
- `label`, `is_phishing`, `target` ou `class`.

Si aucun dataset URL n'est disponible, le projet essaie d'extraire des URLs depuis les emails. Si aucune URL exploitable n'est trouvee, il utilise un petit dataset de demonstration integre pour que l'application reste testable.

---

## 7. Pipeline Email

Le pipeline email sert a transformer un texte brut en donnees utilisables par un modele.

### Etape 1 : Chargement

Le code lit les fichiers CSV disponibles.

Fichier concerne :

```text
src/utils/data_loading.py
```

### Etape 2 : Nettoyage

Le texte est nettoye dans :

```text
src/preprocessing/text.py
```

Le nettoyage effectue :

- suppression du HTML ;
- suppression des URLs ;
- passage en minuscules ;
- suppression de la ponctuation ;
- suppression des nombres ;
- tokenisation ;
- suppression des stopwords ;
- lemmatisation.

### Etape 3 : Vectorisation TF-IDF

Un modele de machine learning ne comprend pas directement le texte.

TF-IDF transforme le texte en valeurs numeriques.

Exemple simplifie :

```text
"verify password account"
```

devient une suite de nombres representant l'importance de chaque mot.

### Etape 4 : Entrainement

Deux modeles sont compares :

- Logistic Regression ;
- Multinomial Naive Bayes.

Le meilleur modele est choisi selon le score F1.

Fichier :

```text
src/ml/train_email.py
```

### Etape 5 : Sauvegarde

Apres entrainement, le modele et le vectorizer sont sauvegardes dans :

```text
models/
```

---

## 8. Pipeline URL

Une URL peut etre suspecte meme sans analyser le contenu de la page.

Le projet extrait donc des indicateurs a partir de l'URL elle-meme.

Fichier :

```text
src/url/features.py
```

Features extraites :

- longueur totale de l'URL ;
- longueur du hostname ;
- longueur du chemin ;
- nombre de chiffres ;
- nombre de points ;
- nombre de slashs ;
- nombre de sous-domaines ;
- nombre de parametres de requete ;
- presence de `@` ;
- presence de `-` ;
- presence de `%` ;
- presence de `=` ;
- presence d'une adresse IP ;
- utilisation de HTTP ou HTTPS ;
- nombre de mots-cles suspects ;
- entropie.

### Exemple

URL :

```text
http://192.168.1.4/verify/password
```

Signaux suspects :

- utilise HTTP ;
- contient une adresse IP ;
- contient `verify` ;
- contient `password`.

---

## 9. Explicabilite

L'explicabilite permet de comprendre la decision du modele.

### Pour les emails

Le projet affiche les mots les plus influents.

Exemples :

- `verify` ;
- `password` ;
- `urgent` ;
- `login` ;
- `account`.

Fichier :

```text
src/explainability/email_explainer.py
```

### Pour les URLs

Le projet affiche les features qui augmentent le risque.

Exemples :

- URL tres longue ;
- trop de chiffres ;
- adresse IP detectee ;
- HTTP au lieu de HTTPS ;
- mots-cles suspects ;
- forte entropie.

Fichier :

```text
src/explainability/url_explainer.py
```

---

## 10. Score De Risque

Le projet calcule :

- la probabilite email ;
- la probabilite URL ;
- la probabilite globale ;
- le niveau de risque.

Niveaux :

```text
LOW
MEDIUM
HIGH
CRITICAL
```

Le score global prend le signal le plus risque entre l'email et l'URL.

Fichier :

```text
src/ml/risk.py
```

---

## 11. Entrainer Les Modeles

Depuis le dossier `Hackathon26` :

```bash
python -m src.ml.train_all
```

Cette commande entraine :

- le modele email ;
- le modele URL ;
- les metriques ;
- certaines figures.

On peut aussi entrainer depuis l'interface Streamlit avec le bouton :

```text
Train models
```

Quand l'entrainement est termine, l'interface affiche :

```text
Models are ready.
```

---

## 12. Lancer L'interface

Commande :

```bash
streamlit run app/streamlit_app.py
```

Streamlit affiche ensuite une URL locale :

```text
http://localhost:8501
```

Il suffit d'ouvrir cette adresse dans le navigateur.

### Pages de l'interface

#### Home

Affiche l'etat general du projet :

- modeles prets ou non ;
- nombre d'analyses ;
- score moyen ;
- nombre de decisions phishing.

#### Analyse

Permet de tester :

- un email ;
- une URL ;
- les deux en meme temps.

La page affiche :

- la prediction ;
- le niveau de risque ;
- le score email ;
- le score URL ;
- le score global ;
- les raisons ;
- un bouton pour telecharger le rapport.

#### History

Affiche l'historique local des analyses.

Fichier genere :

```text
reports/analysis_history.csv
```

#### Statistics

Affiche les metriques et figures generees.

#### About

Explique rapidement le but du projet.

---

## 13. Tester Le Projet

Commande :

```bash
pytest
```

Les tests permettent de verifier que les parties importantes fonctionnent encore apres modification.

Exemples de tests :

- verifier que le HTML est supprime d'un email ;
- verifier qu'une IP est detectee dans une URL ;
- verifier que le score de risque est correct ;
- verifier qu'un modele retourne une probabilite.

---

## 14. Commandes Pour Tester Apres Avoir Clone Le Projet

Cette section est faite pour une personne qui recupere le projet depuis GitHub et qui veut le tester sur son ordinateur.

### Etape 1 : Cloner le projet

```bash
git clone https://github.com/deliasamm829-spec/Hackathon26.git
cd Hackathon26
```

Si la personne veut utiliser directement la branche `mkbranch` :

```bash
git checkout mkbranch
```

### Etape 2 : Creer un environnement virtuel

Sur Windows :

```bash
python -m venv .venv
.venv\Scripts\activate
```

Sur macOS ou Linux :

```bash
python -m venv .venv
source .venv/bin/activate
```

Un environnement virtuel permet d'installer les bibliotheques du projet sans modifier l'installation Python globale de l'ordinateur.

### Etape 3 : Installer les dependances

```bash
pip install -r requirements.txt
```

Cette commande installe les bibliotheques necessaires :

- Streamlit ;
- scikit-learn ;
- pandas ;
- numpy ;
- nltk ;
- matplotlib ;
- wordcloud ;
- joblib ;
- pytest.

### Etape 4 : Entrainer les modeles

```bash
python -m src.ml.train_all
```

Cette commande genere les fichiers de modeles dans le dossier `models/`.

Si la personne prefere utiliser l'interface, elle peut aussi lancer Streamlit puis cliquer sur le bouton `Train models`.

### Etape 5 : Lancer les tests automatiques

```bash
pytest
```

Si tous les tests passent, cela veut dire que les parties principales du projet fonctionnent :

- preprocessing ;
- extraction URL ;
- calcul de risque ;
- prediction ;
- chargement des donnees ;
- helpers de l'interface.

### Etape 6 : Lancer l'interface web

```bash
streamlit run app/streamlit_app.py
```

Streamlit affiche ensuite une adresse comme :

```text
http://localhost:8501
```

Il faut ouvrir cette adresse dans un navigateur.

### Resume Des Commandes

```bash
git clone https://github.com/deliasamm829-spec/Hackathon26.git
cd Hackathon26
git checkout mkbranch
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.ml.train_all
pytest
streamlit run app/streamlit_app.py
```

Sur macOS ou Linux, remplacer :

```bash
.venv\Scripts\activate
```

par :

```bash
source .venv/bin/activate
```

---

## 15. Mettre Le Projet Sur GitHub

Avant de publier :

```bash
git status
```

Ajouter les fichiers :

```bash
git add .
```

Creer un commit :

```bash
git commit -m "Build AI phishing detection system"
```

Envoyer sur GitHub :

```bash
git push
```

Pour pousser directement sur la branche `mkbranch` :

```bash
git push origin mkbranch
```

### Important

Le fichier `.gitignore` ignore certains fichiers generes :

```text
models/*.joblib
reports/analysis_history.csv
reports/figures/*.png
```

Cela veut dire que les utilisateurs qui clonent le projet devront entrainer les modeles eux-memes avec :

```bash
python -m src.ml.train_all
```

ou avec le bouton `Train models` dans l'interface.

C'est un bon choix pour un projet academique, car cela rend l'entrainement reproductible.

---

## 16. Commandes Rapides

Installer :

```bash
pip install -r requirements.txt
```

Entrainer :

```bash
python -m src.ml.train_all
```

Lancer l'interface :

```bash
streamlit run app/streamlit_app.py
```

Tester :

```bash
pytest
```

---

## 17. Limites Actuelles

Ce projet est solide pour une demonstration academique, mais il ne doit pas etre considere comme un systeme de cybersecurite industriel sans ameliorations.

Limites :

- les performances dependent fortement des datasets ;
- les attaquants peuvent modifier leurs techniques ;
- les URLs raccourcies ne sont pas encore developpees ;
- le contenu reel des sites web n'est pas analyse ;
- les pieces jointes ne sont pas analysees.

---

## 18. Ameliorations Possibles

Idees pour aller plus loin :

- ajouter un dataset PhishTank plus complet ;
- ajouter une base d'URLs legitimes plus grande ;
- analyser les pieces jointes ;
- ajouter une API REST ;
- ajouter Docker ;
- ajouter un systeme de login ;
- ajouter une base de donnees au lieu d'un CSV ;
- ajouter SHAP pour des explications plus avancees ;
- suivre les versions des modeles ;
- deployer l'application en ligne.

---

## 19. Resume Simple

Ce projet est une application complete de detection de phishing.

Elle utilise :

- Python pour le code ;
- pandas pour les donnees ;
- scikit-learn pour le machine learning ;
- NLTK pour le traitement du texte ;
- Streamlit pour l'interface ;
- matplotlib et wordcloud pour les graphiques ;
- pytest pour les tests.

Le fonctionnement general est :

```text
Dataset -> Nettoyage -> Entrainement -> Evaluation -> Interface -> Analyse -> Explication
```

Quand l'application affiche :

```text
Models are ready.
```

cela signifie que le projet est pret a analyser des emails et des URLs.
