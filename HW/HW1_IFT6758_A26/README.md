Français (English follows)

## Devoir 1 — Fondements de ML

Rien de piégeux ici : suivez le cours, implémentez les fonctions demandées et faites attention aux typos.

### Environnement

Créez un environnement conda avec les librairies de `requirements.txt` :

```bash
conda create -n ift6758-hw1 python=3.10
conda activate ift6758-hw1
pip install -r requirements.txt
```

Ça garde le code reproductible d'une machine à l'autre.

### Consigne

Remplacez chaque `# TODO` dans [ml_fundamentals.py](ml_fundamentals.py) par votre implémentation, comme décrit dans l'énoncé.

 **Respectez bien le type de retour demandé dans chaque docstring** (float vs numpy array, tuple, etc.) : l'autograder met 0 si le type ne correspond pas.

Le notebook [train_logistic_regression_model.ipynb](train_logistic_regression_model.ipynb) est optionnel, il sert juste à tester vos implémentations sur un exemple concret. Il génère un jeu de données synthétique de classification binaire en 2D, séparé en ensembles d'entraînement et de validation, puis entraîne votre régression logistique par montée de gradient en suivant l'évolution de la perte. Il affiche ensuite la frontière de décision apprise et propose en bonus une analyse de l'effet du taux d'apprentissage sur la convergence.


---

## Assignment 1 — ML Fundamentals

Nothing tricky here: follow the course material, implement the requested functions, and watch out for typos.

### Environment

Create a conda environment with the libraries listed in `requirements.txt`:

```bash
conda create -n ift6758-hw1 python=3.10
conda activate ift6758-hw1
pip install -r requirements.txt
```

This keeps the code reproducible across machines.

### Instructions

Replace each `# TODO` in [ml_fundamentals.py](ml_fundamentals.py) with your implementation, as described in the assignment instructions.

 **Make sure the return type matches what's specified in each docstring** (float vs numpy array, tuple, etc.): the autograder gives 0 if the type is wrong, even when the value is correct.

The notebook [train_logistic_regression_model.ipynb](train_logistic_regression_model.ipynb) is optional, it just lets you test your implementations on a concrete example. It generates a synthetic 2D binary classification dataset split into training and validation sets, then trains your logistic regression via gradient ascent while tracking the loss. It then plots the learned decision boundary and, as a bonus, analyzes the effect of the learning rate on convergence.


