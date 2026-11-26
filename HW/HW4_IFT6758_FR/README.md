<!--
Université de Montréal
IFT-6758-A — A23 — Science des données
-->

# Devoir 4

## Barème

| Composante | Fichier requis | Points |
|---|---|---:|
| Code (5 fonctions) | `nlp_code.py` | 50 |
| Figures, cellules exécutées et sorties | `hw4.ipynb` | 10 |
| Rapport (T1.3, T3.1–3.3, T4.4, T5) | `hw4.pdf` | 40 |
| **Total** | | **100** |

Une partie de ce devoir sera évaluée automatiquement. **Ne modifiez pas les signatures des fonctions fournies** (leurs entrées et leurs sorties doivent rester inchangées).

## Soumission

Soumettez **uniquement les trois fichiers requis** indiqués ci-dessus sur Gradescope. N'incluez pas le jeu de données ni d'autres fichiers supplémentaires.

### Consignes importantes

1. Vérifiez attentivement les définitions des fonctions fournies et les sorties attendues.
2. Effectuez toutes les opérations demandées dans le notebook Jupyter, y compris les graphiques et les sorties affichées. Ces éléments font partie du devoir et seront vérifiés.
3. Pour la tâche 5, vous devrez implémenter un modèle de type transformeur. Nous vous recommandons d'utiliser Google Colab avec un environnement d'exécution GPU pour l'entraînement. Vous pouvez d'abord déboguer votre code sur CPU.
4. DistilBERT nécessite des colonnes nommées `text` et `labels`. Conservez les ensembles d'entraînement et de test définis dans les tâches précédentes, et utilisez les critiques prétraitées ainsi que les étiquettes encodées. Vous pouvez sauvegarder ces données dans des fichiers CSV pour les réutiliser.
5. Les réponses limitées à « oui » ou « non », ainsi que les explications trop succinctes, ne sont pas suffisantes. Appuyez vos analyses sur les graphiques demandés et des références lorsque cela est nécessaire.
6. Indiquez votre nom et votre numéro d'étudiant dans `hw4.pdf`. Pour chaque question ou discussion, ajoutez un titre précisant la section correspondante du devoir.

## Tâche 1 : Préparation des données et exploration initiale

Commencez par charger le jeu de données et examiner sa structure et son contenu. Vérifiez la présence de valeurs manquantes (`NaN`) et de lignes en double. Si des doublons existent, conservez leur première occurrence et supprimez les suivantes. Vérifiez que les observations restantes sont uniques.

- Effectuez les opérations requises dans `hw4.ipynb`.
- Répondez aux questions suivantes dans `hw4.pdf` :
  - Combien d'observations le jeu de données contient-il ?
  - Combien de valeurs distinctes la colonne cible contient-elle ?
  - Y a-t-il des valeurs manquantes (`NaN`) ?
  - Y a-t-il des critiques en double ? Si oui, combien de lignes en double avez-vous trouvées ?

## Tâche 2 : Prétraitement des données

Complétez la fonction `preprocess_text()` dans `nlp_code.py`. Cette fonction prend une chaîne de caractères et renvoie sa version prétraitée. Appliquez-la à toutes les critiques.

- Complétez `nlp_code.py:preprocess_text()`.
- Effectuez les opérations requises dans `hw4.ipynb`.

## Tâche 3 : Analyse exploratoire des données (EDA)

Explorez les critiques et le jeu de données. Réalisez les analyses nécessaires et appuyez vos conclusions sur les graphiques et les explications appropriés.

- Complétez `nlp_code.py:review_lengths()`.
- Complétez `nlp_code.py:word_frequency()`.
- Effectuez les opérations requises dans `hw4.ipynb`.
- Répondez aux questions suivantes dans `hw4.pdf`, en vous appuyant sur les graphiques et les analyses demandés :
  - Comment les valeurs cibles sont-elles réparties ? Le jeu de données est-il approximativement équilibré ?
  - Toutes les critiques ont-elles la même longueur ?
  - Quelle est la longueur moyenne des séquences ?
  - Quels sont les 20 mots les plus fréquents ?
  - Quels sont les 20 mots les moins fréquents ?
  - D'après votre analyse exploratoire, pensez-vous qu'il sera facile de classifier ces critiques ? Pourquoi ou pourquoi pas ?

## Tâche 4 : Extraction de caractéristiques et préparation de la cible

Encodez la colonne cible à l'aide de `encode_sentiment()`. Faites attention au type de données des valeurs de `sentiment` encodées. Extrayez les caractéristiques textuelles avec TF-IDF, entraînez un classificateur simple comme modèle de référence, puis explorez l'explicabilité du modèle avec LIME. Complétez la fonction `explain_instance()`.

- Complétez `nlp_code.py:encode_sentiment()`.
- Complétez `nlp_code.py:explain_instance()`.
- Effectuez les opérations requises dans `hw4.ipynb`.
- Traitez les points suivants dans `hw4.pdf`, en les appuyant sur les graphiques et analyses nécessaires :

**TF-IDF (fréquence des termes – fréquence inverse des documents)**

- Expliquez les inconvénients de cette représentation.
- Proposez une autre approche susceptible d'offrir une meilleure représentation numérique des mots du corpus. Expliquez pourquoi elle pourrait être plus efficace.

**LIME (Local Interpretable Model-agnostic Explanations)**

- Incluez la visualisation LIME obtenue et interprétez-la.

## Tâche 5 : Exploration d'un modèle transformeur

- Implémentez DistilBERT et produisez les scores et la visualisation demandés dans `hw4.ipynb`.
- Conservez les traces de votre implémentation et de ses résultats dans le notebook exécuté.
- Traitez les points suivants dans `hw4.pdf` :

1. **Implémentation :** Implémentez et entraînez `distilbert-base-uncased`. Utilisez les mêmes ensembles d'entraînement et de test que pour le classificateur Naïve Bayes. Vous pouvez enregistrer les données prétraitées dans des fichiers CSV avec des colonnes nommées `text` et `labels`, puis les charger à l'aide de `load_dataset` de la bibliothèque `datasets`. Veillez à respecter le format d'entrée attendu par le transformeur.
2. **Ajustement des hyperparamètres :** Explorez deux approches différentes pour améliorer les performances du modèle (p. ex., le nombre d'époques, le taux d'apprentissage ou la décroissance des poids). Décrivez votre méthodologie et justifiez vos choix. Les modifications aléatoires sans justification seront pénalisées. Présentez les résultats du meilleur modèle au point suivant.
3. **Évaluation et comparaison :** Présentez **l'exactitude (accuracy), la précision (precision), le rappel (recall) et le score F1** de votre meilleur modèle. Expliquez le sens de ces métriques et comment interpréter les différences entre modèles. Incluez une **matrice de confusion sous forme de carte thermique (heatmap)**. Comparez DistilBERT au classificateur Naïve Bayes à l'aide des métriques et des cartes thermiques des matrices de confusion, et discutez de l'importance de l'amélioration obtenue.
