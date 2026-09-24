import numpy as np

def calculate_mse(bias, variance, noise_variance):
    """
    FR : Calcule l'erreur quadratique moyenne (MSE) à partir du compromis biais-variance.
    EN : Calculates the Mean Squared Error (MSE) based on the bias-variance trade-off.

    Parameters / Paramètres:
    bias (float): FR : Le biais du modèle. / EN : The bias of the model.
    variance (float): FR : La variance du modèle. / EN : The variance of the model.
    noise_variance (float): FR : La variance de bruit irréductible (sigma^2). / EN : The irreducible noise variance (sigma^2).

    Returns / Retour:
    float: FR : Le MSE calculé. / EN : The calculated MSE.
    """
    # TODO (FR): Implémenter le calcul du MSE
    # TODO (EN): Implement the MSE calculation
    pass

def relu(x):
    """
    FR : Applique la fonction d'activation ReLU (Rectified Linear Unit).
    EN : Applies the Rectified Linear Unit (ReLU) activation function.

    Parameters / Paramètres:
    x (numpy.ndarray or float): FR : La ou les valeur(s) d'entrée. / EN : The input value(s).

    Returns / Retour:
    numpy.ndarray or float: FR : La sortie après application de ReLU. / EN : The output after applying ReLU.
    """
    # TODO (FR): Implémenter la fonction ReLU
    # TODO (EN): Implement the ReLU function
    pass

def sigmoid(x):
    """
    FR : Applique la fonction d'activation Sigmoid pour transformer une sortie en probabilité entre 0 et 1.
    EN : Applies the Sigmoid activation function to transform an output into a probability between 0 and 1.

    Parameters / Paramètres:
    x (numpy.ndarray or float): FR : La ou les valeur(s) d'entrée. / EN : The input value(s).

    Returns / Retour:
    numpy.ndarray or float: FR : La sortie après application de la fonction Sigmoid. / EN : The output after applying the Sigmoid function.
    """
    # TODO (FR): Implémenter la fonction Sigmoid
    # TODO (EN): Implement the Sigmoid function
    pass

def logistic_regression_predict(X, w, w0):
    """
    FR : Calcule la probabilité prédite P(C_1 | X) pour une régression logistique.
    EN : Calculates the predicted probability P(C_1 | X) for logistic regression.

    Parameters / Paramètres:
    X (numpy.ndarray): FR : Les variables d'entrée, de forme (N, D). / EN : The input features of shape (N, D).
    w (numpy.ndarray): FR : Le vecteur de poids, de forme (D,). / EN : The weight vector of shape (D,).
    w0 (float): FR : Le terme de biais (intercept). / EN : The bias term (intercept).

    Returns / Retour:
    numpy.ndarray: FR : Les probabilités prédites, de forme (N,). / EN : The predicted probabilities of shape (N,).
    """
    # TODO (FR): Implémenter la prédiction de la régression logistique en utilisant votre fonction sigmoid
    # TODO (EN): Implement the logistic regression prediction using your sigmoid function
    pass

def cross_entropy_loss(y_true, y_pred):
    """
    FR : Calcule l'erreur d'entropie croisée (log-vraisemblance négative) pour une classification binaire.
    EN : Calculates the cross-entropy error (negative log-likelihood) for binary classification.

    Parameters / Paramètres:
    y_true (numpy.ndarray): FR : Les étiquettes binaires vraies {0, 1}, de forme (N,). / EN : The true binary labels {0, 1} of shape (N,).
    y_pred (numpy.ndarray): FR : Les probabilités prédites, de forme (N,). / EN : The predicted probabilities of shape (N,).

    Returns / Retour:
    float: FR : La perte d'entropie croisée totale. / EN : The total cross-entropy loss.
    """
    # TODO (FR): Implémenter la formule de la perte d'entropie croisée
    # TODO (EN): Implement the cross-entropy loss formula
    pass

def softmax(z):
    """
    FR : Applique la fonction Softmax pour transformer un vecteur de sorties en une distribution de probabilité discrète.
    EN : Applies the Softmax function to transform a vector of outputs into a discrete probability distribution.

    Parameters / Paramètres:
    z (numpy.ndarray): FR : Le vecteur d'entrée, de forme (K,). / EN : The input vector of shape (K,).

    Returns / Retour:
    numpy.ndarray: FR : La distribution de probabilité, de forme (K,). / EN : The probability distribution of shape (K,).
    """
    # TODO (FR): Implémenter la fonction Softmax
    # TODO (EN): Implement the Softmax function
    pass


def logistic_regression_gradient_step(X, y_true, y_pred, w, w0, eta):
    """
    FR : Effectue une étape de montée de gradient pour mettre à jour les poids et le biais d'une régression logistique.
    EN : Performs a single gradient ascent step to update weights and bias for logistic regression.

    Parameters / Paramètres:
    X (numpy.ndarray): FR : Les variables d'entrée, de forme (N, D), où N est le nombre d'exemples et D le nombre de variables. / EN : The input features of shape (N, D), where N is the number of samples and D is the number of features.
    y_true (numpy.ndarray): FR : Les étiquettes binaires vraies {0, 1}, de forme (N,). / EN : The true binary labels {0, 1} of shape (N,).
    y_pred (numpy.ndarray): FR : Les probabilités prédites, de forme (N,). / EN : The predicted probabilities of shape (N,).
    w (numpy.ndarray): FR : Le vecteur de poids actuel, de forme (D,). / EN : The current weight vector of shape (D,).
    w0 (float): FR : Le terme de biais actuel (intercept). / EN : The current bias term (intercept).
    eta (float): FR : Le taux d'apprentissage. / EN : The learning rate.

    Returns / Retour:
    tuple: FR : Un tuple contenant les poids mis à jour (numpy.ndarray) et le biais mis à jour (float). / EN : A tuple containing the updated weights (numpy.ndarray) and updated bias (float).
    """
    # TODO (FR): Implémenter la mise à jour des poids : w^j = w^j + eta * somme_i (y_i - y_pred_i) * x_i^j
    # TODO (EN): Implement the weight update: w^j = w^j + eta * sum_i (y_i - y_pred_i) * x_i^j
    # TODO (FR): Implémenter la mise à jour du biais : w_0 = w_0 + eta * somme_i (y_i - y_pred_i)
    # TODO (EN): Implement the bias update: w_0 = w_0 + eta * sum_i (y_i - y_pred_i)
    pass
