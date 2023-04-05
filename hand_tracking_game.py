# Import des librairies nécessaires
import cv2
import time
import main as htm

# Création d'un objet VideoCapture pour capturer des images à partir de la caméra par défaut
cap = cv2.VideoCapture(0)

# Initialisation des variables pour suivre le temps et les fps
cTime = 0
pTime = 0

# Création d'un objet handDetector
detector = htm.handDetector()

# Boucle infinie pour lire les images de la caméra et effectuer la détection de la main
while True:
    # Lecture d'une image depuis la caméra
    success, img = cap.read()

    # Utilisation de l'objet handDetector pour détecter les mains dans l'image
    # avec l'option de dessiner ou non les points et les connexions de la main
    img = detector.findHands(img, False)

    # Utilisation de l'objet handDetector pour obtenir une liste des positions des points de la main
    # avec l'option de dessiner ou non les points sur l'image
    lmList = detector.findPosition(img, draw=False)

    # Si la liste des positions n'est pas vide, afficher la position du point d'index 4 (le bout du pouce)
    if len(lmList) != 0:
        print(lmList[4])

    # Mesure du temps écoulé depuis la dernière image
    cTime = time.time()
    # Calcul des fps
    fps = 1 / (cTime - pTime)
    # Mise à jour du temps précédent avec le temps actuel pour le prochain tour de boucle
    pTime = cTime

    # Affichage des fps sur l'image
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Affichage de l'image
    cv2.imshow("Image", img)

    # Attente d'une touche pour quitter le programme
    cv2.waitKey(1)
