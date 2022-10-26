from scipy.spatial import distance as dist
from collections import OrderedDict
from imutils.video import  FileVideoStream , VideoStream
import numpy as np
import imutils
import time
import cv2 
 
class CentroidTracker:
    def __init__(self, maxDisappeared=50 , maxDistance=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance

# Inscrire un objet
    def register(self, centroid):  
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

# Désinscrire un objet
    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

# Mettre à jour le tracker
    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects 

# initialiser un tableau de centroid d'entrée pour la cadre en cours
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        # boucle sur les rectangles de la boîte englobante
        for (i, (startX, startY, endX, endY)) in enumerate(rects): 
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

# si nous ne suivons actuellement aucun objet, saisissez l'entrée centroid et enregistrez chacun d'eux
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

# sinon, nous suivons actuellement des objets, 
# nous devons donc essayer de faire correspondre les centroid d'entrée aux centroid d'objet existants     
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

# afin de déterminer si nous devons mettre à jour, enregistrer ou désenregistrer un objet, 
# nous devons garder une trace de des index de lignes et de colonnes que nous avons déjà examinés
            usedRows = set()
            usedCols = set()
            # boucle sur la combinaison de l'indice (r, c)
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

# calculer les index de ligne et de colonne que nous n'avons pas encore examinés
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

# dans le cas où le nombre de centroid d'objet est égal ou supérieur au nombre de centroid d'entrée
# nous devons vérifier et voir si certains de ces objets ont potentiellement disparu
            if D.shape[0] >= D.shape[1]:
              for row in unusedRows:
                  objectID = objectIDs[row]
                  self.disappeared[objectID] += 1
                  if self.disappeared[objectID] > self.maxDisappeared:
                      self.deregister(objectID)

# sinon, si le nombre de centroid d'entrée est supérieur que le nombre de centroid d'objets existants
#  dont nous avons besoin enregistrer chaque nouveau centroïde d'entrée en tant qu'objet traçable
            else:
               for column in unusedCols:
                   self.register(inputCentroids[column])
        return self.objects    
 

prototxt_path ='deploy.prototxt'
model_path='res10_300x300_ssd_iter_140000.caffemodel'
confidence=0.6

ct = CentroidTracker()
(H, W) = (None, None)
print("Modèle de chargement...")
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
print("Démarrage du flux vidéo...")
#vs=FileVideoStream('vid').start()
vs=VideoStream(0).start()
time.sleep(1.0)



while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0 , (W, H), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    rects = []

# boucle sur les détections
    for i in range(0, detections.shape[2]):
        if detections[0, 0, i, 2] > confidence:
          box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
          rects.append(box.astype("int"))
          (startX, startY, endX, endY) = box.astype("int")
          cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

# mettez à jour notre tracker de centroid en utilisant l'ensemble calculé de rectangles de boîte englobante
    objects = ct.update(rects)
 
    for (objectID, centroid) in objects.items():
      text = "ID {}".format(objectID)
      cv2.putText(frame, text,(centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_SIMPLEX,0.4, (0, 0, 255), 1)
      cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
 
    cv2.imshow("Frame", frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break  
  
cv2.destroyAllWindows() 
vs.stop()
