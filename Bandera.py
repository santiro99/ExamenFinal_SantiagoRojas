import cv2
import matplotlib as plt
import numpy as np
import collections
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from hough import hough

class Bandera:
    def __init__(self, path):                         #Consctructor de la clase, con parámetro de entrada: path de la imagen.
        self.path = path
        self.image = cv2.imread(self.path)            #Lee la imagen ubicada en el path y la almacena en una variable en self.

    def Colores(self, n_color):
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image_rgb = np.array(self.image_rgb, dtype=np.float64) / 255

        # Guarda en el objeto tamaño de la imagen y verifica que tenga 3 canales
        self.rows, self.cols, ch = self.image_rgb.shape
        assert ch == 3
        # Guarda la imagen convertida a un arreglo de 2D
        self.image_array = np.reshape(self.image_rgb, (self.rows * self.cols, ch))

        # Computar segmentación de color para valores n_color según el método guardado en el objeto
        self.n_color = n_color
        image_array_sample = shuffle(self.image_array, random_state=0)[:10000]

        model = KMeans(n_clusters=n_color, random_state=0).fit(image_array_sample)
        self.labels = model.predict(self.image_array)
        self.centers = model.cluster_centers_

        # Calcular suma de distancias intracluster para un valor de n_color
        intracluster = 0
        for label in range(self.centers.shape[0]):
            vector_label = self.image_array[self.labels == label]
            resta = vector_label - self.centers[label]
            magnitude = np.linalg.norm(resta, axis=1)
            distancia = np.sum(magnitude)
            intracluster = intracluster + distancia
        return intracluster

    def Porcentaje(self):
        self.percentaje = collections.Counter(self.labels)
        for i in self.percentaje.keys():
            self.percentaje[i] = self.percentaje[i] / (len(self.labels))
        return self.percentaje

    def Orientacion(self):
        high_thresh = 300
        bw_edges = cv2.Canny(self.image, high_thresh * 0.3, high_thresh, L2gradient=True)

        hough_transf = hough(bw_edges)
        accumulator = hough_transf.standard_HT()

        acc_thresh = 50
        N_peaks = 11
        nhood = [25, 9]
        peaks = hough_transf.find_peaks(accumulator, nhood, acc_thresh, N_peaks)

        [_, cols] = self.image.shape[:2]

        resultado = []
        for i in range(len(peaks)):
            rho = peaks[i][0]
            theta_ = hough_transf.theta[peaks[i][1]]

            theta_ = theta_ - 180

            if (np.abs(theta_) < 80) or (np.abs(theta_) > 100):
                resultado.append("Vertical")
            elif theta_ > 0:
                resultado.append("Horizontal")

        igual = all(elem == resultado[0] for elem in resultado)

        if igual:
            orientacion = resultado[0]
        else:
            orientacion = "Mixta"

        return orientacion

