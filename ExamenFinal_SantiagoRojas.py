from Bandera import *

if __name__ == '__main__':
    image_path = input("Ingrese el path de la imagen con la que desea trabajar ")  # Se pide al usuario ingresar path de imagen a trabajar.
    imagen = Bandera(image_path)    # Crear objeto con parámetros indicados

    num_clusters = 4
    distancias = np.zeros((num_clusters, 1))
    for n_color in range(1, num_clusters + 1):
        distancias[n_color - 1] = imagen.Colores(n_color)  # Para n_color calcular clustering y calcula distancias intracluster para n_color clusters

    num_colores  = int(distancias.argmin()) + 1 # Posición del menor error de distancia intracluster
    print("La bandera tiene", num_colores, " colores.")  # Muestra numero de colores.

    porcentajes = imagen.Porcentaje()
    print("Los porcentajes de los colores en la bandera son: ", porcentajes)  # Muestra porcentajes de colores

    orientacion = imagen.Orientacion()
    print("La orientación de las lineas en la bandera es", orientacion)  # Muestra orientacion de la bandera


