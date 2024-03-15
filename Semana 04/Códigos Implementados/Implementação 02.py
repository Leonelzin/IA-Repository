#Aluno: Douglas Leonel de Almeida
#Matrícula: 2110213

#Implemente um algoritmo de aprendizado de máquina utilizando LinearSVC para reconhecer que dada uma entrada, ela é um ser humano ou um espríto.

#Algoritmo que verifica se é humano ou espírito

#caracteristicas
# fala?
# anda?
# atravessa parede
# tem corpo físico

humano1 = [1, 0, 0, 1]
humano2 = [1, 1, 0, 1]
humano3 = [1, 1, 0, 1]
humano4 = [0, 1, 0, 1]

espirito1 = [0, 1, 1, 0]
espirito2 = [1, 0, 1, 0]
espirito3 = [1, 1, 1, 1]
espirito4 = [0, 0, 0, 0]

treino_x = [humano1, humano2, humano3, humano4, espirito1, espirito2, espirito3, espirito4]

#1 - humano
#0 - espírito

treino_y = [1, 1, 1, 1, 0, 0, 0, 0]

from sklearn.svm import LinearSVC

modelo2 = LinearSVC()
modelo2.fit(treino_x, treino_y)

ser_misterioso = [1, 1, 0, 1]
result = modelo2.predict([ser_misterioso])

if result == 1:
  print ("Este ser é um humano")
else:
  print ("Este ser é um espírito")
