#Este es el prototipo de neurona que he creado por ejemplo para que reciba el tiempo de latencia de
#un servidor y asi aprenda a determinar si esta tardando mucho o lo normal en la respuesta del api, ya sea del 
#lado de Zoho o del lado de Acumatica, ese es un dato inportante 

#SIN LIBRERIAS PARA OPDER ENTENDER MEJOR SU FUNCIONAMIENTO BÁSICO

#El peso que es la importancia que le estoy dando a esta variable:
w = 0.65

#El dato de entrada que es el tiempo que se esta tomando el api para responder:
latencyTime = 0.8

#El sesgo que es un valor que se le da a la neurona para ayudarla a aprender, es como un punto de referencia:
sesgo = 0.2

#Entonces crearé la funcion de procesamiento del dato de entrada:
#Esta es la función que hace Pytorch pero acá la ejecuto de manera sencilla, mas adelante usaré Pytorch:
def latencyTimeFunction(w, latencyTime, sesgo):
    #Multiplicar el peso por el dato de entrada y paso a seguir le sumo el sesgo, toodo de una vez incluido en la sola línea del return:
    return (latencyTime * w) + sesgo

latencyRiskLevel = latencyTimeFunction(w, latencyTime, sesgo)

#Ahora que muestre los valores por consola:
print(f"Tiempo de latencia entrada: {latencyTime}") 
print(f"Peso asignado: {w}")
print(f"Sesgo: {sesgo}")
print(f"Nivel de riesgo de latencia: {latencyRiskLevel}") 