import torch 
import torch.nn as nn

#Hardcodeo simulando el peso de los json que estoy cargando:
#Se dividen entre 1000 para normalizar los datos y que la neurona pueda aprender correctamente
QuantityEntryArray = torch.tensor([[100.0/1000],[150.0/1000],[500.0/1000],[600.0/1000]])

#Aca le asigno el peso que considero que debe tener
#de acuerdo a cada uno de los pesos en megabytes de los json
Weight = torch.tensor([[0.0],[0.0],[1.0],[1.0]])

#Este segmento es solo para verificar por consola que todo esta corriendo correctamente
print("Peso de los archivos json cargados:")
print(f"Peso de los archivos: {QuantityEntryArray}")
print(f"Resultados esperados: {Weight}")

#Acá defino que la neurona recibe un número o sea el tamaño del Json y 
#sale otro número que es el resultado de la multuplicación básica
model = nn.Linear(1, 1)

#Esta funcion sigmoide se encargará de transformar el resultado de la multiplicacion
#en un valor entre 0 y 1
activation = nn.Sigmoid()

#Acá solo pretendo imprimir por consola para verificar que todo vaya bien
print("Estructura creada:")
print(f"Modelo: {model}")

with torch.no_grad():
    initialPrediction = activation(model(QuantityEntryArray))
    print("Respuesta inicial de la IA aún sin enrtenamiento:")
    #Aca muestra lo que la iA "cree" que son esos archivos
    for i in range(len(QuantityEntryArray)):
        print(f"Entrada: {QuantityEntryArray[i].item()*1000} - Predicción: {initialPrediction[i].item():.4f}")

#aca comienza la medicion del error comparando la prediccion 0 a 1 con el resultado real 0 o 1
criteria = nn.BCELoss()
#Se ajusta el lr a 0.1 para una convergencia efectiva
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
#Verifico si todo va bien
print ("Auditoria y optimizador configurados")

#De aca en adelante comienza la fase de entrenamiento
for epoch in range(1000):
    #Se pondrá el optimizador en cero para no traer calculos viejos
    optimizer.zero_grad()
    
    #Arranca la prediccion
    predictions = activation(model(QuantityEntryArray))
    
    #Calculo que tanto error cometió
    error = criteria(predictions, Weight)
    
    #Retropropaga el error
    error.backward()
    
    #Optimizer optimiza los pesos
    optimizer.step()

#Compruebo si corrió bien el código hasta este punto
print("El entrenamiento ha sido exitoso")

print("Resultados finales después del entrenamiento:")
with torch.no_grad():
    finalPrediction = activation(model(QuantityEntryArray))
    for i in range(len(QuantityEntryArray)):
        estado = "ÉXITO (Acumatica)" if finalPrediction[i].item() > 0.5 else "ERROR (Zoho)"
        probabilidad = finalPrediction[i].item() * 100
    
        #Multiplicamos por 1000 en el print para que veas tus valores originales (100, 150, etc)
        print(f"Archivo: {QuantityEntryArray[i].item()*1000:.1f} bytes -> Probabilidad de éxito: {probabilidad:.2f}% -> Clasificación: {estado}")