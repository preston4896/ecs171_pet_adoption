Code explanation:
	If you want to create a new model, use training_model.py
	If you want to continue training an existing model, use refine_model.py. Be sure to save the model with a different name.


Current Models:
```
	Model0:
		With a learning rate of 3, the loss continues decreasing. So feel free to continue training this model if you want to.
		This model is learning slowly.
		1. Hidden_Layers = 2
		2. Nodes_per_layers = 10
		3. activation_func = sigmoid
		4. loss = mean_squared_error
		5. optimizer=sgd 
		6. training_loss: 0.1379 
		7. training_accuracy: 0.4289 
		8. testing_accuracy: 0.3715
```
```
	Model1:
		testing_accuracy stop growing while training loss keeps decreasing, this model probably overfits the data.
		1. Hidden_Layers = 2
		2. Nodes_per_layers = 2*n-1 = 37
		3. activation_func = sigmoid
		4. loss = mean_squared_error
		5. optimizer=adam
		6. training_loss: 0.1290
		7. training_accuracy: 0.4656
		8. testing_accuracy: 0.3554
```
