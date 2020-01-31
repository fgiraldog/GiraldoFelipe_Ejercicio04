import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn.preprocessing 
from itertools import combinations

data = pd.read_csv('Cars93.csv')

y = np.array(data['Price'])
columns = ['MPG.city', 'MPG.highway', 'EngineSize', 'Horsepower', 'RPM', 'Rev.per.mile', 
          'Fuel.tank.capacity', 'Length', 'Width', 'Turn.circle', 'Weight']
x = np.array(data[columns])

scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_scaled, y, test_size=0.3)

score_matrix = np.array((11,1))

combinations_array = combinations([0,1,2,3,4,5,6,7,8,9,10],1)

for i in range(0,11):
	
	score_array = np.zeros((0))
	combinations_array = np.array((list(combinations(np.array([0,1,2,3,4,5,6,7,8,9,10]),i+2))))
	
	for element in combinations_array:
		print(element)
		x_comb = x_train[:,element]
		regression = sklearn.linear_model.LinearRegression()
		regression.fit(x_comb, y_train)

		score = regression.score(x_test[:,element],y_test)
		score_array = np.append(score_array, score)

	print(score_array)


score_array = []
_lambda = np.logspace(-3, 1, 10, endpoint=True)/100
for element in _lambda:
	
	lasso = sklearn.linear_model.Lasso(alpha = element)
	lasso.fit(x_train, y_train)
	score = lasso.score(x_test,y_test)

	score_array.append(score)


plt.figure()
plt.plot(_lambda,score_array)
plt.ylabel('$R^2$')
plt.xlabel(r'$\lambda$')
plt.savefig('lasso.png')