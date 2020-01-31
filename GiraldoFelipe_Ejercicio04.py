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


print(np.transpose((list(combinations_array))))

# for i in range(0,11):
	
# 	score_array = np.zeros((0))
# 	combinations_array = np.transpose(np.array(list(combinations([0,1,2,3,4,5,6,7,8,9,10],i+1))))
	
# 	for j in range(1,len(combinations_array)):
# 		x_comb = x_train[:,np.array(list(combinations_array))[j]]
# 		regression = sklearn.linear_model.LinearRegression()
# 		regression.fit(x_comb, y_train)

# 		score = regression.score(x_test,y_test)
# 		score_array = np.append(score_array, score)

# 	print(score_array)


score_array = []
_lambda = np.logspace(0.1, 3, 10, endpoint=True)/100
for element in _lambda:
	
	lasso = sklearn.linear_model.Lasso(alpha = element)
	lasso.fit(x_train, y_train)
	score = lasso.score(x_test,y_test)

	score_array.append(score)


plt.figure()
plt.plot(_lambda,score_array)
plt.savefig('lasso.png')