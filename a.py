import numpy as np
import cvxpy as cp
import pandas as pd

X = pd.read_csv('Xsvm.csv',header=None)
Y = pd.read_csv('ysvm.csv',header=None)
X = np.array(X,dtype=np.float64)
Y = np.array(Y,dtype=np.float64)
print(X.shape,Y.shape)

# Convex Optimization
alpha = cp.Variable(len(Y))
R = cp.matmul(cp.diag(alpha),Y)
R = cp.matmul(X.T,R)
R = cp.norm(R)**2
R.shape

for i in range len(Y):
    intermediate_matrix[i,:] = y[i] * x[i,:]

#putting constraints 
Constant1 = cp.matmul(alpha.T,Y)
constraint1=alpha<=0
constraint2=constant1==0
Constraint= [constraint1,constraint2]
objective = cp.Maximize(cp.sum(alpha) - 0.5*R)
prob = cp.Problem(objective, Constraint)
prob.solve(verbose=True)

# get w
W = np.dot(intermediate_matrix.T,alpha)

print(W)
#for bias parameter
W0 = 0
lambda = 1e-3
for i in range len(Y):
	if alpha[i] > lambda:
		W0 += Y[i]-np.dot(W,X[i]) 
print(W0)

#testing data
Test = np.array([[2,0.5],[0.8,0.7],[1.58,1.33],[0.008, 0.001]])
for i in range(len(Test)):
    prediction = np.sign(np.dot(W,Test[i])+W0)
    if prediction > 0:
		  prediction = 1
	  else:
		  prediction = -1
    print(Test[i],est)

