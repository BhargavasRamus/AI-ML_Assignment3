import numpy as np

print("provide number of hidden layers:")
H=int(input())
print("number of training samples:")
n=int(input())
print("provide the standard deviation for noise:")
sigma=float(input())
print("enter the number of times training samples be trained:")
Range=int(input())
print("learning rate") 
lr=float(input())

#functions
def Sigma(x):
	return (1/(1+np.exp(-x)))

def Differ_Sigma(x):
	return (Sigma(x)-Sigma(x)**2)

def equation(x,weight,bias):
	return (np.matmul(weight.T,x.reshape(len(x),1))+bias)
	
#for AND
In=np.array([[0,0],[0,1],[1,0],[1,1]])
Out=np.array([[0],[0],[0],[1]])
X=[]
Y=[]
#training samples
for i in range (len(In)):
	for j in range (n):
		X.append(In[i]+np.random.normal(0,sigma,(1,2)))
		Y.append(Out[i]+np.random.normal(0,sigma))
		
X=np.array(X)
X=X.reshape((len(Y),2))
Y=np.array(Y)


#initial condition for weights
W1=np.random.normal(0,1,(2,H))
B1=np.random.normal(0,1,(H,1))
W2=np.random.normal(0,1,(H,1))
B2=np.random.normal(0,1,(1,1))

for i in range (Range):
	#initializing weights
	w1=np.zeros(W1.shape)
	w2=np.zeros(W2.shape)
	b1=np.zeros(B1.shape)
	b2=np.zeros(B2.shape)
	
	for j in range (len(Y)):
		Output_1=equation(X[j],W1,B1)
		t=Sigma(Output_1)
		Output_2=equation(t,W2,B2)
		y=Sigma(Output_2)
		#back_propagation
		b2+=2*(y-Y[j])*Differ_Sigma(Output_2)
		w2+=2*(y-Y[j])*Differ_Sigma(Output_2)*t
		Sq_error=(y-Y[j])**2
		for k in range (H):
			b1[k]+=(2*(y-Y[j])*Differ_Sigma(Output_2)*Differ_Sigma(Output_1[k])*W2[k]).reshape(1,)
			W1[:,k]+=(2*(y-Y[j])*Differ_Sigma(Output_2)*Differ_Sigma(Output_1[k])*W2[k]*X[j]).reshape(2,)
	print(Sq_error)
	W1 -= lr*w1
	W2 -= lr*w2
	B1 -= lr*b1
	B2 -= lr*b2

while(1):
	a=[]
	for i in range (0,2):
		print("enter the testing sample")
		a.append(float(input()))
	a=np.array(a)
	print(a)
	Output_1=equation(a,W1,B1)
	t=Sigma(Output_1)
	Output_2=equation(t,W2,B2)
	Y_test=Sigma(Output_2)-0.5
	if(Y_test>=0.0):
		Y_test=1
	else:
		Y_test=0
	print("AND of testing sample:",Y_test)
