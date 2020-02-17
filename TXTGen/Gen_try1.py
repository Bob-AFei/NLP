# Read the txt

import numpy as np
import torch
import matplotlib.pyplot as plt

PATH = './data/test.txt'
file = open(PATH)
data = file.read()

i2c = dict(enumerate(sorted(set(''.join(data)))))
c2i = {char:i for i,char in i2c.items()}

STEP = 10
DICT = len(i2c)

def rearange_data (test):
	ans = []
	for i in range(len(test)-STEP):
		ans.append(test[i:i+STEP])
	return ans

def hot_point(test):
	ans = []
	for sentence in test:
		ans.append([c2i[i] for i in sentence])
	return ans

def change_X (X):
	ans = np.zeros((len(X),1,STEP,DICT))
	for i in range(len(X)):
		for j in range(STEP):
			ans[i,0,j,X[i][j]] = 1
	return ans 

def get_Xy (data):
	ans = rearange_data(data)
	ans = hot_point(ans)
	x = ans[:-1]
	X = change_X(ans[:-1])
	y = ans[1:]
	return x,torch.tensor(X),torch.tensor(y)

x,X,y = get_Xy(data)

# Net work define
import torch.nn as nn
import torch.nn.functional as F

class Net (nn.Module):
	def __init__ (self, input_size, hidden_size, output_size, layers):
		super().__init__()
		self.hidden_size = hidden_size
		self.lstm = nn.LSTM(input_size, hidden_size, layers, batch_first = True, bidirectional = True)
		self.fc = nn.Linear(hidden_size*2, output_size)
		self.function = nn.Softmax(1)

	def forward(self, X):
		X,_x = self.lstm(X)
		X = X.contiguous().view(-1, self.hidden_size*2)
		X = self.fc(X)
		# print(X.shape)
		# X = self.function(X)
		return X

net = Net(input_size=DICT, hidden_size = 100, output_size=DICT, layers=3)

import torch.optim as op
# criteria = nn.NLLLoss()
criteria = nn.CrossEntropyLoss()
optimiser = op.Adam(net.parameters())
EPCHO = 10

# train
def train ():
	for epcho in range(EPCHO):
		for i in range(len(X)):
			optimiser.zero_grad()
			predict = net(X[i].float())
			_y = y[i].long().view(-1)
			loss = criteria(predict, _y)
			loss.backward()
			optimiser.step()
		print('Epcho {:3} ----- Loss {}'.format(epcho,loss))
		# print(predict,_y)
	# torch.save(net,'net.plk')

def test_by_vector(n=0):
	net = torch.load('net.plk')
	predict = net(X[n].float())
	# print(predict, y[n])
	predict = torch.argmax(predict,dim=1)

	inp = [i2c[int(c)] for c in x[n]]
	predict = [i2c[int(c)] for c in predict]
	actual = [i2c[int(c)] for c in y[n]]

	print('input: {}\npredi: {}\nactua: {}'.format(''.join(inp),''.join(predict),''.join(actual)))

	# print(torch.argmax(predict,dim=1),y[0])

def test_by_string(string):
	net = torch.load('net.plk')
	while len(string) < STEP:
		string = ' '+string
	ans = hot_point([string])
	ans = torch.tensor(change_X(ans))
	predict = net(ans[0].float())
	predict = torch.argmax(predict,dim=1)
	predict = [i2c[int(c)] for c in predict]
	return ''.join(predict)

def test (string):
	ans = string
	print('input:  '+string)
	for i in range(20):
		ans = test_by_string(ans)
		string+=ans[-1]
	print('output: '+string)

# train()	
test('this ')








