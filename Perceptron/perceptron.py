def predict(row, weights):
    activation = weights[0]
    for i in range(len(row)-2):
        activation += weights[i+1] * row[i+1]
    return 1.0 if activation > 0.0 else 0.0

def train_weights(train, l_rate, n_epoch):
    weights = [0.0 for i in range(len(train[0])-1)]
    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in train:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            sum_error += error**2
            weights[0] = weights[0] + l_rate * error
            for i in range(len(row)-2):
                weights[i+1] = weights[i+1] + l_rate * error * row[i+1]
            print('>pattern=', row, ', net=', prediction, ', weights=', weights)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return weights

"""
dataset [bias, w1, w2, w3, target]
"""
dataset = [[1,0,0,1,0],
	[1,1,1,1,1],
	[1,1,0,1,1],
	[1,0,1,1,0]]
l_rate = 1
n_epoch = 5
weights = train_weights(dataset, l_rate, n_epoch)
print(weights)
