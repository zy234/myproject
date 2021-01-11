
DATA_DIR = './data/'

w_f = open(DATA_DIR + 'new_train.txt', 'w')
with open(DATA_DIR + 'new_SEtrain_set.csv', 'r') as f:
    while True:
        sentence = f.readline()
        label = f.readline()
        if not label:
            break
        sentence = sentence.split()
        label = label.split()
        for ix in range(len(sentence)):
            w_f.write(sentence[ix] + ' ' + label[ix])
            w_f.write('\n')

        w_f.write('\n')

w_f.close()

