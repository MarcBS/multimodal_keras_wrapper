splits = ['train_split.txt', 'val_split.txt', 'test.txt']
result_labels = ['train_labels.txt', 'val_labels.txt', 'test_labels.txt']

list_labels = 'classes.txt'

# Read list of labels
classes = dict()
with open(list_labels, 'r') as f:
    for i, line in enumerate(f):
        line = line.strip('\n')
        classes[line] = i

# Read data split and assign numerical label to each image
for s, r in zip(splits, result_labels):
    s = open(s, 'r')
    r = open(r, 'w')
    for line in s:
        line = line.strip('\n').split('/')
        r.write(str(classes[line[0]]) + '\n')
    s.close()
    r.close()

print 'Done!'
