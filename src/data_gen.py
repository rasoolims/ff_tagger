import os, sys

sens = open(os.path.abspath(sys.argv[1]), 'r').read().strip().split('\n\n')
writer = open(os.path.abspath(sys.argv[2]), 'w')

for sen in sens:
    lines = sen.strip().split('\n')
    words, tags = ['<s>', '<s>'], ['<s>', '<s>']
    for line in lines:
        word, tag = line.strip().split()
        words.append(word)
        tags.append(tag)
    words += ['</s>', '</s>']

    for i in range(len(lines)):
        feats = [words[i], words[i + 1], words[i + 2], words[i + 3], words[i + 4], tags[i], tags[i + 1]]
        label = tags[i + 2]
        writer.write('\t'.join(feats) + '\t' + label + '\n')
writer.close()
