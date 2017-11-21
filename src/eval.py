import os, sys

gold_file = os.path.abspath(sys.argv[1])
predicted_file = os.path.abspath(sys.argv[2])

correct, all_tag = 0, 0
for g, p in zip(open(gold_file, 'r'), open(predicted_file, 'r')):
    g_s, p_s = g.strip().split('\t'), p.strip().split('\t')
    if len(g_s) > 1:
        g_t, p_t = g_s[1], p_s[1]
        if g_t == p_t:
            correct += 1
        all_tag += 1

print 'accuracy', round(100 * float(correct) / all_tag)
