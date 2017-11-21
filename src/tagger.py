from optparse import OptionParser
from network import *
import pickle
from net_properties import *
from utils import *

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="train_file", metavar="FILE", default=None)
    parser.add_option("--train_data", dest="train_data_file", metavar="FILE", default=None)
    parser.add_option("--test", dest="test_file", metavar="FILE", default=None)
    parser.add_option("--output", dest="output_file", metavar="FILE", default=None)
    parser.add_option("--model", dest="model_path", metavar="FILE", default=None)
    parser.add_option("--vocab", dest="vocab_path", metavar="FILE", default=None)
    parser.add_option("--we", type="int", dest="we", default=100)
    parser.add_option("--pe", type="int", dest="pe", default=50)
    parser.add_option("--hidden", type="int", dest="hidden", default=200)
    parser.add_option("--minibatch", type="int", dest="minibatch", default=1000)
    parser.add_option("--epochs", type="int", dest="epochs", default=5)

    (options, args) = parser.parse_args()

    if options.train_file and options.train_data_file and options.model_path and options.vocab_path:
        net_properties = NetProperties(options.we, options.pe, options.hidden, options.minibatch)

        # creating vocabulary file
        vocab = Vocab(options.train_file)

        # writing properties and vocabulary file into pickle
        pickle.dump((vocab, net_properties), open(options.vocab_path, 'w'))

        # constructing network
        network = Network(vocab, net_properties)

        # training
        network.train(options.train_data_file,options.epochs)

        # saving network
        network.save(options.model_path)

    if options.test_file and options.model_path and options.vocab_path and options.output_file:
        # loading vocab and net properties
        vocab, net_properties = pickle.load(open(options.vocab_path, 'r'))

        # constructing default network
        network = Network(vocab, net_properties)

        # loading network trained model
        network.load(options.model_path)

        writer = open(options.output_file, 'w')
        for sentence in open(options.test_file, 'r'):
            words = sentence.strip().split()
            tags = network.decode(words)
            output = [word + '\t' + tag for word, tag in zip(words, tags)]
            writer.write('\n'.join(output) + '\n\n')
        writer.close()
