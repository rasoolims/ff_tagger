from collections import defaultdict


class Vocab:
    def __init__(self, data_path):
        sentences = open(data_path, 'r').read().strip().split('\n\n')

        # first read into counts
        word_count, tags = defaultdict(int), set()
        for sentence in sentences:
            lines = sentence.strip().split('\n')
            for line in lines:
                word, tag = line.strip().split('\t')
                word_count[word] += 1
                tags.add(tag)
        tags = list(tags)
        words = [word for word in word_count.keys() if word_count[word] > 1]

        # Including unknown as the first features. <s> and </s> are for start and end of sentence.
        self.words = ['<UNK>', '<s>', '</s>'] + words
        self.word_dict = {word: i for i, word in enumerate(self.words)}

        self.output_tags = tags
        self.output_tag_dict = {tag: i for i, tag in enumerate(self.output_tags)}

        # <s> is for start and end of sentence. This is because we are only using the previous tag. In some cases
        # the previous tag can be the start of sentence.
        self.feat_tags = ['<s>'] + tags
        self.feat_tags_dict = {tag: i for i, tag in enumerate(self.feat_tags)}

    def tagid2tag_str(self, id):
        return self.output_tags[id]

    def tag2id(self, tag):
        return self.output_tag_dict[tag]

    def feat_tag2id(self, tag):
        return self.feat_tags_dict[tag]

    def word2id(self, word):
        return self.word_dict[word] if word in self.word_dict else self.word_dict['<UNK>']

    def num_words(self):
        return len(self.words)

    def num_tag_feats(self):
        return len(self.feat_tags)

    def num_tags(self):
        return len(self.output_tags)
