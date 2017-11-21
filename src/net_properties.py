class NetProperties:
    def __init__(self, word_embed_dim, pos_embed_dim, hidden_dim, minibatch_size):
        self.word_embed_dim = word_embed_dim
        self.pos_embed_dim = pos_embed_dim
        self.hidden_dim = hidden_dim
        self.minibatch_size = minibatch_size