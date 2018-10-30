import tensorflow as tf

class TrainLogger:
    def __init__(self, filepath):
        self._filepath = filepath
        try:
            with open(filepath) as f: pass                
        except FileNotFoundError:
            with open(filepath, 'w') as f:
                f.write(','.join(['train_loss', 'validation_accuracy',
                          'n_train_tuples', 'n_validation_tuples', 'elapsed_seconds',
                          'batch_size', 'learning_rate', 'model_updated']) + '\n')
    def log_update(self, train_loss, validation_accuracy, n_train_tuples, n_validation_tuples,
                 elapsed_seconds, batch_size, learning_rate, model_updated):
            with open(self._filepath, 'a') as f:
                f.write('%.10f,%.5f,%d,%d,%.5f,%d,%f,%s\n' % (train_loss, validation_accuracy,
                        n_train_tuples, n_validation_tuples, elapsed_seconds, batch_size, learning_rate,
                        model_updated))

class ContentBasedLearn2RankNetwork:
    def __init__(self, user_model_mode='DEFAULT'):
        
        # --- placeholders
        self._pretrained_embeddings = tf.placeholder(shape=[None, 2048], dtype=tf.float32,
                                                     name='pretrained_embeddings')            
        self._profile_item_indexes = tf.placeholder(shape=[None,None], dtype=tf.int32,
                                                    name='profile_item_indexes')
        self._profile_sizes = tf.placeholder(shape=[None], dtype=tf.float32,
                                                   name='profile_sizes')        
        self._positive_item_index = tf.placeholder(shape=[None], dtype=tf.int32,
                                                   name='positive_item_index')
        self._negative_item_index = tf.placeholder(shape=[None], dtype=tf.int32,
                                                   name='negative_item_index')
        self._learning_rate = tf.placeholder(shape=[], dtype=tf.float32)

        # ---- user profile vector
        
        # profile item embeddings average
        tmp = tf.gather(self._pretrained_embeddings, self._profile_item_indexes)
        self._profile_item_embeddings = self.trainable_item_embedding(tmp)
        self._profile_masks = tf.expand_dims(tf.sequence_mask(self._profile_sizes, dtype=tf.float32), -1)
        self._masked_profile_item_embeddings = tf.multiply(self._profile_item_embeddings, self._profile_masks)        
        self._profile_items_average =\
            tf.reduce_sum(self._masked_profile_item_embeddings, axis=1) /\
            tf.reshape(self._profile_sizes, [-1, 1])
            
        if user_model_mode == 'BIGGER':
            # user hidden layer 1
            self._user_hidden_1 = tf.layers.dense(
                inputs=self._profile_items_average,
                units=256,
                activation=tf.nn.selu,
                name='user_hidden_1'
            )
            # user hidden layer 2
            self._user_hidden_2 = tf.layers.dense(
                inputs=self._user_hidden_1,
                units=256,
                activation=tf.nn.selu,
                name='user_hidden_2'
            )
            # user final vector
            self._user_vector = tf.layers.dense(
                inputs=self._user_hidden_2,
                units=128,
                activation=tf.nn.selu,
                name='user_vector'
            )
        elif user_model_mode == 'BIG':
            # user hidden layer
            self._user_hidden = tf.layers.dense(
                inputs=self._profile_items_average,
                units=256,
                activation=tf.nn.selu,
                name='user_hidden'
            )
            # user final vector
            self._user_vector = tf.layers.dense(
                inputs=self._user_hidden,
                units=128,
                activation=tf.nn.selu,
                name='user_vector'
            )
        elif user_model_mode == 'DEFAULT':
            # user hidden layer
            self._user_hidden = tf.layers.dense(
                inputs=self._profile_items_average,
                units=128,
                activation=tf.nn.selu,
                name='user_hidden'
            )
            # user final vector
            self._user_vector = tf.layers.dense(
                inputs=self._user_hidden,
                units=128,
                activation=tf.nn.selu,
                name='user_vector'
            )
        else: assert False
        
        # ---- positive item vector
        tmp = tf.gather(self._pretrained_embeddings, self._positive_item_index)
        self._positive_item_vector = self.trainable_item_embedding(tmp)
        
        # ---- negative item vector
        tmp = tf.gather(self._pretrained_embeddings, self._negative_item_index)
        self._negative_item_vector = self.trainable_item_embedding(tmp)
        
        # --- train loss
        dot_pos = tf.reduce_sum(tf.multiply(self._user_vector, self._positive_item_vector), 1)
        dot_neg = tf.reduce_sum(tf.multiply(self._user_vector, self._negative_item_vector), 1)
        dot_delta = dot_pos - dot_neg
        ones = tf.fill(tf.shape(self._user_vector)[:1], 1.0)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=dot_delta, labels=ones)
        loss = tf.reduce_mean(loss, name='train_loss')
        self._train_loss = loss
        
        # --- test accuracy
        accuracy = tf.reduce_sum(tf.cast(dot_delta > .0, tf.float32), name = 'test_accuracy')
        self._test_accuracy = accuracy
        
        # --- optimizer
        self._optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(self._train_loss)
        
    @staticmethod
    def trainable_item_embedding(X):
        with tf.variable_scope("trainable_item_embedding", reuse=tf.AUTO_REUSE):
            fc1 = tf.layers.dense( # None -> 256
                inputs=X,
                units=256,
                activation=tf.nn.selu,
                name='fc1'
            )
            fc2 = tf.layers.dense( # 256 -> 128
                inputs=fc1,
                units=128,
                activation=tf.nn.selu,
                name='fc2'
            )
            return fc2
    
    def optimize_and_get_train_loss(self, sess, learning_rate, pretrained_embeddings,
                                    profile_item_indexes, profile_sizes,
                                    positive_item_index, negative_item_index):
        return sess.run([
            self._optimizer,
            self._train_loss,
        ], feed_dict={
            self._learning_rate: learning_rate,
            self._pretrained_embeddings: pretrained_embeddings,
            self._profile_item_indexes: profile_item_indexes,
            self._profile_sizes: profile_sizes,
            self._positive_item_index: positive_item_index,
            self._negative_item_index: negative_item_index,
        })
    
    def get_train_loss(self, sess, pretrained_embeddings, profile_item_indexes, profile_sizes,
             positive_item_index, negative_item_index):
        return sess.run(
            self._train_loss, feed_dict={
            self._pretrained_embeddings: pretrained_embeddings,
            self._profile_item_indexes: profile_item_indexes,
            self._profile_sizes: profile_sizes,
            self._positive_item_index: positive_item_index,
            self._negative_item_index: negative_item_index,
        })
    
    def get_test_accuracy(self, sess, pretrained_embeddings, profile_item_indexes, profile_sizes,
             positive_item_index, negative_item_index):
        return sess.run(
            self._test_accuracy, feed_dict={
            self._pretrained_embeddings: pretrained_embeddings,
            self._profile_item_indexes: profile_item_indexes,
            self._profile_sizes: profile_sizes,
            self._positive_item_index: positive_item_index,
            self._negative_item_index: negative_item_index,
        })

class ContentBasedLearn2RankNetwork_Evaluation:
    def __init__(self, user_model_mode='DEFAULT'):
        
        # --- placeholders
        self._precomputed_embeddings = tf.placeholder(shape=[None, 128], dtype=tf.float32)
        self._profile_item_indexes = tf.placeholder(shape=[None], dtype=tf.int32)
        self._candidate_item_indexes = tf.placeholder(shape=[None], dtype=tf.int32)
            
        # ---- user profile vector
        
        # profile item embeddings average
        tmp = tf.gather(self._precomputed_embeddings, self._profile_item_indexes) 
        self._profile_items_average = tf.reshape(tf.reduce_mean(tmp, axis=0), (1, 128))
        
        
        if user_model_mode == 'BIGGER':
            # user hidden layer 1
            self._user_hidden_1 = tf.layers.dense(
                inputs=self._profile_items_average,
                units=256,
                activation=tf.nn.selu,
                name='user_hidden_1'
            )
            # user hidden layer 2
            self._user_hidden_2 = tf.layers.dense(
                inputs=self._user_hidden_1,
                units=256,
                activation=tf.nn.selu,
                name='user_hidden_2'
            )
            # user final vector
            self._user_vector = tf.layers.dense(
                inputs=self._user_hidden_2,
                units=128,
                activation=tf.nn.selu,
                name='user_vector'
            )
        elif user_model_mode == 'BIG':
            # user hidden layer
            self._user_hidden = tf.layers.dense(
                inputs=self._profile_items_average,
                units=256,
                activation=tf.nn.selu,
                name='user_hidden'
            )
            # user final vector
            self._user_vector = tf.layers.dense(
                inputs=self._user_hidden,
                units=128,
                activation=tf.nn.selu,
                name='user_vector'
            )
        elif user_model_mode == 'DEFAULT':
            # user hidden layer
            self._user_hidden = tf.layers.dense(
                inputs=self._profile_items_average,
                units=128,
                activation=tf.nn.selu,
                name='user_hidden'
            )
            # user final vector
            self._user_vector = tf.layers.dense(
                inputs=self._user_hidden,
                units=128,
                activation=tf.nn.selu,
                name='user_vector'
            )
        else: assert False
        
        # ---- candidate item vectors
        self._candidate_item_vectors = tf.gather(self._precomputed_embeddings, self._candidate_item_indexes)
        
        # ---- match score
        self._match_score = tf.reduce_sum(tf.multiply(self._user_vector, self._candidate_item_vectors), 1)
    
    def get_match_scores(self, sess, precomputed_embeddings, profile_item_indexes, candidate_items_indexes):
        return sess.run(
            self._match_score, feed_dict={
            self._precomputed_embeddings: precomputed_embeddings,
            self._profile_item_indexes: profile_item_indexes,
            self._candidate_item_indexes: candidate_items_indexes,
        })