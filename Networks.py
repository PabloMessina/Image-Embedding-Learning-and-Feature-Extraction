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

class ContentBasedLearn2RankNetwork_Base:
    
    @staticmethod
    def compute_user_vector__from_avgpool(profile_items_avgpool, user_model_mode):
        if user_model_mode == 'BIGGER':
            # user hidden layer 1
            user_hidden_1 = tf.layers.dense(
                inputs=profile_items_avgpool,
                units=256,
                activation=tf.nn.selu,
                name='user_hidden_1'
            )
            # user hidden layer 2
            user_hidden_2 = tf.layers.dense(
                inputs=user_hidden_1,
                units=256,
                activation=tf.nn.selu,
                name='user_hidden_2'
            )
            # user final vector
            return tf.layers.dense(
                inputs=user_hidden_2,
                units=128,
                activation=tf.nn.selu,
                name='user_vector'
            )
        if user_model_mode == 'BIG':
            # user hidden layer
            user_hidden = tf.layers.dense(
                inputs=profile_items_avgpool,
                units=256,
                activation=tf.nn.selu,
                name='user_hidden'
            )
            # user final vector
            return tf.layers.dense(
                inputs=user_hidden,
                units=128,
                activation=tf.nn.selu,
                name='user_vector'
            )
        if user_model_mode == 'DEFAULT':
            # user hidden layer
            user_hidden = tf.layers.dense(
                inputs=profile_items_avgpool,
                units=128,
                activation=tf.nn.selu,
                name='user_hidden'
            )
            # user final vector
            user_vector = tf.layers.dense(
                inputs=user_hidden,
                units=128,
                activation=tf.nn.selu,
                name='user_vector'
            )
        assert False
    
    @staticmethod
    def compute_user_vector__from_avgpool_maxpool(profile_items_avgpool, profile_items_maxpool):
        
        # concatenate avgpool + maxpool
        profile_vector = tf.concat([profile_items_avgpool, profile_items_maxpool], 1)
        
        # user hidden layer 1
        user_hidden_1 = tf.layers.dense(
            inputs=profile_vector,
            units=256,
            activation=tf.nn.selu,
            name='user_hidden_1'
        )

        # user final vector
        return tf.layers.dense(
            inputs=user_hidden_1,
            units=128,
            activation=tf.nn.selu,
            name='user_vector'
        )
    
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

class ContentBasedLearn2RankNetwork_Train(ContentBasedLearn2RankNetwork_Base):
    def __init__(self, profile_pooling_mode='AVG', *user_vector_args):
        
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
        
        # profile item embeddings
        tmp = tf.gather(self._pretrained_embeddings, self._profile_item_indexes)
        self._profile_item_embeddings = self.trainable_item_embedding(tmp)
        
        # avgpool masking
        self._profile_masks__avgpool = tf.expand_dims(tf.sequence_mask(self._profile_sizes, dtype=tf.float32), -1)        
        self._masked_profile_item_embeddings__avgpool =\
            self._profile_item_embeddings * self._profile_masks__avgpool
        
        # maxpool masking
        self._profile_masks__maxpool = (1. - self._profile_masks__avgpool) * -9999.
        self._masked_profile_item_embeddings__maxpool =\
            self._masked_profile_item_embeddings__avgpool + self._profile_masks__maxpool
        
        # items avgpool
        self._profile_items_avgpool =\
            tf.reduce_sum(self._masked_profile_item_embeddings__avgpool, axis=1) /\
            tf.reshape(self._profile_sizes, [-1, 1])
        
        # items maxpool
        self._profile_items_maxpool =\
            tf.reduce_max(self._masked_profile_item_embeddings__maxpool, axis=1)
        
        # user vector
        if profile_pooling_mode == 'AVG':            
            self._user_vector = self.compute_user_vector__from_avgpool(
                self._profile_items_avgpool, *user_vector_args)
        else:
            assert profile_pooling_mode == 'AVG+MAX'
            self._user_vector = self.compute_user_vector__from_avgpool_maxpool(
                self._profile_items_avgpool,
                self._profile_items_maxpool)
        
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

class ContentBasedLearn2RankNetwork_Precomputation(ContentBasedLearn2RankNetwork_Base):
    def __init__(self):
        
        # --- placeholders
        self._pretrained_resnet50_vectors = tf.placeholder(shape=[None, 2048], dtype=tf.float32)
        
        # --- item vectors
        self._item_vectors = self.trainable_item_embedding(self._pretrained_resnet50_vectors)
        
    def precompute_tensors(self, sess, pretrained_resnet50_vectors):
        return sess.run(self._item_vectors, feed_dict={
            self._pretrained_resnet50_vectors: pretrained_resnet50_vectors,
        })

class ContentBasedLearn2RankNetwork_Evaluation(ContentBasedLearn2RankNetwork_Base):
    def __init__(self, profile_pooling_mode='AVG', *user_vector_args):
        
        # --- placeholders
        self._precomputed_item_vectors = tf.placeholder(shape=[None, 128], dtype=tf.float32)
        self._profile_item_indexes = tf.placeholder(shape=[None], dtype=tf.int32)
        self._candidate_item_indexes = tf.placeholder(shape=[None], dtype=tf.int32)
            
        # ---- user profile vector
        
        tmp = tf.gather(self._precomputed_item_vectors, self._profile_item_indexes) 
        
        # profile items avgpool
        self._profile_items_avgpool = tf.reshape(tf.reduce_mean(tmp, axis=0), (1, 128))
        
        # profile items maxpool
        self._profile_items_maxpool = tf.reshape(tf.reduce_max(tmp, axis=0), (1, 128))
        
        # user vector
        if profile_pooling_mode == 'AVG':            
            self._user_vector = self.compute_user_vector__from_avgpool(
                self._profile_items_avgpool, *user_vector_args)
        else:
            assert profile_pooling_mode == 'AVG+MAX'
            self._user_vector = self.compute_user_vector__from_avgpool_maxpool(
                self._profile_items_avgpool,
                self._profile_items_maxpool)
        
        # ---- candidate item vectors
        self._candidate_item_vectors = tf.gather(self._precomputed_item_vectors,
                                                 self._candidate_item_indexes)
        
        # ---- match scores
        self._match_scores = tf.reduce_sum(self._user_vector * self._candidate_item_vectors, 1)
    
    def get_match_scores(self, sess, precomputed_item_vectors, profile_item_indexes, candidate_items_indexes):
        return sess.run(
            self._match_scores, feed_dict={
            self._precomputed_item_vectors: precomputed_item_vectors,
            self._profile_item_indexes: profile_item_indexes,
            self._candidate_item_indexes: candidate_items_indexes,
        })
    
class VBPR_Network:
    def __init__(self, n_users, n_items, user_latent_dim, item_latent_dim, item_visual_dim, pretrained_dim=2048):
        
        assert (user_latent_dim == item_latent_dim + item_visual_dim)
        
        self._item_visual_dim = item_visual_dim
        
        # --- placeholders
        self._pretrained_image_embeddings = tf.placeholder(shape=[None, pretrained_dim], dtype=tf.float32,
                                                     name='pretrained_image_embeddings')    
        self._user_index = tf.placeholder(shape=[None], dtype=tf.int32,
                                          name='user_index')
        self._positive_item_index = tf.placeholder(shape=[None], dtype=tf.int32,
                                                   name='positive_item_index')
        self._negative_item_index = tf.placeholder(shape=[None], dtype=tf.int32,
                                                   name='negative_item_index')
        self._learning_rate = tf.placeholder(shape=[], dtype=tf.float32)
            
        # ------------------------------------
        # ---- Global trainable variables
        
        # -- user latent factor matrix
        # (n_users x user_latent_dim)
        self._user_latent_factors = tf.Variable(
            tf.random_uniform([n_users, user_latent_dim], -1.0, 1.0),
            name='user_latent_factors'
        )
        
        # -- item latent factor matrix
        # (n_items x item_latent_dim)
        self._item_latent_factors = tf.Variable(
            tf.random_uniform([n_items, item_latent_dim], -1.0, 1.0),
            name='item_latent_factors'
        )
        
        # -- item latent biases
        self._item_latent_biases = tf.Variable(
            tf.random_uniform([n_items], -1.0, 1.0),
            name='item_latent_biases'
        )
        
        # -- global visual bias
        self._visual_bias = tf.Variable(
            tf.random_uniform([pretrained_dim], -1.0, 1.0),
            name='visual_bias'
        )
        
        # -------------------------------
        # ---- minibatch tensors
        
        # -- user
        self._user_latent_vector = tf.gather(self._user_latent_factors, self._user_index)
        
        # -- positive item
        self._pos_vector,\
        self._pos_latent_bias,\
        self._pos_visual_bias = self.get_item_variables(self._positive_item_index)
        self._pos_score = tf.reduce_sum(self._user_latent_vector * self._pos_vector, 1) +\
                    self._pos_latent_bias +\
                    self._pos_visual_bias
        
        # -- negative item
        self._neg_vector,\
        self._neg_latent_bias,\
        self._neg_visual_bias = self.get_item_variables(self._negative_item_index)
        self._neg_score = tf.reduce_sum(self._user_latent_vector * self._neg_vector, 1) +\
                    self._neg_latent_bias +\
                    self._neg_visual_bias
        
        # -------------------------------
        # ---- train-test tensors
        
        # -- train loss
        delta_score = self._pos_score - self._neg_score
        ones = tf.fill(tf.shape(self._user_latent_vector)[:1], 1.0)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=delta_score, labels=ones)
        loss = tf.reduce_mean(loss, name='train_loss')
        self._train_loss = loss
        
        # -- test accuracy
        accuracy = tf.reduce_sum(tf.cast(delta_score > .0, tf.float32), name='test_accuracy')
        self._test_accuracy = accuracy
        
        # -- optimizer
        self._optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(self._train_loss)
        
    def get_item_variables(self, item_index):
        pre_vector = tf.gather(self._pretrained_image_embeddings, item_index)
        # 1) item vector
        #    1.1) visual vector
        visual_vector = self.trainable_image_embedding(pre_vector, self._item_visual_dim)
        #    1.2) latent vector
        latent_vector = tf.gather(self._item_latent_factors, item_index)
        #    1.3) concatenation
        final_vector = tf.concat([visual_vector, latent_vector], 1)
        # 2) latent bias
        latent_bias = tf.gather(self._item_latent_biases, item_index)
        # 3) visual bias
        visual_bias = tf.reduce_sum(pre_vector * self._visual_bias, 1)
        # return
        return final_vector, latent_bias, visual_bias
        
    @staticmethod
    def trainable_image_embedding(X, output_dim):
        with tf.variable_scope("trainable_image_embedding", reuse=tf.AUTO_REUSE):
            fc1 = tf.layers.dense( # None -> output_dim
                inputs=X,
                units=output_dim,
                name='fc1'
            )
            return fc1
    
    def optimize_and_get_train_loss(self, sess, pretrained_image_embeddings, user_index, positive_item_index,
                                    negative_item_index, learning_rate):
        return sess.run([
            self._optimizer,
            self._train_loss,
        ], feed_dict={
            self._pretrained_image_embeddings: pretrained_image_embeddings,
            self._user_index: user_index,
            self._positive_item_index: positive_item_index,
            self._negative_item_index: negative_item_index,
            self._learning_rate: learning_rate,
        })
    
    def get_train_loss(self, sess, pretrained_image_embeddings, user_index, positive_item_index, negative_item_index):
        return sess.run( 
            self._train_loss, feed_dict={
            self._pretrained_image_embeddings: pretrained_image_embeddings,
            self._user_index: user_index,
            self._positive_item_index: positive_item_index,
            self._negative_item_index: negative_item_index,
        })
    
    def get_test_accuracy(self, sess, pretrained_image_embeddings, user_index, positive_item_index, negative_item_index):
        return sess.run(
            self._test_accuracy, feed_dict={
            self._pretrained_image_embeddings: pretrained_image_embeddings,
            self._user_index: user_index,
            self._positive_item_index: positive_item_index,
            self._negative_item_index: negative_item_index,
        })
    
class VBPR_Network_Evaluation:
    def __init__(self, n_users, n_items, user_latent_dim, item_latent_dim, item_visual_dim,
                 pretrained_dim=2048):
        
        # --- placeholders
        self._pretrained_image_embeddings = tf.placeholder(shape=[None, pretrained_dim], dtype=tf.float32)
        self._item_index = tf.placeholder(shape=[None], dtype=tf.int32)
            
        # ------------------------------------
        # ---- Global trainable variables
        
        # -- user latent factor matrix
        # (n_users x user_latent_dim)
        self._user_latent_factors = tf.Variable(
            tf.random_uniform([n_users, user_latent_dim], -1.0, 1.0),
            name='user_latent_factors'
        )
        
        # -- item latent factor matrix
        # (n_items x item_latent_dim)
        self._item_latent_factors = tf.Variable(
            tf.random_uniform([n_items, item_latent_dim], -1.0, 1.0),
            name='item_latent_factors'
        )
        
        # -- item latent biases
        self._item_latent_biases = tf.Variable(
            tf.random_uniform([n_items], -1.0, 1.0),
            name='item_latent_biases'
        )
        
        # -- global visual bias
        self._visual_bias = tf.Variable(
            tf.random_uniform([pretrained_dim], -1.0, 1.0),
            name='visual_bias'
        )
        
        # -------------------------------
        # ---- minibatch tensors
        
        item_pre_vector = tf.gather(self._pretrained_image_embeddings, self._item_index)
        
        # 1) item vector
        #    1.1) visual vector
        item_visual_vector = self.trainable_image_embedding(item_pre_vector, item_visual_dim)
        #    1.2) latent vector
        item_latent_vector = tf.gather(self._item_latent_factors, self._item_index)
        #    1.3) concatenation
        self._item_final_vector = tf.concat([item_visual_vector, item_latent_vector], 1)
        
        # 2) item bias
        #    1.1) visual bias
        item_visual_bias = tf.reduce_sum(item_pre_vector * self._visual_bias, 1)
        #    1.2) latent bias
        item_latent_bias = tf.gather(self._item_latent_biases, self._item_index)
        #    1.3) final bias
        self._item_final_bias = item_visual_bias + item_latent_bias
        
    @staticmethod
    def trainable_image_embedding(X, output_dim):
        with tf.variable_scope("trainable_image_embedding", reuse=tf.AUTO_REUSE):
            fc1 = tf.layers.dense( # None -> output_dim
                inputs=X,
                units=output_dim,
                name='fc1'
            )
            return fc1
    
    def get_item_final_vector_bias(self, sess, pretrained_image_embeddings, item_index):
        return sess.run([
            self._item_final_vector,
            self._item_final_bias,
        ], feed_dict={
            self._pretrained_image_embeddings: pretrained_image_embeddings,
            self._item_index: item_index,
        })
    
    def get_user_latent_vectors(self, sess):
        return sess.run(self._user_latent_factors)
    

def CNN_F(input_tensor, output_dim):
    with tf.variable_scope("CNN_F", reuse=tf.AUTO_REUSE):
        
        # ---- conv layer 1
        conv1 = tf.layers.conv2d(
            inputs=input_tensor,
            filters=64,
            kernel_size=[11,11],
            strides=4,
            padding='same',
            activation=tf.nn.selu,
            name='conv1')
        pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[2,2],
            strides=2,
            name='pool1')
        
        # ---- conv layer 2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=256,
            kernel_size=[5,5],
            strides=1,
            padding='same',
            activation=tf.nn.selu,
            name='conv2')
        pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=[2,2],
            strides=2,
            name='pool2')
        
        # ---- conv layer 3
        conv3 = tf.layers.conv2d(
            inputs=pool2,
            filters=256,
            kernel_size=[3,3],
            strides=1,
            padding='same',
            activation=tf.nn.selu,
            name='conv3')
        
        # ---- conv layer 4
        conv4 = tf.layers.conv2d(
            inputs=conv3,
            filters=256,
            kernel_size=[3,3],
            strides=1,
            padding='same',
            activation=tf.nn.selu,
            name='conv4')
        
        # ---- conv layer 5
        conv5 = tf.layers.conv2d(
            inputs=conv4,
            filters=256,
            kernel_size=[3,3],
            strides=1,
            padding='same',
            activation=tf.nn.selu,
            name='conv5')
        pool5 = tf.layers.max_pooling2d(
            inputs=conv5,
            pool_size=[2,2],
            strides=2,
            name='pool5')
        pool5_flat = tf.layers.flatten(
            pool5,
            name='pool5_flat')
        
        # ---- full layer 6
        fc6 = tf.layers.dense(
            inputs=pool5_flat,
            units=2048,
            activation=tf.nn.selu,
            name='fc6'
        )
        
        # ---- full layer 7
        fc7 = tf.layers.dense(
            inputs=fc6,
            units=512,
            activation=tf.nn.selu,
            name='fc7'
        )
        
        # ---- full layer 8
        fc8 = tf.layers.dense(
            inputs=fc7,
            units=output_dim,
            name='fc8'
        )
        
        return fc8
    
    
class DVBPR_Network:
    def __init__(self, n_users, n_items, latent_dim):        
        
        # ------------------------
        # ---- placeholders
        
        self._RGB_images = tf.placeholder(shape=[None, 224, 224, 3], dtype=tf.float32,
                                                     name='RGB_images')
        self._user_index = tf.placeholder(shape=[None], dtype=tf.int32,
                                          name='user_index')
        self._positive_item_index = tf.placeholder(shape=[None], dtype=tf.int32,
                                                   name='positive_item_index')
        self._negative_item_index = tf.placeholder(shape=[None], dtype=tf.int32,
                                                   name='negative_item_index')
        self._learning_rate = tf.placeholder(shape=[], dtype=tf.float32)
            
        # ------------------------------------
        # ---- Global trainable tensors
        
        # -- user latent factor matrix
        # (n_users x user_latent_dim)
        self._user_latent_factors = tf.Variable(
            tf.random_uniform([n_users, latent_dim], -1.0, 1.0),
            name='user_latent_factors'
        )
        
        # -------------------------------
        # ---- minibatch tensors
        
        # -- user
        self._user_latent_vector = tf.gather(self._user_latent_factors, self._user_index)
        
        # -- positive item
        self._pos_image = tf.gather(self._RGB_images, self._positive_item_index)
        self._pos_vector = CNN_F(self._pos_image, latent_dim)
        self._pos_score = tf.reduce_sum(self._user_latent_vector * self._pos_vector, 1)
        
        # -- negative item
        self._neg_image = tf.gather(self._RGB_images, self._negative_item_index)
        self._neg_vector = CNN_F(self._neg_image, latent_dim)
        self._neg_score = tf.reduce_sum(self._user_latent_vector * self._neg_vector, 1)
        
        # -------------------------------
        # ---- train-test tensors
        
        # -- train loss
        delta_score = self._pos_score - self._neg_score
        ones = tf.fill(tf.shape(self._user_latent_vector)[:1], 1.0)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=delta_score, labels=ones)
        loss = tf.reduce_mean(loss, name='train_loss')
        self._train_loss = loss
        
        # -- test accuracy
        accuracy = tf.reduce_sum(tf.cast(delta_score > .0, tf.float32), name='test_accuracy')
        self._test_accuracy = accuracy
        
        # -- optimizer
        self._optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(self._train_loss)
    
    def optimize_and_get_train_loss(self, sess, RGB_images, user_index, positive_item_index,
                                    negative_item_index, learning_rate):
        return sess.run([
            self._optimizer,
            self._train_loss,
        ], feed_dict={
            self._RGB_images: RGB_images,
            self._user_index: user_index,
            self._positive_item_index: positive_item_index,
            self._negative_item_index: negative_item_index,
            self._learning_rate: learning_rate,
        })
    
    def get_test_accuracy(self, sess, RGB_images, user_index, positive_item_index, negative_item_index):
        return sess.run(
            self._test_accuracy, feed_dict={
            self._RGB_images: RGB_images,
            self._user_index: user_index,
            self._positive_item_index: positive_item_index,
            self._negative_item_index: negative_item_index,
        })