

""" Some repeat use logging utilities for working with Tensorboard """
import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector



class TBLogger():
    def __init__(self, logdir):
        self.train_logs = os.path.abspath("../logs/train/"+logdir) 
        self.test_logs = os.path.abspath("../logs/test"+logdir) 
        self.train_writer = tf.summary.FileWriter(logdir=self.train_logs)
        self.test_writer = tf.summary.FileWriter(logdir=self.test_logs)

        self.summary_merge = None

    def write_train_state(self, summary, epoch):
        self.train_writer.add_summary(summary, epoch)


    def write_train_graph(self, sess):
        self.train_writer.add_graph(sess.graph)


    def write_test_state(self, summary, epoch):
        self.test_writer.add_summary(summary, epoch)


    def image_summary(self, image):
        tf.summary.image('input', tf.reshape(image, [-1, 28, 28, 1]), 3)


    def merge_summaries(self):
        return tf.summary.merge_all() 


    def get_train_log_dir(self):
        return self.train_logs


    def get_test_log_dir(self):
        return self.test_logs


class Projector():
    def __init__(self, log_dir, metadata_labels, metadata_data):
        """ Projector objest for tensorboard.
        args: str(log_dir), numpy.ndarray(metadata_labels), numpy.ndarray(metadata_data)
        """
        self.log_dir = log_dir
        self.metadata_labels = metadata_labels
        self.dataset = tf.Variable(metadata_data, name = 'dataset')
        self.metad = log_dir+'/metadata.tsv'
        with open(self.metad, 'w') as metad_file:
            for row in self.metadata_labels:
                metad_file.write('%d\n' % row)

        self.config = projector.ProjectorConfig()
        self.embedding = self.config.embeddings.add()
        self.embedding.tensor_name = self.dataset.name
        self.embedding.metadata_path = self.metad

        self.embedding.sprite.image_path = os.path.abspath('../docs/mnist_10k_sprite.png') 
        self.embedding.sprite.single_image_dim.extend([28, 28])

        projector.visualize_embeddings(tf.summary.FileWriter(self.log_dir), self.config)



    def update(self, sess, step):
        saver = tf.train.Saver([self.dataset])

        sess.run(self.dataset.initializer)
        saver.save(sess, self.log_dir+'/projector_dataset.ckpt', step) 



