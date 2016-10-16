
# coding: utf-8

# In[ ]:

import tensorflow as tf
import segmentation_model

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_dir', '',
                           """Path to checkpoint file.""")
tf.app.flags.DEFINE_string('output_graph_file', 'graph.pbtxt',
                           """Output graph filename.""")

def main(unused_args):
    with tf.Graph().as_default():
        image_placeholder = tf.placeholder(tf.float32, shape=(None,None,None,3))
        logits = segmentation_model.inference(image_placeholder)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                print('Resotring checkpoint %s' % ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found')
            tf.train.write_graph(sess.graph_def, FLAGS.checkpoint_dir, FLAGS.output_graph_file)
        
if __name__ == "__main__":
    tf.app.run()

