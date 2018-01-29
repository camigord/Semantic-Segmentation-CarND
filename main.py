import time
from datetime import timedelta
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import pickle


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


DATA_DIR = './data'
RUNS_DIR = './runs'
SAVE_DIR = './model'

L2_REG = 1e-4
STDEV = 1e-3
KEEP_PROB = 0.5
LEARNING_RATE = 1e-4
EPOCHS = 50
BATCH_SIZE = 10
IMAGE_SHAPE = (160, 576)
NUM_CLASSES = 2


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    image_input = sess.graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = sess.graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = sess.graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = sess.graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = sess.graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    layer7_conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, 1,
                                       padding='same', kernel_initializer=tf.random_normal_initializer(stddev=STDEV),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))
    output = tf.layers.conv2d_transpose(layer7_conv_1x1, num_classes, 4, 2,
                                        padding='same', kernel_initializer=tf.random_normal_initializer(stddev=STDEV),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))
    layer4_conv_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, 1,
                                       padding='same', kernel_initializer=tf.random_normal_initializer(stddev=STDEV),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))
    output = tf.add(output, layer4_conv_1x1)
    output = tf.layers.conv2d_transpose(output, num_classes, 4, 2,
                                        padding='same', kernel_initializer=tf.random_normal_initializer(stddev=STDEV),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))
    layer3_conv_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, 1,
                                       padding='same', kernel_initializer=tf.random_normal_initializer(stddev=STDEV),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))
    output = tf.add(output, layer3_conv_1x1)
    output = tf.layers.conv2d_transpose(output, num_classes, 16, 8,
                                        padding='same', kernel_initializer=tf.random_normal_initializer(stddev=STDEV),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))
    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))

    # Adding regularization to loss function
    reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    regularization_loss = tf.reduce_sum(reg_variables)
    total_loss = cross_entropy_loss + regularization_loss

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
    return logits, train_op, total_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, saver = None, data_dir = None):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    best_loss = float('inf')
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        s_time = time.time()
        num_batches = 0
        avg_loss = 0
        for image, ground_truth in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                feed_dict = {input_image: image, correct_label: ground_truth, keep_prob: KEEP_PROB,
                             learning_rate: LEARNING_RATE })

            avg_loss += loss
            num_batches += 1

        avg_loss /= num_batches
        train_losses.append(avg_loss)

        # Get loss in test set
        test_imgs, gt_test = helper.gen_test_batch(os.path.join(DATA_DIR, 'data_road/training'), IMAGE_SHAPE)
        loss = sess.run(cross_entropy_loss, feed_dict = {input_image: test_imgs, correct_label: gt_test, keep_prob: 1.0})
        test_losses.append(loss)

        print("Epoch: {}".format(epoch + 1), "/ {}".format(epochs), " Avg. Loss: {:.3f}".format(avg_loss), " Time: ",
              str(timedelta(seconds=(time.time() - s_time))))

        # Saves the best model so far
        if saver is not None and (epoch + 1) % 5 == 0 and loss < best_loss: # Save every 5 epochs if loss is decreasing
            best_loss = loss
            save_path = saver.save(sess, os.path.join(data_dir, 'best_epoch.ckpt'))

    # Saves the final model
    if saver is not None:
        save_path = saver.save(sess, os.path.join(data_dir, 'epoch_' + str(epochs) + '.ckpt'))  # Save model in last epoch

        # Save losses for visualization
        losses = {'train': train_losses,
                  'test': test_losses}
        pickle.dump(obj=losses ,file=open(os.path.join(data_dir, 'losses.p'), mode='wb'))

tests.test_train_nn(train_nn)


def run():
    tf.set_random_seed(123)
    tests.test_for_kitti_dataset(DATA_DIR)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(DATA_DIR)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(DATA_DIR, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(DATA_DIR, 'data_road/training'), IMAGE_SHAPE)

        # OPTIONAL: Augment Images for better results
        # Data augmentation is performed in helpers.gen_batch_function()

        # Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        output = layers(layer3, layer4, layer7, NUM_CLASSES)

        correct_label = tf.placeholder(dtype = tf.float32, shape = (None, None, None, NUM_CLASSES))
        learning_rate = tf.placeholder(dtype = tf.float32)

        logits, train_op, cross_entropy_loss = optimize(output, correct_label, learning_rate, NUM_CLASSES)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        # Train NN using the train_nn function
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label,
                 keep_prob, learning_rate, saver, SAVE_DIR)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(RUNS_DIR, DATA_DIR, sess, IMAGE_SHAPE, logits, keep_prob, input_image)


if __name__ == '__main__':
    run()
