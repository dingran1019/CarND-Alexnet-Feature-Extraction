import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import pandas as pd
import numpy as np
import time
import os
import inspect
from tqdm import tqdm
from sklearn.utils import shuffle

sign_names = pd.read_csv('signnames.csv')
nb_classes = 43


def main():
    if 0:
        training_file = 'data/train.p'
        validation_file = 'data/valid.p'
        testing_file = 'data/test.p'

        with open(training_file, mode='rb') as f:
            train = pickle.load(f)
        with open(validation_file, mode='rb') as f:
            valid = pickle.load(f)
        with open(testing_file, mode='rb') as f:
            test = pickle.load(f)

        X_train, y_train = train['features'], train['labels']
        X_valid, y_valid = valid['features'], valid['labels']
        X_test, y_test = test['features'], test['labels']

        result = train_model(X_train, y_train, X_valid, y_valid, X_test, y_test, resuming=False, learning_rate=0.001)
    else:
        training_file = 'train.p'

        with open(training_file, mode='rb') as f:
            data = pickle.load(f)

        X_train, X_val, y_train, y_val = train_test_split(data['features'], data['labels'], test_size=0.33,
                                                          random_state=0)

        result = train_model(X_train, y_train, X_val, y_val, X_val, y_val, resuming=False, learning_rate=0.001)


def train_model(X_train, y_train, X_valid, y_valid, X_test, y_test,
                resuming=False,
                learning_rate=0.001, max_epochs=1001, batch_size=128,
                early_stopping_enabled=True, early_stopping_patience=10,
                log_epoch=1, print_epoch=1,
                top_k=5, return_top_k=False,
                plot_featuremap=False):
    print('========= train_model() arguments: ==========')
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    for i in args[6:]:
        print("{} = {}".format(i, values[i]))
    print('=============================================')

    model_dir = os.path.join(os.getcwd(), 'models', '001')
    os.makedirs(model_dir, exist_ok=True)
    print('model dir: {}'.format(model_dir))
    model_fname = os.path.join(model_dir, 'model_cpkt')
    model_fname_best_epoch = os.path.join(model_dir, 'best_epoch')
    model_train_history = os.path.join(model_dir, 'training_history.npz')

    start = time.time()

    graph = tf.Graph()
    with graph.as_default():
        # Input data. For the training data, we use a placeholder that will be fed at run time with a training minibatch.
        x = tf.placeholder(tf.float32, (None, 32, 32, X_test.shape[-1]))
        y = tf.placeholder(tf.int32, (None))
        one_hot_y = tf.one_hot(y, nb_classes)
        is_training = tf.placeholder(tf.bool)

        resized = tf.image.resize_images(x, (227, 227))

        fc7 = AlexNet(resized, feature_extract=True)
        # NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
        # past this point, keeping the weights before and up to `fc7` frozen.
        # This also makes training faster, less work to do!
        fc7 = tf.stop_gradient(fc7)

        shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix
        fc8W = tf.get_variable(name='fc8W', shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        fc8b = tf.get_variable(name='fc8b', shape=nb_classes, initializer=tf.constant_initializer(0.0))
        logits = tf.matmul(fc7, fc8W) + fc8b

        predictions = tf.nn.softmax(logits)
        top_k_predictions = tf.nn.top_k(predictions, top_k)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
        loss_operation = tf.reduce_mean(cross_entropy, name='loss')
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        training_operation = optimizer.minimize(loss_operation)
        pred_y = tf.argmax(logits, 1, name='prediction')
        actual_y = tf.argmax(one_hot_y, 1)
        correct_prediction = tf.equal(pred_y, actual_y)
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())

        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            print(variable)
            print(shape)
            # print(len(shape))
            variable_parametes = 1
            for dim in shape:
                # print(dim)
                variable_parametes *= dim.value
            # print(variable_parametes)
            total_parameters += variable_parametes
        print('total # of parameters: ', total_parameters)

        def output_top_k(X_data):
            top_k_preds = sess.run([top_k_predictions], feed_dict={x: X_data, is_training: False})
            return top_k_preds

        def evaluate(X_data, y_data, aux_output=False):
            n_data = len(X_data)
            correct_pred = np.array([])
            y_pred = np.array([])
            y_actual = np.array([])
            loss_batch = np.array([])
            acc_batch = np.array([])
            batch_sizes = np.array([])
            for offset in range(0, n_data, batch_size):
                batch_x, batch_y = X_data[offset:offset + batch_size], y_data[offset:offset + batch_size]
                batch_sizes = np.append(batch_sizes, batch_y.shape[0])

                if aux_output:
                    accuracy, loss, cp_, yp_, ya_ = \
                        sess.run([accuracy_operation, loss_operation, correct_prediction, pred_y, actual_y],
                                 feed_dict={x: batch_x, y: batch_y, is_training: False})

                    correct_pred = np.append(correct_pred, cp_)
                    y_pred = np.append(y_pred, yp_)
                    y_actual = np.append(y_actual, ya_)
                else:
                    accuracy, loss = sess.run([accuracy_operation, loss_operation],
                                              feed_dict={x: batch_x, y: batch_y, is_training: False})

                loss_batch = np.append(loss_batch, loss)
                acc_batch = np.append(acc_batch, accuracy)

            final_acc = np.average(acc_batch, weights=batch_sizes)
            final_loss = np.average(loss_batch, weights=batch_sizes)

            if aux_output:
                return final_acc, final_loss, correct_pred, y_pred, y_actual
            else:
                return final_acc, final_loss

        # If we chose to keep training previously trained model, restore session.
        if resuming:
            try:
                tf.train.Saver().restore(sess, model_fname)
                print('Restored session from {}'.format(model_fname))
            except Exception as e:
                print("Failed restoring previously trained model: file does not exist.")
                print("Trying to restore from best epoch from previously training session.")
                try:
                    tf.train.Saver().restore(sess, model_fname_best_epoch)
                    print('Restored session from {}'.format(model_fname_best_epoch))
                except Exception as e:
                    print("Failed to restore, will train from scratch now.")

                    # print([v.op.name for v in tf.all_variables()])
                    # print([n.name for n in tf.get_default_graph().as_graph_def().node])

        saver = tf.train.Saver()
        early_stopping = EarlyStopping(tf.train.Saver(), sess, patience=early_stopping_patience, minimize=True,
                                       restore_path=model_fname_best_epoch)

        train_loss_history = np.empty([0], dtype=np.float32)
        train_accuracy_history = np.empty([0], dtype=np.float32)
        valid_loss_history = np.empty([0], dtype=np.float32)
        valid_accuracy_history = np.empty([0], dtype=np.float32)
        if max_epochs > 0:
            print("================= TRAINING ==================")
        else:
            print("================== TESTING ==================")
        print(" Timestamp: " + get_time_hhmmss())

        for epoch in range(max_epochs):
            X_train, y_train = shuffle(X_train, y_train)

            for offset in tqdm(range(0, X_train.shape[0], batch_size)):
                end = offset + batch_size
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, is_training: True})

            # If another significant epoch ended, we log our losses.
            if epoch % log_epoch == 0:
                train_accuracy, train_loss = evaluate(X_train, y_train)
                valid_accuracy, valid_loss = evaluate(X_valid, y_valid)

                if epoch % print_epoch == 0:
                    print("-------------- EPOCH %4d/%d --------------" % (epoch, max_epochs))
                    print("     Train loss: %.8f, accuracy: %.2f%%" % (train_loss, 100 * train_accuracy))
                    print("Validation loss: %.8f, accuracy: %.2f%%" % (valid_loss, 100 * valid_accuracy))
                    print("      Best loss: %.8f at epoch %d" % (
                        early_stopping.best_monitored_value, early_stopping.best_monitored_epoch))
                    print("   Elapsed time: " + get_time_hhmmss(start))
                    print("      Timestamp: " + get_time_hhmmss())
            else:
                valid_loss = 0.
                valid_accuracy = 0.
                train_loss = 0.
                train_accuracy = 0.

            valid_loss_history = np.append(valid_loss_history, [valid_loss])
            valid_accuracy_history = np.append(valid_accuracy_history, [valid_accuracy])
            train_loss_history = np.append(train_loss_history, [train_loss])
            train_accuracy_history = np.append(train_accuracy_history, [train_accuracy])

            if early_stopping_enabled:
                # Get validation data predictions and log validation loss:
                if valid_loss == 0:
                    _, valid_loss = evaluate(X_valid, y_valid)
                if early_stopping(valid_loss, epoch):
                    print("Early stopping.\nBest monitored loss was {:.8f} at epoch {}.".format(
                        early_stopping.best_monitored_value, early_stopping.best_monitored_epoch
                    ))
                    break

        # Evaluate on test dataset.
        valid_accuracy, valid_loss, valid_cp, valid_yp, valid_ya = evaluate(X_valid, y_valid, aux_output=True)
        test_accuracy, test_loss, test_cp, test_yp, test_ya = evaluate(X_test, y_test, aux_output=True)
        print("=============================================")
        print(" Valid loss: %.8f, accuracy = %.2f%%)" % (valid_loss, 100 * valid_accuracy))
        print(" Test loss: %.8f, accuracy = %.2f%%)" % (test_loss, 100 * test_accuracy))
        print(" Total time: " + get_time_hhmmss(start))
        print("  Timestamp: " + get_time_hhmmss())

        # Save model weights for future use.
        saved_model_path = saver.save(sess, model_fname)
        print("Model file: " + saved_model_path)
        np.savez(model_train_history, train_loss_history=train_loss_history,
                 train_accuracy_history=train_accuracy_history, valid_loss_history=valid_loss_history,
                 valid_accuracy_history=valid_accuracy_history)
        print("Train history file: " + model_train_history)

        if return_top_k:
            top_k_preds = output_top_k(X_test)

    result_dict = dict(test_accuracy=test_accuracy, test_loss=test_loss, test_cp=test_cp, test_yp=test_yp,
                       test_ya=test_ya,
                       valid_accuracy=valid_accuracy, valid_loss=valid_loss, valid_cp=valid_cp, valid_yp=valid_yp,
                       valid_ya=valid_ya)
    if return_top_k:
        return result_dict, top_k_preds
    else:
        return result_dict


class EarlyStopping(object):
    """
    Provides early stopping functionality. Keeps track of model accuracy,
    and if it doesn't improve over time restores last best performing
    parameters.
    """

    def __init__(self, saver, session, patience=100, minimize=True, restore_path=None):
        """
        Initialises a `EarlyStopping` isntance.

        Parameters
        ----------
        saver     :
                    TensorFlow Saver object to be used for saving and restoring model.
        session   :
                    TensorFlow Session object containing graph where model is restored.
        patience  :
                    Early stopping patience. This is the number of epochs we wait for
                    accuracy to start improving again before stopping and restoring
                    previous best performing parameters.

        Returns
        -------
        New instance.
        """
        self.minimize = minimize
        self.patience = patience
        self.saver = saver
        self.session = session
        self.best_monitored_value = np.inf if minimize else 0.
        self.best_monitored_epoch = 0
        self.restore_path = restore_path

    def __call__(self, value, epoch):
        """
        Checks if we need to stop and restores the last well performing values if we do.

        Parameters
        ----------
        value     :
                    Last epoch monitored value.
        epoch     :
                    Last epoch number.

        Returns
        -------
        `True` if we waited enough and it's time to stop and we restored the
        best performing weights, or `False` otherwise.
        """
        if (self.minimize and value < self.best_monitored_value) or (
                    not self.minimize and value > self.best_monitored_value):
            self.best_monitored_value = value
            self.best_monitored_epoch = epoch
            self.saver.save(self.session, self.restore_path)
        elif self.best_monitored_epoch + self.patience < epoch:
            if self.restore_path is not None:
                self.saver.restore(self.session, self.restore_path)
            else:
                print("ERROR: Failed to restore session")
            return True

        return False


def get_time_hhmmss(start=None):
    """
    Calculates time since `start` and formats as a string.
    """
    if start is None:
        return time.strftime("%Y/%m/%d %H:%M:%S")
    end = time.time()
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    time_str = "%02d:%02d:%02d" % (h, m, s)
    return time_str


if __name__ == '__main__':
    main()
