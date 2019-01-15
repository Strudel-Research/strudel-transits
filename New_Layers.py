# import tensorflow as tf
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import keras
from keras.models import Model, Sequential
from keras.layers import Conv1D, Dense, Flatten, Input, MaxPooling1D, LSTM,  add, MaxPooling2D, GlobalMaxPooling1D, \
    Reshape, concatenate, BatchNormalization, Activation
from keras.layers import Bidirectional, Dropout
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model

from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np
from tqdm import tqdm
# from spectogram import Spectrogram

from keras import backend as K
from keras.engine.topology import Layer
from keras.callbacks import Callback
from keras_tqdm import TQDMNotebookCallback, TQDMCallback
import os


# ---------------- Losses -----------------
smooth_coef = 0.

def absolute_true_error(y_true, y_pred):
    mismatch = K.abs(y_pred - y_true)    
    mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
    return mismatch * mask_true

def intersection_true_error(y_true, y_pred):
    mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
    mask_true_2 = K.cast(K.not_equal(y_pred, 0), K.floatx())
    
    y_true_normed = (mask_true * 2) - 1 #convert to [-1 , 1] range
    y_pred_normed = (mask_true_2 * 2) - 1 #convert to [-1 , 1] range
    
    absolute_mismatch = ((y_true_normed * y_pred_normed) - 1) / -2 #mult = -1 if they are not the same, then scale -1 to 1 and 1 to 0
    return absolute_mismatch * mask_true

def normalized_true_error(y_true, y_pred):
    mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
    mask_true_2 = K.cast(K.not_equal(y_pred, 0), K.floatx())
    
    y_true_normed = (mask_true * 2) - 1 #convert to [-1 , 1] range
    y_pred_normed = (mask_true_2 * 2) - 1 #convert to [-1 , 1] range
    
    absolute_mismatch = K.abs(y_pred - y_true)
    absolute_mismatch = K.cast(K.greater(absolute_mismatch, 0.01), K.floatx())

    return absolute_mismatch * mask_true


def absolute_false_error(y_true, y_pred):
    mismatch = K.abs(y_pred - y_true)    
    mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
    return mismatch * (1 - mask_true)

def intersection_false_error(y_true, y_pred):
    mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
    mask_true_2 = K.cast(K.not_equal(y_pred, 0), K.floatx())
    
    y_true_normed = (mask_true * 2) - 1 #convert to [-1 , 1] range
    y_pred_normed = (mask_true_2 * 2) - 1 #convert to [-1 , 1] range
    
    absolute_mismatch = ((y_true_normed * y_pred_normed) - 1) / -2 #mult = -1 if they are not the same, then scale -1 to 1 and 1 to 0
    return absolute_mismatch * (1 - mask_true)

def normalized_false_error(y_true, y_pred):
    mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
    mask_true_2 = K.cast(K.not_equal(y_pred, 0), K.floatx())
    
    y_true_normed = (mask_true * 2) - 1 #convert to [-1 , 1] range
    y_pred_normed = (mask_true_2 * 2) - 1 #convert to [-1 , 1] range
    
    absolute_mismatch = K.abs(y_pred - y_true)
    absolute_mismatch = K.cast(K.greater(absolute_mismatch, 0.01), K.floatx())
    return absolute_mismatch * (1 - mask_true)


def intersection(y_true, y_pred):
    mask_true = K.tanh(1000 * y_true / K.min(K.max(K.batch_flatten(y_true), axis=-1))) #approximate not_equal(y_true,0)
    mask_true_2 = K.tanh(1000 * y_pred / K.min(K.max(K.batch_flatten(y_true), axis=-1))) #approximate not_equal(y_pred,0)
    
    y_true_f = K.batch_flatten(mask_true)
    y_pred_f = K.batch_flatten(mask_true_2)
    
    y_true_f = (y_true_f * 2) - 1 #convert to [-1 , 1] range
    y_pred_f = (y_pred_f * 2) - 1 #convert to [-1 , 1] range
    
    missmatch = ((y_true_f * y_pred_f) - 1) / -2 #mult = -1 if they are not the same, then scale -1 to 1 and 1 to 0
    
    y_true_f = (y_true_f+1)/2 #convert back to [0,1]
    y_pred_f = (y_pred_f+1)/2 #convert back to [0,1]

    transit_missmatch = K.sum((missmatch * y_true_f), axis=-1) / K.sum(y_true_f, axis=-1)
    no_transit_missmatch = K.sum((missmatch * (1-y_true_f)), axis=-1) / K.sum(1-y_true_f, axis=-1)

    return transit_missmatch + no_transit_missmatch

def intersection_loss(y_true, y_pred):
    return intersection(y_true, y_pred) - dice_coef1(y_true, y_pred)

def j_coef1(y_true, y_pred):
    S = dice_coef1(y_true, y_pred)
    return S / (2-S)

def j_loss(y_true, y_pred):
    return -j_coef1(y_true, y_pred)


def dice_coef1(y_true, y_pred):
    mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
    mask_true_2 = K.cast(K.not_equal(y_pred, 0), K.floatx())
    y_true_f = K.flatten(mask_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth_coef) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth_coef)

def dice_coef2(y_true, y_pred):
    mask_true = K.cast(K.equal(y_true, 0), K.floatx())
    y_true_f = K.flatten(mask_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth_coef) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth_coef)

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth_coef) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth_coef)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef1(y_true, y_pred)


def masked_mse(y_true, y_pred):
        mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
        masked_squared_error = K.square(mask_true * (y_true - y_pred))
        masked_mse = K.sum(masked_squared_error, axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1)
        return masked_mse

def dice_coef_loss_times_mse(y_true, y_pred):
    return (-dice_coef(y_true, y_pred)) + (K.mean(K.square( ( (y_pred - y_true) * y_true ) ), axis=-1) )

# ---------------- Losses -----------------


# ---------------- Evaluation -----------------

class PlotDataCallback(keras.callbacks.Callback):
    def __init__(self, predict_batch_size=80, include_on_batch=False):
        super(PlotDataCallback, self).__init__()
        self.predict_batch_size = predict_batch_size
        self.include_on_batch = include_on_batch

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        pass

    def on_train_begin(self, logs={}):
        pass

    def on_train_end(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        if (self.validation_data):
            y_pred = self.model.predict(self.validation_data[0][0:1])[0, :20610]
            y_test = self.validation_data[1][0, :20610]
            time = np.linspace(0, 28.625, 20610)
            print y_pred.shape, y_test.shape, time.shape
            
            plt.figure(1, figsize=(10,10))
            plt.subplot(211)
            plt.scatter(time, y_test, s=0.5)
            plt.title("Simulation Output")
            
            plt.subplot(212)
            plt.scatter(time, y_pred, s=0.5)
            plt.title("Neural Net Output")
            
            plt.tight_layout()
            plt.show()



def mape_non_zero(y_true, y_pred):
    neqz = K.cast(K.not_equal(y_true, 0), np.float32)
    return keras.losses.mean_absolute_percentage_error(neqz * y_true, neqz * y_pred) * K.sum(K.ones_like(neqz))/K.sum(neqz)


def mae_non_zero(y_true, y_pred):
    neqz = K.cast(K.not_equal(y_true, 0), np.float32)
    return keras.losses.mean_absolute_error(neqz * y_true, neqz * y_pred) * K.sum(K.ones_like(neqz))/K.sum(neqz)


class Split1D(keras.layers.Layer):
    def __init__(self, splits, **kwargs):
        assert isinstance(splits, int), "Must be an integer number of splits"
        assert splits > 0, "Invalid vecotrs, supply at least 1 output vector length"
        self.splits = splits
        self.window_size = 0
        self.stride = 0
        super(Split1D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.window_size = int(np.ceil(float(input_shape[1]) / self.splits))
        self.stride = int(np.floor(input_shape[1] / self.splits))
        self.in_shape = input_shape
        super(Split1D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        new_vec_list = []
        for i in range(self.splits):
            if (self.stride * i + self.window_size) <= int(self.in_shape[1]):
                temp = x[:, self.stride * i: self.stride * i + self.window_size]
            else:
                temp = x[:, self.stride * i: int(self.in_shape[1])]
                temp = K.temporal_padding(temp, (0, self.window_size - K.int_shape(temp)[1]))
            new_vec_list.append(temp)
        return new_vec_list

    def compute_output_shape(self, input_shape):
        outputs = []
        for i in range(self.splits):
            outputs.append((input_shape[0], int(np.ceil(float(input_shape[1]) / self.splits)), input_shape[2]))
        return outputs

    def get_config(self):
        return dict(list(super(Split1D, self).get_config().items()) + list({'splits': self.splits}.items()))


'''
class internal_Fold1D(keras.layers.Layer):
    def __init__(self, length, **kwargs):
        self.length = length
        super(internal_Fold1D, self).__init__(**kwargs)

    def build(self, input_shape):
        super(internal_Fold1D, self).build(input_shape)  # Be sure to call this at the end

    def slice_tensor(self, x, max_i,i):
        if K.equal(i-1, max_i) is not K.variable([True]):
            return K.spatial_2d_padding(x[:, self.length*i:, :, :], padding=((0,self.length*(i+1) - K.int_shape(x)[1]),
                                                                             (0, 0)))
        else:
            return x[:, self.length*i:self.length*(i+1)]
    def call(self, x):
        new_vec_list = []
        x = K.expand_dims(x)
        i = 0
        max_i = K.int_shape(x)[1]//self.length
        i_list = K.arange(0,max_i+1, dtype=np.int32)
        #self.length = K.cast(self.length, np.int32)  # TODO figure out while loops with tensors
        new_vec_list = K.map_fn(fn=lambda i: self.slice_tensor(x, max_i, i), elems=i_list, dtype="float32")
        temp = K.mean(new_vec_list, axis=-1)
        temp = K.mean(temp, axis=-1)
        return temp

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.length, input_shape[2]

    def get_config(self):
        return dict(list(super(internal_Fold1D, self).get_config().items()) + list({'length': self.length}.items()))

class Fold1D(keras.layers.Layer):
    def __init__(self, folds, **kwargs):
        self.folds = folds
        self.new_layers_list = []
        super(Fold1D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.windows_init = np.random.randint(0, input_shape[1]//3, self.folds)
        self.windows = K.variable(self.windows_init, name="windows", dtype=np.int32)
        self.trainable_weights = [self.windows]
        super(Fold1D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        self.new_layers_list = []
        new_vec_list = []
        #x = K.expand_dims(x)
        for i in range(self.folds):
            self.new_layers_list.append(internal_Fold1D(self.windows[i]))
        for l in self.new_layers_list:
            new_vec_list.append(l(x))
        return new_vec_list

    def compute_output_shape(self, input_shape):
        output_shapes = []
        for l in self.new_layers_list:
            output_shapes.append(l.compute_output_shape(input_shape))
        return output_shapes

    def get_config(self):
        return dict(list(super(Fold1D, self).get_config().items()) + list({'folds': self.folds}.items()))

    def compute_mask(self, inputs, mask=None):
        return self.folds * [None]

'''


class Fold1D(keras.layers.Layer):
    def __init__(self, folds, **kwargs):
        self.folds = folds
        self.new_layers_list = []
        super(Fold1D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.windows = np.random.randint(2048, input_shape[1] / 3, self.folds)
        super(Fold1D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        self.new_layers_list = []
        out_vecs = []
        for window in self.windows:
            window = int(window)
            new_vec_list = []
            i = 0
            while K.int_shape(x)[1] > window * (i + 1):
                temp = x[:, window * i:window * (i + 1)]
                new_vec_list.append(temp)
                i += 1
            temp = x[:, window * i:]
            temp = K.temporal_padding(temp, padding=(0, window * (i + 1) - K.int_shape(x)[1]))
            new_vec_list.append(temp)
            temp = concatenate(new_vec_list, axis=-1)
            temp = K.mean(temp, axis=-1, keepdims=True)
            out_vecs.append(temp)
        return out_vecs

    def compute_output_shape(self, input_shape):
        output_shapes = []
        for window in self.windows:
            window = int(window)
            output_shapes.append((input_shape[0], window, input_shape[2]))
        return output_shapes

    def get_config(self):
        return dict(list(super(Fold1D, self).get_config().items()) + list({'folds': self.folds}.items()))

    def compute_mask(self, inputs, mask=None):
        return self.folds * [None]


class SPP(Layer):
    def __init__(self, out_vec_lengths, **kwargs):
        assert len(out_vec_lengths) > 0, "Invalid vecotrs, supply at least 1 output vector length"
        for i in out_vec_lengths:
            assert isinstance(i, int) or i < 1, "Invalid vector size, all vectors should be positive integers"
        self.ovl = out_vec_lengths
        super(SPP, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SPP, self).build(input_shape)  # Be sure to call this at the end

    def ceil(self, t):
        rounded = K.round(t)
        if K.equal(t, rounded) is not K.constant([False]):
            return K.cast(t, dtype=np.int32)
        else:
            return K.cast(t + 1, dtype=np.int32)

    def floor(self, t):
        return K.cast(t, dtype=np.int32)

    def call(self, x):
        new_vec_list = []
        shape = K.get_value(self.floor(K.int_shape(x)[1]))
        for l in self.ovl:
            temp_vec_list = []
            for i in range(l):
                self.window_size = K.get_value(self.ceil(K.int_shape(x)[1] / l))
                self.stride = K.get_value(self.floor(K.int_shape(x)[1] / l))
                if (self.stride * i + self.window_size) <= shape:
                    temp = x[:, self.stride * i: self.stride * i + self.window_size]
                else:
                    temp = x[:, self.stride * i:, :]
                    temp = K.temporal_padding(temp,
                                              (0, self.window_size - K.get_value(self.floor(K.int_shape(temp)[1]))))
                temp = K.max(temp, axis=1, keepdims=True)
                temp_vec_list.append(temp)
            if l == 1:
                new_vec_list = temp_vec_list
            else:
                new_vec_list.append(keras.layers.concatenate(temp_vec_list, axis=1))
        if len(self.ovl) == 1:
            return new_vec_list
        ret = keras.layers.concatenate(new_vec_list, axis=1)
        # print(self.compute_output_shape(x.shape))
        return ret

    def compute_output_shape(self, input_shape):
        return input_shape[0], sum(self.ovl), input_shape[2]

    def get_config(self):
        config = {'out_vec_lengths': self.ovl}
        base_config = super(SPP, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class int_Options_Conv1D(Layer):
    def __init__(self, kernel_initializer='glorot_uniform', **kwargs):
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        super(int_Options_Conv1D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_hat = self.add_weight(shape=(1,),
                                     initializer=self.kernel_initializer,
                                     name='W_hat')
        super(int_Options_Conv1D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return self.W_hat * x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer)
        }
        base_config = super(int_Options_Conv1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class CombinedAucMetricCallback(keras.callbacks.Callback):
    def __init__(self, predict_batch_size=80, include_on_batch=False):
        super(CombinedAucMetricCallback, self).__init__()
        self.predict_batch_size = predict_batch_size
        self.include_on_batch = include_on_batch

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        pass

    def on_train_begin(self, logs={}):
        if not ('roc_auc_val' in self.params['metrics']):
            self.params['metrics'].append('roc_auc_val')
        if not ('period_auc_val' in self.params['metrics']):
            self.params['metrics'].append('period_auc_val')

    def on_train_end(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        logs['roc_auc_val'] = float('-inf')
        logs['period_auc_val'] = float('-inf')
        if (self.validation_data):
            preds = self.model.predict(self.validation_data[0])
            y_test = self.validation_data[1][:, 0]
            p_test = self.validation_data[2]
            y_pred = preds[0][:, 0]
            p_pred = preds[1]
            fpr, tpr, thresholds_keras = roc_curve(y_test, y_pred)

            logs['roc_auc_val'] = roc_auc_score(y_test, y_pred)
            print("Classification AUC: " + str(logs['roc_auc_val']))

            #BLS = np.loadtxt(open("ROC_BLS_BT10.csv", "rb"), delimiter=",")
            #BLS_auc = auc(BLS[:, 0], BLS[:, 1])
            
            plot.plot([0, 1], [0, 1], 'k--')

            plot.plot(fpr, tpr, label='Keras (area = {:.5f})'.format(logs['roc_auc_val']))
            #plot.plot(BLS[:, 0], BLS[:, 1], label='BLS (area = {:.5f})'.format(BLS_auc))

            plot.xlabel('False positive rate')
            plot.ylabel('True positive rate')
            plot.title('ROC curve')
            plot.legend(loc='best')
            plot.ylim(0, 1)
            plot.xlim(0, 1)
            plot.show()

            logs['period_auc_val'], percentages, epsilon_range = p_epsilon_chart(p_test[p_test > 1], p_pred[p_test > 1])
            print("Period AUC: " + str(logs['period_auc_val']))

            plot.plot([0, 1], [0, 1], 'k--')
            plot.plot(epsilon_range, percentages, label='Keras (area = {:.5f})'.format(logs['period_auc_val']))
            plot.xlabel('epsilon value')
            plot.ylabel('percentage of periods within this epsilon')
            plot.title('epsilon period curve')
            plot.legend(loc='best')
            plot.ylim(0, 1)
            plot.xlim(0, 1)
            plot.show()

            plot.plot([0, 1], [0, 1], 'k--')
            plot.plot(epsilon_range, percentages, label='Keras (area = {:.5f})'.format(logs['period_auc_val']))
            plot.xlabel('epsilon value')
            plot.ylabel('percentage of periods within this epsilon')
            plot.title('epsilon period curve')
            plot.legend(loc='best')
            plot.ylim(0, 1)
            plot.xlim(0, 0.1)
            plot.show()

            period_fracs = np.clip(np.abs(1 - (p_pred[p_test > 1] / p_test[p_test > 1])), 0, 1)
            plot.hist(period_fracs, bins=100)
            plot.xlabel('epsilon value')
            plot.title('epsilon period curve')
            plot.show()

            period_fracs = np.clip(np.abs(p_pred[p_test > 1] - p_test[p_test > 1]), 0, 5)
            plot.hist(period_fracs, bins=100)
            plot.xlabel('period diff value')
            plot.title('period diff curve')
            plot.show()

class RocAucMetricCallback(keras.callbacks.Callback):
    def __init__(self, predict_batch_size=80, include_on_batch=False):
        super(RocAucMetricCallback, self).__init__()
        self.predict_batch_size = predict_batch_size
        self.include_on_batch = include_on_batch

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        pass

    def on_train_begin(self, logs={}):
        if not ('roc_auc_val' in self.params['metrics']):
            self.params['metrics'].append('roc_auc_val')

    def on_train_end(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        logs['roc_auc_val'] = float('-inf')
        if (self.validation_data):
            y_pred = self.model.predict(self.validation_data[0])[:, 0]
            y_test = self.validation_data[1][:, 0]
            fpr, tpr, thresholds_keras = roc_curve(y_test, y_pred)
            logs['roc_auc_val'] = roc_auc_score(y_test, y_pred)
            print(logs['roc_auc_val'])

            plot.plot([0, 1], [0, 1], 'k--')
            plot.plot(fpr, tpr, label='Keras (area = {:.5f})'.format(logs['roc_auc_val']))
            plot.xlabel('False positive rate')
            plot.ylabel('True positive rate')
            plot.title('ROC curve')
            plot.legend(loc='best')
            plot.ylim(0, 1)
            plot.xlim(0, 1)
            plot.show()
            ''''''

class PeriodAucMetricCallback(keras.callbacks.Callback):
    def __init__(self, predict_batch_size=80, include_on_batch=False):
        super(PeriodAucMetricCallback, self).__init__()
        self.predict_batch_size = predict_batch_size
        self.include_on_batch = include_on_batch

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        pass

    def on_train_begin(self, logs={}):
        if not ('period_auc_val' in self.params['metrics']):
            self.params['metrics'].append('period_auc_val')

    def on_train_end(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        logs['period_auc_val'] = float('-inf')
        if (self.validation_data):
            y_pred = self.model.predict(self.validation_data[0])[:]
            y_test = self.validation_data[1][:]
            logs['period_auc_val'], percentages, epsilon_range = p_epsilon_chart(y_test, y_pred)
            # print(logs['period_auc_val'])

            plot.plot([0, 1], [0, 1], 'k--')
            plot.plot(epsilon_range, percentages, label='Keras (area = {:.5f})'.format(logs['period_auc_val']))
            plot.xlabel('epsilon value')
            plot.ylabel('percentage of periods within this epsilon')
            plot.title('epsilon period curve')
            plot.legend(loc='best')
            plot.ylim(0, 1)
            plot.xlim(0, 1)
            plot.show()

            plot.plot([0, 1], [0, 1], 'k--')
            plot.plot(epsilon_range, percentages, label='Keras (area = {:.5f})'.format(logs['period_auc_val']))
            plot.xlabel('epsilon value')
            plot.ylabel('percentage of periods within this epsilon')
            plot.title('epsilon period curve')
            plot.legend(loc='best')
            plot.ylim(0, 1)
            plot.xlim(0, 0.1)
            plot.show()

            period_fracs = np.clip(np.abs(1 - (y_pred[y_test > 0] / y_test[y_test > 0])), 0, 1)
            plot.hist(period_fracs, bins=100)
            plot.xlabel('epsilon value')
            plot.title('epsilon period curve')
            plot.show()

            period_fracs = np.clip(np.abs(y_pred[y_test > 0] - y_test[y_test > 0]), 0, 10)
            plot.hist(period_fracs, bins=100)
            plot.xlabel('period diff value')
            plot.title('period diff curve')
            plot.show()

def predict_period(test_p, pred_p, header=''):
    auc_p, percentages, epsilon_range = p_epsilon_chart(test_p, pred_p)
    plot.plot([0, 1], [0, 1], 'k--')
    plot.plot(epsilon_range, percentages, label='Keras (area = {:.5f})'.format(auc_p))
    plot.xlabel('epsilon value')
    plot.ylabel('percentage of periods within this epsilon')
    plot.title('epsilon period curve')
    plot.legend(loc='best')
    plot.ylim(0, 1)
    plot.xlim(0, 0.1)
    plot.savefig(header + 'Period-ROC.png')
    plot.show()

    period_fracs = np.clip(np.abs(1 - (pred_p[test_p > 0] / test_p[test_p > 0])), 0, 1)
    plot.hist(period_fracs, bins=100)
    plot.xlabel('epsilon value')
    plot.title('epsilon period curve')
    plot.show()

    period_fracs = np.clip(np.abs(pred_p[test_p > 0] - test_p[test_p > 0]), 0, 10)
    plot.hist(period_fracs, bins=100)
    plot.xlabel('epsilon value')
    plot.title('epsilon period curve')
    plot.show()

def predict_transit(test_y, pred_y, header=''):
    snr_y = np.load(header + 'total_SNR_sim_test_3.npy')
    fpr, tpr, thresholds_keras = roc_curve(test_y[:, 0], pred_y[:, 0])
    print(tpr[230:315])
    print(fpr[230:315])
    print(thresholds_keras[230:315])
    plot.figure(1)
    plot.plot([0, 1], [0, 1], 'k--')

    to_csv = np.stack([np.array(fpr), np.array(tpr)], axis=1)
    np.savetxt(header + 'DNNROC.csv', to_csv, delimiter=',')

    # auc_keras = auc(fpr,tpr)
    auc_keras = roc_auc_score(test_y[:, 0], pred_y[:, 0])
    plot.plot(fpr, tpr, label='Keras (area = {:.5f})'.format(auc_keras))
    plot.xlabel('False positive rate')
    plot.ylabel('True positive rate')
    plot.title('ROC curve')
    plot.legend(loc='best')
    plot.ylim(0, 1)
    plot.xlim(0, 1)
    plot.savefig(header + 'ROC-Curve.png', dpi=300)
    plot.show()

    thresh = thresholds_keras[np.argmin(np.abs(fpr - 0.001))]
    print(thresh)
    FN_rp = []
    FN_SNR = []
    FP_counter = 0
    for i in tqdm(range(test_y.shape[0])):
        if test_y[i, 0] == 1 and pred_y[i, 0] < thresh:
            FN_SNR += [snr_y[i, 0]]
        elif test_y[i, 0] == 0 and pred_y[i, 0] >= thresh:
            FP_counter += 1
    print(len(FN_SNR), FP_counter, np.sum(test_y[:, 0] == 1), test_y.shape[0])
    snr_data = np.clip(snr_y[test_y[:, 0] == 1][:, 0], a_min=0, a_max=500)
    plot.hist(snr_data, bins=100)
    plot.hist(FN_SNR + [500], bins=100)
    plot.title("FN SNR Hist")
    plot.savefig(header + "FN-SNR-HIST.png", dpi=300)
    plot.show()
    return thresh

def run_preds(header='', model_path='', test_period=False, custom_objects={}):
    test_x = np.load(header + 'total_x_sim_test' + ('_true' if test_period else '') + '_3.npy')
    test_y = np.load(header + 'total_y_sim_test' + ('_true' if test_period else '') + '_3.npy')
    test_p = np.power(10, np.load(header + 'total_params_sim_test' + ('_true' if test_period else '') + '_3.npy')[:, 1])
    model = load_model(header + model_path[:-5] + ('-period' if test_period else '') + '.hdf5', custom_objects=custom_objects)
    model.summary()
    if test_period:
        pred_p = model.predict(test_x, verbose=1)[:, 0]
        predict_period(test_p, pred_p, header)
    else:
        pred_y = model.predict(test_x, verbose=1)
        thresh = predict_transit(test_y, pred_y, header)
        model = load_model(header + model_path[:-5] + '-period.hdf5', custom_objects=custom_objects)
        model.summary()
        pred_p = model.predict(test_x[test_y[:,0] == 1], verbose=1)[:, 0]
        predict_period(test_p[test_y[:,0] == 1], pred_p, header)
        test_x = test_x[pred_y[:, 0] >= thresh]
        test_p = test_p[pred_y[:, 0] >= thresh]
        pred_p = model.predict(test_x, verbose=1)[:, 0]
        predict_period(test_p, pred_p, header)


def p_epsilon_chart(p_test, p_pred):
    percentages = []
    auc_p = 0
    # p_test = np.power(10, p_test)
    # p_pred = np.power(10, p_pred)
    epsilon_range = np.linspace(0, 1, 10000)
    for epsilon in epsilon_range:
        current_correct = p_pred[np.abs(1 - (p_pred / p_test)) < epsilon]
        #print(epsilon, current_correct.shape, float(current_correct.shape[0]) / float(p_pred.shape[0]))
        percentages.append(float(current_correct.shape[0]) / float(p_pred.shape[0]))
        auc_p += percentages[-1] / 10000
    return auc_p, percentages, epsilon_range


