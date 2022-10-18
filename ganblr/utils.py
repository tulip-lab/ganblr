from tensorflow.python.ops import math_ops, array_ops
import numpy as np
import tensorflow as tf

class softmax_weight(tf.keras.constraints.Constraint):
    """Constrains weight tensors to be under softmax `."""
  
    def __init__(self,feature_uniques):
        if isinstance(feature_uniques, np.ndarray):
            idxs = math_ops.cumsum(np.hstack([np.array([0]),feature_uniques]))
        else:
            idxs = math_ops.cumsum([0] + feature_uniques)
        idxs = [i.numpy() for i in idxs]
        self.feature_idxs = [
            (idxs[i],idxs[i+1]) for i in range(len(idxs)-1)
        ]
  
    def __call__(self, w):     
        w_new = [
            math_ops.log(tf.nn.softmax(w[i:j,:], axis=0))
            for i,j in self.feature_idxs
        ]
        return tf.concat(w_new, 0)
  
    def get_config(self):
        return {'feature_idxs': self.feature_idxs}

def elr_loss(KL_LOSS):
  def loss(y_true, y_pred):
    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)+ KL_LOSS
  return loss

def KL_loss(prob_fake):
    return np.mean(-np.log(np.subtract(1,prob_fake)))

def get_lr(input_dim, output_dim, constraint=None,KL_LOSS=0):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(output_dim, input_dim=input_dim, activation='softmax',kernel_constraint=constraint))
    model.compile(loss=elr_loss(KL_LOSS), optimizer='adam', metrics=['accuracy'])
    #log_elr = model.fit(*train_data, validation_data=test_data, batch_size=batch_size,epochs=epochs)
    return model 