"""rcbo Optimizer written for Keras"""
from keras.optimizers import Optimizer
from keras import backend as K
import numpy as np
if K.backend() == 'tensorflow':
    import tensorflow as tf
class rcbo(Optimizer):
    def __init__(self, alpha=100, **kwargs):
        super(rcbo, self).__init__(**kwargs)
        self._alpha = alpha
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')

    def get_updates(self, params, loss, contraints=None):
        self.updates = [K.update_add(self.iterations, 1)]
        grads = self.get_gradients(loss, params)
        shapes = [K.int_shape(p) for p in params]
        L = [K.variable(np.full(fill_value=1e-8, shape=shape)) for shape in shapes]
        reward = [K.zeros(shape) for shape in shapes]
        tilde_w = [K.zeros(shape) for shape in shapes]
        gradients_sum = [K.zeros(shape) for shape in shapes]
        gradients_norm_sum = [K.zeros(shape) for shape in shapes]
        for p, g, li, ri, twi, gsi, gns in zip(params, grads, L, reward, tilde_w,gradients_sum, gradients_norm_sum):
            grad_sum_update = gsi + g
            grad_norm_sum_update = gns + K.abs(g)
            l_update = K.maximum(li, K.abs(g))
            reward_update = K.maximum(ri - g * twi, 0)
            new_w = - grad_sum_update / (l_update * (K.maximum(grad_norm_sum_update + l_update, self._alpha * l_update))) * (reward_update + l_update)
            #----------new formula--------------
            s=1e-6
            Xt=grad_sum_update
            r1=np.random.random()+1e-5
            c=np.random.random()+1e-6
            fl=0.5
            dt=l_update
            udt=grad_norm_sum_update
            udt1=reward_update
            # Xij=((1-la*dt-udt+udt1)/(1-la*dt-udt+udt1-delta*beta))*(Xt*(1-beta)-((Xt*la*dt*beta)/(1-la*dt-udt+udt1)))
            import math
            X_alp=udt
            X_bet=udt1
            try:
                Xij = (udt+ (-0.05 + 0.1 * r1) * (udt1 - r1 * udt))
                new_w=new_w+s*Xij
            except:
                new_w = new_w
            #--------------------------------------

            param_update = p - twi + new_w
            tilde_w_update = new_w
            self.updates.append(K.update(gsi, grad_sum_update))
            self.updates.append(K.update(gns, grad_norm_sum_update))
            self.updates.append(K.update(li, l_update))
            self.updates.append(K.update(ri, reward_update))
            self.updates.append(K.update(p, param_update))
            self.updates.append(K.update(twi, tilde_w_update))
        return self.updates

    def get_config(self):
        config = {'alpha': float(K.get_value(self._alpha)) }
        base_config = super(rcbo, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




