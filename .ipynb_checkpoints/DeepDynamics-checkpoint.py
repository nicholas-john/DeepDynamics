"""
@author: Nicholas John

Machine learning of ordinary differential equations with artificial neural networks.

Implements signal-noise decomposition as proposed in
"Deep learning of dynamics and signal-noise decomposition
with time-stepping constraints" 
Samuel H. Rudy, J. Nathan Kutz, Steven L. Brunton
Journal of Computational Physics 396 (2019) 483-506

Optional included loss terms inspired by the method of Regularized Numerical Differentiation devised by J. Cullum:
"Numerical Differentiation and Regularization"
SIAM Journal of Numerical Analysis
Vol. 8 No. 2, June 1971

Also includes optional loss terms for "Neural ODE" [Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud, 2019], most of which is implemented in a TensorFlow library.  

Incorporates code by Maziar Raissi:
@article{raissi2018multistep,
  title={Multistep Neural Networks for Data-driven Discovery of Nonlinear Dynamical Systems},
  author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
  journal={arXiv preprint arXiv:1801.01236},
  year={2018}
}
https://github.com/maziarraissi/MultistepNNs

An early example of artificial neural networks as differential equations:
"Continuous time nonlinear signal processing:
A neural network based approach for gray box
identification"
R. Rico-Martinez, J. S. Anderson and I. G. Kevrekidis
IEEE 1994
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import nodepy.linear_multistep_method as lm
import timeit
from icecream import ic
import sys

np.random.seed(1234)
tf.random.set_seed(1234)


# ---------------------
# Notes:
#
# You will have to decide on an initializer
#
# Keras has a constraints module that we should try to use when the time arrives
# Okay actually constraints are for the weights
# We can probably use Lagrange multipliers in the loss function
# for constraints
# such as (in SIS) i_prime + s_prime = 0
# and l_II_prime + l_SS_prime + l_SI_prime = 0
#
# It may also be possible to use a constrained least squares 
# procedure for the output layer
#
# ---------------------

class DeepDynamics:
    def __init__(self, dt, X, layers, 
                M=4, 
                scheme='AB',
                reg_param=1e-2,
                noise_reg_param=1e-2,
                weight_reg_param=1e-2,
                smoothening_param=0,
                optimizer=keras.optimizers.Adam(),
                activation='elu',
                rk_steps=1):
        
        self.dt = dt
        self.reg_param = reg_param
        self.noise_reg_param = noise_reg_param
        self.weight_reg_param = weight_reg_param
        self.smoothening_param = smoothening_param
        self.rk_steps = rk_steps
        self.optimizer = optimizer
        self.activation = activation
        self.X = X # list of S x N x D tensors
        
        #self.FD = self.make_FD(X[0][0], dt)
        
        self.S = X[0].shape[0] # number of trajectories
        self.N = X[0].shape[1] # number of time snapshots
        self.D = X[0].shape[2] # number of dimensions (including parameters)
        # ------------------------
        
        self.M = M # number of Adams-Moulton steps
        
        self.trange = np.linspace(0, self.N * self.dt, self.N) # used for array slicing in Neural ODE loss
        
        self.layers = layers
        
        self.D_out = layers[-1]
        
        self.w_hat = tf.Variable( tf.zeros( self.X[0][...,:self.D_out].shape, dtype=tf.float32 ), trainable=True )
        #mask = np.ones(self.w_hat.shape, dtype=np.float32)
        #mask[:,0,:] = 0
        #mask[:,-1:] = 0
        #self.w_mask = tf.constant(mask)
        
        # a place to put noise estimates between batches
        self.w_storage = [tf.zeros(x[...,:self.D_out].shape, dtype=tf.float32) for x in self.X]
        
        switch = {'AM': lm.Adams_Moulton,
                  'AB': lm.Adams_Bashforth,
                  'BDF': lm.backward_difference_formula}
        method = switch[scheme](M)
        self.alpha = np.float32(-method.alpha[::-1])
        self.beta = np.float32(method.beta[::-1])
        
        self.gamma_reg = 0
        self.beta_reg = 0
        
        self.model = self.build_model()
        
    def update_X(self, X):
        """
        Consider calling this function in the constructor
        """
        self.X = X # list of S x N x D tensors
        
        self.S = X[0].shape[0] # number of trajectories
        self.N = X[0].shape[1] # number of time snapshots
        self.D = X[0].shape[2] # number of dimensions (including parameters)
        
        self.trange = np.linspace(0, self.N * self.dt, self.N) # used for array slicing in Neural ODE loss
        self.w_hat = tf.Variable( tf.zeros( self.X[0][...,:self.D_out].shape, dtype=tf.float32 ), trainable=True )
        mask = np.ones(self.w_hat.shape, dtype=np.float32)
        mask[:,0,:] = 0
        mask[:,-1:] = 0
        self.w_mask = tf.constant(mask)
        
        # a place to put noise estimates between batches
        self.w_storage = [tf.zeros(x[...,:self.D_out].shape, dtype=tf.float32) for x in self.X]
        
        
    def build_model(self):
        # ----------------------
        #
        # This is the place where you can add 
        # regularization in the form of, e.g.,
        # dropout layers or L_1 / L_2 regularization
        #
        # ----------------------
        model = keras.Sequential()
        model.add(keras.layers.Dense(units=self.layers[1], 
                                    activation=self.activation,
                                    input_shape=(self.layers[0],)
                                    #kernel_regularizer=keras.regularizers.l1
                                    )
                 )
        
        if len(self.layers) > 3:
            for index in range( 2, len(self.layers) - 1):
                model.add(keras.layers.Dense(units=self.layers[index],
                                             activation=self.activation))
        
        # can you make this last layer a constrained least-squares regression?
        model.add(keras.layers.Dense(units=self.layers[-1], 
                                    activation=None))
        
        return model
    
    @tf.function
    def net_F(self, X): # S x (N-M+1) x D
        X_reshaped = tf.reshape(X, [-1,self.D]) # S(N-M+1) x D
        F_reshaped = self.model(X_reshaped) # S(N-M+1) x D_out
        F = tf.reshape(F_reshaped, [self.S,-1,self.D_out]) # S x (N-M+1) x D_out
        return F # S x (N-M+1) x D_out
    
    @tf.function
    def net_Y(self, X): # S x N x D
        
        M = self.M
        
        #Y = self.alpha[0]*X[:,M:,:self.D_out] + self.dt*self.beta[0]*self.net_F(X[:,M:,:])
        al = self.alpha[0]*X[:,M:,:self.D_out]
        be = self.dt*self.beta[0]*self.net_F(X[:,M:,:])
        Y = tf.math.add( al , tf.cast(be, al.dtype) )
        for m in range(1, M+1):
            al = Y + self.alpha[m]*X[:,M-m:-m,:self.D_out]
            be = self.dt*self.beta[m]*self.net_F(X[:,M-m:-m,:])
            Y = tf.math.add( al , tf.cast(be, al.dtype) )
        return Y # S x (N-M+1) x D
    
    @tf.function
    def rk_eval(self, y):
        dynamic = self.net_F(y)
        constant = y[...,self.D_out:]
        return tf.concat( [dynamic, constant], -1 )
    
    @tf.function
    def ode_fn(self, t, y):
        
        dynamic = self.model(y)
        constant = 0 * y[...,self.D_out:]
        return tf.concat( [dynamic, constant], -1 )
    
    @tf.function
    def compute_loss(self, X_sample):
        
        X = tf.cast(X_sample, tf.float32)
        X = X - tf.concat( [self.w_hat, tf.zeros(self.X[0][...,self.D_out:].shape, dtype=tf.float32)], -1 )
        
        loss_denoising = self.noise_reg_param * tf.math.reduce_sum( tf.math.square( self.w_hat ) )
        #loss_denoising = tf.cast( loss_denoising, tf.float32 )
        
        loss_weight_reg = 0
        for w in self.model.trainable_weights[0::2]:
            loss_weight_reg += tf.math.reduce_sum( tf.math.square( w ) )
        loss_weight_reg = self.weight_reg_param * self.S * self.D_out * loss_weight_reg
        
        # encouraging the noise estimate to yield a smooth curve
        loss_smoothness = self.smoothening_param * tf.math.reduce_sum(
            tf.math.square( (X[:,2:,:] - X[:,:-2,:]) / (2 * self.dt) )
        )
        
        """
        # Do you have a conserved quantity? Consider adding and modifying this term.
        f = self.net_F(X)
        loss_cons = 1000*tf.math.reduce_sum( 
            tf.math.square( f[:,:,0] + f[:,:,1] )
            + tf.math.square( f[:,:,2] + f[:,:,3] + f[:,:,4] )
        )
        """
        
        """
         # Multistep Neural Network
        Y = self.net_Y(X)
        loss_MNN = tf.math.reduce_sum(tf.math.square(Y), axis=1)
        loss_MNN = tf.math.sqrt(loss_MNN)
        loss_MNN = tf.math.reduce_sum(loss_MNN)
        #loss_MNN = tf.cast(loss_MNN, tf.float32)
        """
        """
        # Neural ODE
        integ = tfp.math.ode.DormandPrince().solve(
            ode_fn=self.ode_fn,
            initial_time=0,
            initial_state=X[:,0,:],
            #solution_times=np.linspace(0, self.N * self.dt, 10),
            #solution_times = self.trange[::400],
            solution_times=[0, self.N * self.dt],
            batch_ndims=0
        )
        y = tf.concat( [tf.expand_dims(X[:,0,:self.D_out], 1) , tf.expand_dims(X[:,-1,:self.D_out], 1)], 1 )
        #y = tf.cast( X[:,::400,:self.D_out] , tf.float32 )
        yhat = tf.transpose( integ.states[:][...,:self.D_out] , [1,0,2] )
        loss_MSE = (self.N) * tf.math.reduce_sum( tf.math.square( y - yhat ) )
        """
        
        """
        last = integ.states[-1][...,:self.D_out]
        true_last = tf.cast( X[:,-1,:self.D_out], tf.float32 )
        loss_final = self.N * tf.math.reduce_sum(tf.math.square( last - true_last )) 
        """
        
        # Runge-Kutta (4)
        h = self.dt
        y = [0] * (self.rk_steps+1)
        y[0] = tf.cast( X[:,:-self.rk_steps,:], tf.float32 ) 
        for i in range(1, self.rk_steps+1):
            k1 = self.rk_eval(y[i-1]) 
            k2 = self.rk_eval(y[i-1] + h * k1/2)
            k3 = self.rk_eval(y[i-1] + h * k2/2)
            k4 = self.rk_eval(y[i-1] + h * k3)
            y[i] = y[i-1] + (1/6) * h * (k1 + 2*k2 + 2*k3 + k4)
        loss_RK = 0
        for i in range(self.rk_steps-1):
            loss_RK += tf.math.reduce_sum(
                tf.math.square( y[i+1][...,:self.D_out] - X[:,i+1:-self.rk_steps+i+1,:self.D_out] )
            )
        loss_RK += tf.math.reduce_sum(
            tf.math.square( y[self.rk_steps][...,:self.D_out] - X[:,self.rk_steps:,:self.D_out] )
        )
        loss_RK = (1/self.rk_steps) * loss_RK
        
        """
        # Stabilized Fitting?
        #hs = self.stable_steps
        stable_steps = int(self.N / 10)
        ys = [0] * stable_steps
        hs = self.dt * 10
        ys[0] = tf.cast( X[:,0:1,:], tf.float32 )
        for i in range(1, stable_steps):
            k1 = self.rk_eval(ys[i-1]) 
            k2 = self.rk_eval(ys[i-1] + hs * k1/2)
            k3 = self.rk_eval(ys[i-1] + hs * k2/2)
            k4 = self.rk_eval(ys[i-1] + hs * k3)
            ys[i] = ys[i-1] + (1/6) * hs * (k1 + 2*k2 + 2*k3 + k4)
        ys = tf.convert_to_tensor(ys)
        loss_stab = 10 * tf.math.reduce_sum( tf.math.square( ys[...,:self.D_out] - tf.cast(X[:,::10,:self.D_out], tf.float32) ) ) 
        """
        
        # Regularization: \alpha ( ||f||^2 + ||f'||^2 ) 
        # consider using self.net_F(...) for code reuse
        #X_reshape = tf.reshape(X, [-1,self.D])
        #f = self.model(X_reshape)
        #f = tf.reshape(f, [-1,self.N,self.D_out])
        """
        f = self.net_F(X)
        fprime = (f[:,2:,:] - f[:,:-2,:]) / (2 * self.dt)
        loss_reg = self.reg_param * (
            tf.math.reduce_sum(tf.math.square(f[:,1:-1,:]))
            + tf.math.reduce_sum(tf.math.square(fprime))
        )
        """
        
        """
        # Total variation regularization
        f = self.net_F(X)
        variation = f[:,2:,:] - f[:,:-2,:]
        loss_TV = self.reg_param * tf.math.reduce_sum(
            tf.math.abs( variation )
        )
        """
        
        """
        # LASSO
        f = self.net_F(X)
        loss_LASSO = self.reg_param * tf.math.reduce_sum( tf.math.abs( f ) )
        """
        
        loss_total =  (1/self.S) * (1/self.D_out) * (1/self.N) * (loss_RK + loss_denoising + loss_weight_reg + loss_smoothness)
        return loss_total
    
    #@tf.function
    def train_step(self, sample_index, iters):
        X = self.X[sample_index]
        #self.w_hat.assign( self.w_storage[sample_index] * self.w_mask )
        self.w_hat.assign( self.w_storage[sample_index] )
        #print(self.w_hat)
        for index in range(iters):
            with tf.GradientTape() as tape:
                loss_value = self.compute_loss(X)
            coeffs = [self.model.trainable_weights, self.w_hat]
            [grad_model, grad_w] = tape.gradient(loss_value, coeffs)
            self.optimizer.apply_gradients(zip(grad_model, self.model.trainable_weights))
            self.optimizer.apply_gradients(zip([grad_w], [self.w_hat]))
            #self.w_hat.assign(self.w_hat * self.w_mask)
        #self.w_storage[sample_index] = tf.constant(self.w_hat * self.w_mask)
        self.w_storage[sample_index] = tf.constant(self.w_hat)
        return loss_value
    
    def train(self, N_epochs, iters, rand_sample=False):
        """
        iters is the number of iterations on each batch at a time
        """
        n_batches = len(self.X)
        start_time = timeit.default_timer()
        loss_value = 0
        for batch in range(n_batches):
            loss_value += self.train_step(batch, iters)
        loss_value /= n_batches
        elapsed = timeit.default_timer() - start_time
        print('Epoch: %d, Loss: %.3e, Time: %.2f' % 
                (1, loss_value, elapsed))
        start_time = timeit.default_timer()
        for epoch in range(2, N_epochs+1):
            loss_value = 0
            for batch in range(n_batches):
                sample = batch
                if rand_sample:
                    X_len = len(self.X)
                    sample = np.random.randint(low=0, high=X_len)
                loss_value = self.train_step(sample, iters)
            loss_value /= n_batches
            # print things out every once in a while
            #if True:
            if epoch % (N_epochs//10) == 0:
                elapsed = timeit.default_timer() - start_time
                print('Epoch: %d, Loss: %.3e, Time: %.2f' % 
                      (epoch, loss_value, elapsed))
                #start_time = timeit.default_timer()

    def predict(self, X):
        f = self.model.predict(X[None,:])
        return f.flatten()