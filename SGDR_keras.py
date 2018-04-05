from keras.optimizers import Optimizer
from keras.legacy import interfaces
from keras import backend as K

class SGDR(Optimizer):
	"""
	Stochastic Gradient Descent with Warm Restart optimizer using keras optimizer.
	https://arxiv.org/abs/1608.03983

	Simply modified SGD class in keras/optimizers.py
	https://github.com/keras-team/keras/blob/master/keras/optimizers.py

	# Arguments
		lr: float >= 0. Learning rate.
		momentum: float >= 0. Parameter that accelerates SGD
			in the relevant direction and dampens oscillations.
		decay: float >= 0. Learning rate decay over each update.
		nesterov: boolean. Whether to apply Nesterov momentum.

		iter_per_epoch: iteration per one epoch. Learning rate resets to lr_max after this one epoch.
	"""
		
	def __init__(self, lr=0.01, momentum=0., decay=0., nesterov=False, iter_per_epoch=1, **kwargs):
		super(SGDR, self).__init__(**kwargs)
		with K.name_scope(self.__class__.__name__):
			self.iterations = K.variable(0, dtype='int64', name='iterations') 
			self.lr = K.variable(lr, name='lr')
			self.lr_max = K.variable(lr, name='lr_max')
			self.iter_per_epoch = K.variable(iter_per_epoch, name='iter_per_epoch')
			self.lr_min = K.variable(1e-5, name='lr_min')
			self.momentum = K.variable(momentum, name='momentum') 
			self.decay = K.variable(decay, name='decay')
		self.initial_decay = decay
		self.nesterov = nesterov



	@interfaces.legacy_get_updates_support
	def get_updates(self, loss, params):
		grads = self.get_gradients(loss, params)
		self.updates = [K.update_add(self.iterations, 1)]
		
		
		ipe = self.iter_per_epoch
		total_iter = K.cast(self.iterations, K.dtype(self.lr_min))
		lr_curr = total_iter%(ipe+1)
		lr = self.lr_min + (self.lr_max - self.lr_min) * (1. + K.cos(3.1415 * lr_curr/self.iter_per_epoch)) / 2.
		
		
		
		if self.initial_decay > 0:
			lr *= (1. / (1. + self.decay * K.cast(self.iterations, K.dtype(self.decay))))
		# momentum 
		shapes = [K.int_shape(p) for p in params] 
		moments = [K.zeros(shape) for shape in shapes] 
		self.weights = [self.iterations] + moments 
		for p, g, m in zip(params, grads, moments):
			v = self.momentum * m - lr * g  # velocity
			
			if lr_curr == 0:
				v = -1 * lr * g
			
			self.updates.append(K.update(m, v)) 


			if self.nesterov: 
				new_p = p + self.momentum * v - lr * g 
			else: 
				new_p = p + v 


			# Apply constraints. 
			if getattr(p, 'constraint', None) is not None: 
				new_p = p.constraint(new_p) 

			self.updates.append(K.update(p, new_p))
		return self.updates 


	def get_config(self): 
		config = {'lr': float(K.get_value(self.lr)), 
				  'momentum': float(K.get_value(self.momentum)), 
				  'decay': float(K.get_value(self.decay)), 
				  'nesterov': self.nesterov} 
		base_config = super(SGDR, self).get_config() 
		return dict(list(base_config.items()) + list(config.items()))