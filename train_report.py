### Import Everything needed
from dataStructures import *

import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K

import tensorflow as tf

import random
from collections import deque

#needed for storage
import pickle
import sys
sys.setrecursionlimit(10000)

class ActorCritic:
	#In Init create the actor critic models (need to go through this to make sure its created right)
	def __init__(self, chip, sess, state):
		self.chip = chip
		self.sess = sess
		self.action = np.array([[0, 0]])
		
		self.score = 0
		self.num_macros = chip.get_num_macros()
		
		self.learning_rate = 0.001
		self.epsilon = 1.0
		self.epsilon_decay = .995
		self.gamma = .95
		self.tau   = .125
		
		self.cur_state = state
		
		# ===================================================================== #
		#                               Actor Model                             #
		# Chain rule: find the gradient of chaging the actor network params in  #
		# getting closest to the final value network predictions, i.e. de/dA    #
		# Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
		# ===================================================================== #
		#Don't know what this does
		#Maybe remember how far back it needs to remember
		self.memory = deque(maxlen=5000)
		self.actor_state_input, self.actor_model = self.create_actor_model()
		_, self.target_actor_model = self.create_actor_model()

		#Feed in the number of actions should be 2 columns
		self.actor_critic_grad = tf.placeholder(tf.float32, 
			[None, 2]) # where we will feed de/dC (from critic)
		
		actor_model_weights = self.actor_model.trainable_weights
		#made none negative since I do want it to be the minimum
		self.actor_grads = tf.gradients(self.actor_model.output, 
			actor_model_weights, self.actor_critic_grad) # dC/dA (from actor)
		grads = zip(self.actor_grads, actor_model_weights)
		self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

		# ===================================================================== #
		#                              Critic Model                             #
		# ===================================================================== #		

		self.critic_state_input, self.critic_action_input, \
			self.critic_model = self.create_critic_model()
		_, _, self.target_critic_model = self.create_critic_model()

		self.critic_grads = tf.gradients(self.critic_model.output, 
			self.critic_action_input) # where we calcaulte de/dC for feeding above
		
		# Initialize for later gradient calculations
		self.sess.run(tf.global_variables_initializer())
	
	# ===================================================================== #
	#                        Calcualte Score                                #
	# ===================================================================== #	

	def calc_score(self, buffer_size):
		#Iterate through all rectangles
		macroList = self.chip.get_rectangle_list()
		#reset score before calculating
		self.score = 0
		for macro in macroList:
			#Calculate Distance
			connectionList = macro.get_connect()
			#iterate through all connection centers to get total score
			for connection in connectionList:
				scoreAdd = macro.get_midpoint_dist(connection)
				self.score = self.score + scoreAdd
			
			#Determine if overlap and if there is add 0 points N^2 ouch...
			for macro_compare in macroList:
				#Remove possibility of comparing same macro because duh they overlap
				if (macro.is_intersect(macro_compare, buffer_size)):
					if (macro_compare != macro):
						self.score = self.score + 100000


	def get_score(self):
		return self.score
		
		
		
	# ========================================================================= #
	#                              Model Definitions                            #
	# ========================================================================= #
	#Need to figure out state_input and output DONE And for now is good?
	def create_actor_model(self):
		#Create observation space see webpage for details
		#Currently thinking (x, y) lower left height and width of macro plus all other macros, and chip bounds
		#Do it within the data structure
		#Shape will be (num_rectangles + bounds, 4)
		#where bounds is frist followed by macros each one needs max_x, max_y that x and y can't exceed
		#[[ x, y, width, height, max_x, max_y, totConnection, overlaps, score ], [ x, y, width, height, max_x, max_y, totConnection ...],
		#[ x, y, width, height, max_x, max_y, totConnection ...]... ]
		state_input = Input(shape=(9,))
		#try with high density first trying 4096, going down from there
		h1 = Dense(1024, activation='relu')(state_input)
		h2 = Dense(1024, activation='relu')(h1)
		h3 = Dense(512, activation='relu')(h2)
		#need dense for output which is the action in our case move 10, 100, 500, 1000 units in any direction maybe?
		#Should only be two numbers THIS NEEDS TO CHANGE TO ACCOMDATE ALL THE OUTPUTs?
		output = Dense(2, activation='relu')(h3)
		
		model = Model(input=state_input, output=output)
		adam  = Adam(lr=0.001)
		model.compile(loss="mse", optimizer=adam)
		model.summary()
		return state_input, model
	#Need to figure out state_input and output
	def create_critic_model(self):
		##[[ x, y, width, height, max_x, max_y, totConnection], [ x, y, width, height, max_x, max_y, totConnect],... ] see above for more detail
		state_input = Input(shape=(9,))
		state_h1 = Dense(1024, activation='relu')(state_input)
		state_h2 = Dense(1024)(state_h1)
		
		#8 actions to input so make that shape.. Does this need to be 2 , None?
		action_input = Input(shape=(2,))
		action_h1    = Dense(1024)(action_input)
		
		merged    = Add()([state_h2, action_h1])
		merged_h1 = Dense(512, activation='relu')(merged)
		#This is going to be the score I believe
		output = Dense(1, activation='relu')(merged_h1)
		model  = Model(input=[state_input,action_input], output=output)
		
		adam  = Adam(lr=0.001)
		model.compile(loss="mse", optimizer=adam)
		model.summary()
		return state_input, action_input, model
		
	# ========================================================================= #
	#                               Save Models                                 #
	# ========================================================================= #

	def save_models(self):
		#Save all keras models
		self.critic_model.save('critic_model.h5')
		self.target_critic_model.save('target_critic_model.h5')
		self.target_actor_model.save('target_actor_model.h5')
		self.actor_model.save('actor_model.h5')
		
		#Save the tensorflow model
		saver = tf.train.Saver()
		self.saveMemory()
		saver.save(self.sess, './model/tf_model')
		
	def load_models(self):
		saver = tf.train.Saver()
		saver.restore(self.sess, './model/tf_model')
		self.critic_model = load_model('critic_model.h5')
		self.target_critic_model = load_model('target_critic_model.h5')
		self.target_actor_model = load_model('target_actor_model.h5')
		self.actor_model = load_model('actor_model.h5')
	
		
	# ========================================================================= #
	#                               Model Training                              #
	# ========================================================================= #

	def remember(self, cur_state, action, reward, new_state, done):
		self.memory.append([cur_state, action, reward, new_state, done])

	def _train_actor(self, samples, idx):
		for sample in samples:
			cur_state, action, reward, new_state, _ = sample
			#print(cur_state)
			predicted_action = self.actor_model.predict(cur_state[idx])
			grads = self.sess.run(self.critic_grads, feed_dict={
				self.critic_state_input:  cur_state[idx],
				self.critic_action_input: predicted_action
			})[0]

			self.sess.run(self.optimize, feed_dict={
				self.actor_state_input: cur_state[idx],
				self.actor_critic_grad: grads
			})
            
	def _train_critic(self, samples, idx):
		for sample in samples:
			cur_state, action, reward, new_state, done = sample
			if not done:
				#if (len(new_state) > idx):
				#	idx = len(new_state) - 100
				target_action = self.target_actor_model.predict(new_state[idx])
				#THIS USED TO BE TARGET_ACTION so change it back if you want the standard
				future_reward = self.target_critic_model.predict(
					[new_state[idx], target_action])[0][0]
				#future_reward = random.randint(-1000,1000)
				reward += self.gamma * future_reward
			reward = np.array([reward])
			#print("Reward: ", reward)
			self.critic_model.fit([cur_state[idx], action], reward, verbose=0)
		
	def train(self, idx):
		batch_size = 128
		if len(self.memory) < batch_size:
			return

		rewards = []
		samples = random.sample(self.memory, batch_size)
		self._train_critic(samples, idx)
		self._train_actor(samples, idx)
		
	# ========================================================================= #
	#                         Target Model Updating                             #
	# ========================================================================= #

	def _update_actor_target(self):
		actor_model_weights  = self.actor_model.get_weights()
		actor_target_weights = self.target_critic_model.get_weights()
		
		for i in range(len(actor_target_weights)):
			actor_target_weights[i] = actor_model_weights[i]
		self.target_critic_model.set_weights(actor_target_weights)

	def _update_critic_target(self):
		critic_model_weights  = self.critic_model.get_weights()
		critic_target_weights = self.critic_target_model.get_weights()
		
		for i in range(len(critic_target_weights)):
			critic_target_weights[i] = critic_model_weights[i]
		self.critic_target_model.set_weights(critic_target_weights)		

	def update_target(self):
		self._update_actor_target()
		self._update_critic_target()

	# ========================================================================= #
	#                              Model Predictions                            #
	# ========================================================================= #

	def act(self, cur_state, macro_index):
		#pick a current chip (but maybe infuture move more than one?)
		self.epsilon *= self.epsilon_decay
		if np.random.random() < self.epsilon:
			#create random action
			#Should return something like this
			# [ [14, 45]] where x and y can be positive or negative
			return self.randomAct()
		return self.actor_model.predict(cur_state[macro_index])
		
	def randomAct(self):
		x = random.randint(-1000,1000)
		y = random.randint(-1000,1000)
		return np.array([[x, y]])
		
	def update_state(self, action, index):
		#get curr_stateiterator
		old_score = self.cur_state[index][0][8]
		#change x by action x
		#print("x before: ", self.cur_state[index][0])
		#First is index, second is inside [[ ]] then its the actuall array which is [0] 
		self.cur_state[index][0][0] = self.cur_state[index][0][0] + action[0][0]
		#print("x after: ", self.cur_state[index][0])
		#change y by action y
		self.cur_state[index][0][1] = self.cur_state[index][0][1] + action[0][1]
		#Set x and y in the chip
		self.chip.update_rectangle_list_xy(self.cur_state[index][0][0], self.cur_state[index][0][1], index)
		#Update Overlaps in the chip
		self.chip.update_rectangle_overlaps()
		#Pull the new score
		score = self.chip.rectangle_list[index].calculate_score()
		#print("Score: ", score)
		self.cur_state[index][0][8] = score
		#WNAT TO MINIMIZE (I think check this later)
		score_dif = self.cur_state[index][0][8] - old_score
		return self.cur_state, score_dif
		
	def saveMemory(self):
		pickle.dump( self.memory, open( "save_memory_2000.p", "wb" ) )
		
	def loadMemory(self):
		self.memory = pickle.load( open( "save_memory_2000.p", "rb" ) )
		
		
def createStructure(chip):
	rectangleList = chip.get_rectangle_list()
	envirorment = []
	#[ x, y, width, height, max_x, max_y, totConnection ...]
	for count, rectangle in enumerate(rectangleList):
		#x = rectangle.get_minx()
		#y = rectangle.get_miny()
		x = -10
		y = -10
		width = rectangle.get_width()
		height = rectangle.get_height()
		max_x = chip.get_width() - width
		max_y = chip.get_height() - height
		connectDistance = rectangle.get_total_conn_dist()
		overlaps = rectangle.get_num_overlap()
		#Update rectangle with 0,0 and max_y max_x cord possible cord
		chip.update_rectangle_list_xy(x,y, count)
		chip.update_rectangle_list_x_max_y_max(max_x, max_y, count)
		
		score = rectangle.calculate_score()
		rectValue = np.array([[x, y, width, height, max_x, max_y, connectDistance, overlaps, score]])
		envirorment.append(rectValue)
	return envirorment, chip
	
def get_linear_score(ideal_score, intial_score, current_score):
	return (intial_score - current_score)/(intial_score - ideal_score)
	
def update_xy_max(chip):
	rectangleList = chip.get_rectangle_list()
	chip.update_rectangle_overlaps()
	for count, rectangle in enumerate(rectangleList):
		width = rectangle.get_width()
		height = rectangle.get_height()
		max_x = chip.get_width() - width
		max_y = chip.get_height() - height
		chip.update_rectangle_list_x_max_y_max(max_x, max_y, count)
	
def main():
	#Create Tensor flow session
	sess = tf.Session()
	K.set_session(sess)
	#create chip enviorment
	chip_array = pickle.load( open( "save_chips_10.p", "rb" ) )
	actor_critic_array = []
	#Get array of scores
	ideal_score_array = []
	post_score_array = []
	
	#Need to update the rectanlge list for x_max_y_max before hand
	for iterator, chip in enumerate(chip_array):
		update_xy_max(chip_array[iterator])
	
	#View Plot Before
	plot_name = "test_base_"
	chip_array[0].print_plot()
	plt.savefig(plot_name)
	plt.close()
	#print("Before: ", chip_array[0].get_chip_score())
	for iterator, chip in enumerate(chip_array):
		#Store Ideal Score
		ideal_score_array.append(chip.get_chip_score())
		#Create Envirorment for all chips
		cur_state, chip_array[iterator] = createStructure(chip)
		chip_array[iterator].update_rectangle_overlaps()
		actor_critic = ActorCritic(chip_array[iterator], sess, cur_state)
		actor_critic_array.append(actor_critic)
		
	
	#How many times we should move the macro when
	#iterating through all macros on chip
	move_single_mac = 3
	count = 0
	#while count < 1:
	current_best_linear_score = 0
	read_to_load = False
	for iterator, chip in enumerate(chip_array):
		actor_critic = actor_critic_array[iterator]
		cur_state,_ = createStructure(chip)
		if (read_to_load):
			actor_critic.load_models()
		#cur_state = cur_state.reshape((1, env.shape[0]))
		#print(cur_state)
		#print(env.observation_space.shape)
		#action = action.reshape((1, env.action_space.shape[0]))
		#Take selected Action here
		#Iterate through all macros to move initially
		initial_score = actor_critic.chip.get_chip_score()
		for idx, macro in enumerate(chip.get_rectangle_list()):
			for x in range(0, move_single_mac):
				action = actor_critic.act(cur_state, idx)
				#Need to mimic this below
				score_change = 0
				#new_state, reward, done, _ = env.step(action)
				new_state, score_change = actor_critic.update_state(action, idx)
				reward = score_change
				done = 0
				#new_state = new_state.reshape((1, env.observation_space.shape[0]))
				#print("Action: ", action)
				#print("Reward: ", reward)
				actor_critic.remember(cur_state, action, reward, new_state, done)
				actor_critic.train(idx)
			
				#need to seperate but good start
				cur_state = new_state
				
			#for testing for now pick one item with highest score then once it doesn't improve after 
			#50 moves move on OR DO THIS EVERY 100 MOVES NEED TO FIGURE OUT
			curMaxScore = 0
			curMaxIter = 0
			
		#Iterate through the worst 100 to try to improve the score
		number =0
		while (number < 100):
			curMaxScore = 0
			for idx, rectangle in enumerate(cur_state):
				if rectangle[0][8] > curMaxScore:
					curMaxScore = rectangle[0][8]
					curMaxIter = idx
			
			#Change worst one
			action = actor_critic.act(cur_state, idx)
			#Need to mimic this below
			score_change = 0
			#new_state, reward, done, _ = env.step(action)
			new_state, score_change = actor_critic.update_state(action, idx)
			reward = score_change
			done = 0
			actor_critic.remember(cur_state, action, reward, new_state, done)
			actor_critic.train(idx)
			number = number + 1
	
		#need to seperate but good start
		cur_state = new_state
		#print(count)
		
		#Update the actor critic session and chip
		actor_critic_array[iterator] = actor_critic
		chip_array[iterator] = chip
		
		#get linear score chip_array[iterator].get_chip_score() = after trained score
		#ideal_score_array[iterator] = ideal score of original data
		#
		linear_score = get_linear_score(ideal_score_array[iterator], initial_score, chip_array[iterator].get_chip_score())
		if (linear_score > current_best_linear_score):
			count = count + 1
			read_to_load = True
			plot_name = "new_best_" + str(count)
			actor_critic.chip.print_plot()
			plt.savefig(plot_name)
			plt.close()
			current_best_linear_score = linear_score
			#Figure Out how to save the models here..
			actor_critic.save_models()


	actor_critic.saveMemory()
	actor_critic.chip.print_plot()
	

if __name__ == "__main__":
	main()
	