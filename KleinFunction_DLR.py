#!/usr/bin/env python
import numpy as np
import random
#@np.vectorize

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def rectifier(x):
    return np.where(x>0., x, 0.)
def leaky(x):
    return np.where(x>0., x, 0.01*x)

from scipy.stats import truncnorm
def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd,
                     (upp - mean) / sd,
                     loc=mean,
                     scale=sd)

def transform_labels_into_one_hot(labels, no_of_different_labels):
    lr = np.arange(no_of_different_labels)
    # transform labels into one hot representation
    labels_one_hot = (lr==labels).astype(np.float)
    # we don't want zeroes and ones in the labels neither:
    labels_one_hot[labels_one_hot==0] = 0.01
    labels_one_hot[labels_one_hot==1] = 0.99
    return labels_one_hot
    
class NeuralNetwork:

        def __init__(self,
                     no_of_in_nodes,
                     no_of_out_nodes,
                     no_of_hidden_nodes,
                     learning_rate,
                     data_array,
                     labels,
                     no_of_different_labels,
                     accuracy=0.96,
                     epochs=20,
                     bias=None,
                     mean_square_error_cost=True,
                     activation_function=sigmoid,
                     intermediate_results=False,
                     data_array_for_testing=None,
                     labels_for_testing=None
                     ):
            self.no_of_in_nodes = no_of_in_nodes
            self.no_of_out_nodes = no_of_out_nodes
            self.no_of_hidden_nodes = no_of_hidden_nodes
            
            self.learning_rate = learning_rate
            self.data_array = data_array
            self.labels = labels
            self.no_of_different_labels = no_of_different_labels
            self.accuracy = accuracy
            self.epochs = epochs
            self.mean_square_error_cost = mean_square_error_cost # True: mean square error; False: cross entropy error
            self.activation_function = activation_function # rectifier and leaky only available for mean square error cost
            self.intermediate_results = intermediate_results
            
            self.learning_rate_bias = None
            
            self.create_weight_matrices()
            self.labels_one_hot = transform_labels_into_one_hot(labels, no_of_different_labels)
            self.order_data_array = range(len(self.data_array))

            self.nesterov = False
            
            self.adamQ = False
            self.alpha = 0.001    # adam parameters
            self.beta_1 = 0.9
            self.beta_2 = 0.999
            self.epsilon = 1e-8
            self.t = 0            # adam running variables
            self.m_t = 0
            self.v_t = 0
            self.m2_t = 0
            self.v2_t = 0

            if(data_array_for_testing is None):
                self.data_array_for_testing = data_array
                self.labels_for_testing = self.labels_one_hot
            else:
                self.data_array_for_testing = data_array_for_testing
                self.labels_for_testing = transform_labels_into_one_hot(labels_for_testing, no_of_different_labels)

        def map_algorithm(self,algorithm,momentum_para):
            if(algorithm == "AlgoStandard"):
                momentum_para = 0.
                return self.AlgoStandard
            elif(algorithm == "AlgoMomentum"):
                return self.AlgoMomentum
            elif(algorithm == "AlgoNesterov"): # Nesterov's Momentum
                self.nesterov = True
                return self.AlgoMomentum            
            elif(algorithm == "AlgoAdamQ"):
                self.adamQ = True # setting the Adam flag
                return self.AlgoStandard


        def create_weight_matrices(self):
            """ 
            A method to initialize the weight matrices 
            of the neural network with optional 
            bias nodes"""

            self.wih_original = np.random.normal(0.,0.01,
                                            (self.no_of_hidden_nodes,
                                             self.no_of_in_nodes))
            self.wih = np.copy(self.wih_original)
            self.dWih = np.zeros(shape=(self.no_of_hidden_nodes,
                                        self.no_of_in_nodes))
            self.dWih_previous = np.zeros(shape=(self.no_of_hidden_nodes,
                                        self.no_of_in_nodes))
            self.learning_rate_wih = self.learning_rate * np.ones(shape=(self.no_of_hidden_nodes, self.no_of_in_nodes))
                
            self.who_original = np.random.normal(0.,0.01,
                                        (self.no_of_out_nodes,
                                        self.no_of_hidden_nodes))
            self.who = np.copy(self.who_original)
            self.dWho = np.zeros(shape=(self.no_of_out_nodes,
                                        self.no_of_hidden_nodes))
            self.dWho_previous = np.zeros(shape=(self.no_of_out_nodes,
                                        self.no_of_hidden_nodes))
            self.learning_rate_who = self.learning_rate * np.ones(shape=(self.no_of_out_nodes, self.no_of_hidden_nodes))

        def learning_rate_rescaling(self):
            # ScaleWithPre, summing over all the post-synaptic weights of a pre-synaptic neuron
            self.learning_rate_wih = self.learning_rate * ( np.fabs(self.wih+self.dWih) + self.learning_rate_bias ) / np.tile(np.linalg.norm(self.wih+self.dWih, axis=0) + self.learning_rate_bias, (self.no_of_hidden_nodes,1))
            self.learning_rate_who = self.learning_rate * ( np.fabs(self.who+self.dWho) + self.learning_rate_bias ) / np.tile(np.linalg.norm(self.who+self.dWho, axis=0) + self.learning_rate_bias, (self.no_of_out_nodes,1))

            # ScaleWithPost, summing over all the pre-synaptic weights of a post-synaptic neuron
            # self.learning_rate_wih = self.learning_rate * ( np.fabs(self.wih+self.dWih) + self.learning_rate_bias ) / np.tile(np.linalg.norm(self.wih+self.dWih, axis=1) + self.learning_rate_bias, (self.no_of_in_nodes,1)).transpose()
            # self.learning_rate_who = self.learning_rate * ( np.fabs(self.who+self.dWho) + self.learning_rate_bias ) / np.tile(np.linalg.norm(self.who+self.dWho, axis=1) + self.learning_rate_bias, (self.no_of_hidden_nodes,1)).transpose()
                
        def regenerate_weight_matrices(self):
            self.wih = np.copy(self.wih_original)
            self.dWih = np.zeros(shape=(self.no_of_hidden_nodes,
                                        self.no_of_in_nodes))
            self.dWih_previous = np.zeros(shape=(self.no_of_hidden_nodes,
                                                 self.no_of_in_nodes))
            self.learning_rate_wih = self.learning_rate * np.ones(shape=(self.no_of_hidden_nodes, self.no_of_in_nodes))
            self.who = np.copy(self.who_original)
            self.dWho = np.zeros(shape=(self.no_of_out_nodes,
                                        self.no_of_hidden_nodes))
            self.dWho_previous = np.zeros(shape=(self.no_of_out_nodes,
                                        self.no_of_hidden_nodes))
            self.learning_rate_who = self.learning_rate * np.ones(shape=(self.no_of_out_nodes, self.no_of_hidden_nodes))            

                
        def AlgoStandard(self, threshold):
                self.who += self.dWho
                ddWho = np.copy(self.dWho)
                self.dWho.fill(0.)
                self.wih += self.dWih
                ddWih = np.copy(self.dWih)
                self.dWih.fill(0.)

        def AlgoMomentum(self, threshold):
                self.who += self.dWho
                self.dWho_previous = np.copy(self.dWho)
                self.dWho.fill(0.)
                self.wih += self.dWih
                self.dWih_previous = np.copy(self.dWih)
                self.dWih.fill(0.)
                
        def train_single(self, input_vector, target_vector,
                         algorithm, threshold, momentum_para):
                """
                input_vector and target_vector can be tuple, 
                list or ndarray
                """
                input_vector = np.array(input_vector, ndmin=2).T

                input_hidden = np.dot(self.wih+self.dWih,input_vector)
                output_hidden = self.activation_function(input_hidden)

                input_last_layer = np.dot(self.who+self.dWho, output_hidden)
                output_last_layer = self.activation_function(input_last_layer)
                
                # actually E=(target-output)^2, so outputerror = -dE/dy
                output_errors = np.array(target_vector, ndmin=2).T - output_last_layer
                # calculate hidden errors:
                hidden_errors = np.dot(self.who.T+self.dWho.T, output_errors)
                
                if self.mean_square_error_cost:
                    if(self.activation_function==sigmoid):
                        # find update to hid->out weights:
                        tmp = output_errors * output_last_layer * (1.0 - output_last_layer)
                        # find update to the in -> hid weights:
                        tmp2 = hidden_errors * output_hidden * (1.0 - output_hidden)
                    elif(self.activation_function==leaky):
                        tmp = np.where(input_last_layer>0.,output_errors,0.01*output_errors)
                        tmp2 = np.where(input_hidden>0.,hidden_errors,0.01*hidden_errors)
                    else: # linear rectifier
                        tmp = np.where(input_last_layer>0.,output_errors,0.)
                        tmp2 = np.where(input_hidden>0.,hidden_errors,0.)
                else:
                    tmp = output_errors
                    tmp2 = hidden_errors * output_hidden
    
                y = np.dot(tmp, output_hidden.T)
                x = np.dot(tmp2, input_vector.T)

                self.t += 1 # iter counter, needed for Adam

                if self.adamQ:
                    g_t = -y
                    self.m_t = self.beta_1*self.m_t + (1-self.beta_1)*g_t
                    self.v_t = self.beta_2*self.v_t + (1-self.beta_2)*(g_t*g_t)
                    m_cap = self.m_t/(1-(self.beta_1**self.t))
                    v_cap = self.v_t/(1-(self.beta_2**self.t))
                    self.dWho += -(self.alpha*m_cap)/(np.sqrt(v_cap)+self.epsilon)

                    # note, separate memory vars
                    g_t = -x
                    self.m2_t = self.beta_1*self.m2_t + (1-self.beta_1)*g_t
                    self.v2_t = self.beta_2*self.v2_t + (1-self.beta_2)*(g_t*g_t)
                    m2_cap = self.m2_t/(1-(self.beta_1**self.t))
                    v2_cap = self.v2_t/(1-(self.beta_2**self.t))
                    self.dWih += -(self.alpha*m2_cap)/(np.sqrt(v2_cap)+self.epsilon)
                elif(self.nesterov):
                    tmp = self.learning_rate_who * y + momentum_para * self.dWho_previous
                    tmp2 = self.learning_rate_wih * x + momentum_para * self.dWih_previous
                    self.dWho += -momentum_para*self.dWho_previous + (1+momentum_para)*tmp
                    self.dWih += -momentum_para*self.dWih_previous + (1+momentum_para)*tmp2
                else:
                    self.dWho += self.learning_rate_who * y + momentum_para * self.dWho_previous
                    self.dWih += self.learning_rate_wih * x + momentum_para * self.dWih_previous
                
                algorithm(threshold) 

        
        def train(self, algorithm, accuracy_buffer,
                  threshold=0., momentum_para=0., randomise_patterns=False):
            # accuracy_buffer: when to stop the code if accuracy not improving
            intermediate_wih = []
            intermediate_who = []
            accuracy = []
            algorithm = self.map_algorithm(algorithm,momentum_para)
            
            self.regenerate_weight_matrices()
            for epoch in range(self.epochs):
                corrects, wrongs = 0, 0
                if(randomise_patterns):
                    self.order_data_array = list(range(len(self.data_array)))
                    random.shuffle(self.order_data_array)
                for i in range(len(self.data_array)):
                    self.train_single(self.data_array[self.order_data_array[i]],
                                      self.labels_one_hot[self.order_data_array[i]],
                                      algorithm, threshold,
                                      momentum_para)
                    if(self.learning_rate_bias is not None):
                        self.learning_rate_rescaling()                    
                if self.intermediate_results:
                    intermediate_wih.append(self.wih+self.dWih)
                    intermediate_who.append(self.who+self.dWho)
                for i in range(len(self.data_array_for_testing)):
                    res = self.run(self.data_array_for_testing[i])
                    res_max = res.argmax()
                    if res_max == self.labels_for_testing[i].argmax():
                        corrects += 1
                    else:
                        wrongs += 1
                accuracy.append(1.0*corrects / ( corrects + wrongs))
                print("accuracy: ", accuracy[epoch])
                if(accuracy[epoch] > self.accuracy):
                    break
                if( (len(accuracy)>(accuracy_buffer+1)) and (accuracy[epoch] <= min(accuracy[-(accuracy_buffer+1):-1])) ):
                    break
                    
            if(self.nesterov is True):
                self.nesterov = False

            if(self.adamQ is True): # remove the Adam flag after running Adam
                self.adamQ = False

            return intermediate_wih, intermediate_who, accuracy

        
        def train_more_save(self, algorithm, accuracy_buffer, save_frequency,
                            threshold=0., momentum_para=0.,
                            randomise_patterns=False):
            # accuracy_buffer: when to stop the code if accuracy not improving
            # save_frequency: how often accuracy are measured
            intermediate_wih = []
            intermediate_who = []
            accuracy = []
            accuracy_more_save = []
            algorithm = self.map_algorithm(algorithm,momentum_para)
            
            self.regenerate_weight_matrices()
            bool_BreakLoop = False
            for epoch in range(self.epochs):
                if(randomise_patterns):
                    self.order_data_array = list(range(len(self.data_array)))
                    random.shuffle(self.order_data_array)
                for i in range(len(self.data_array)):
                    self.train_single(self.data_array[self.order_data_array[i]],
                                      self.labels_one_hot[self.order_data_array[i]],
                                      algorithm, threshold,
                                      momentum_para)
                    if(self.learning_rate_bias is not None):
                        self.learning_rate_rescaling()                    
                    if(((i+1)%save_frequency)==0):
                        intermediate_wih.append(self.wih+self.dWih)
                        intermediate_who.append(self.who+self.dWho)
                        corrects, wrongs = self.evaluate(self.data_array_for_testing,self.labels_for_testing)
                        accuracy_more_save.append(1.0*corrects / ( corrects + wrongs))
                        #print("accuracy: ", accuracy_more_save[-1])
                        if(accuracy_more_save[-1] > self.accuracy):
                            print("accuracy: ", accuracy_more_save[-1])
                            bool_BreakLoop = True
                            break
                accuracy.append(accuracy_more_save[-1])
                print("accuracy: ", accuracy[epoch])
                if(bool_BreakLoop):
                    break
                if( (len(accuracy)>(accuracy_buffer+1)) and (accuracy[epoch] <= min(accuracy[-(accuracy_buffer+1):-1])) ):
                    break

            if(self.nesterov is True):
                self.nesterov = False
                
            if(self.adamQ is True): # remove the Adam flag after running Adam
                self.adamQ = False
                
            return intermediate_wih, intermediate_who, accuracy_more_save

        
        def run(self, input_vector):
            # input_vector can be tuple, list or ndarray
            input_vector = np.array(input_vector, ndmin=2).T
            output_vector = np.dot(self.wih+self.dWih,
                                   input_vector)
            output_vector = self.activation_function(output_vector)
            
            output_vector = np.dot(self.who+self.dWho,
                                   output_vector)
            output_vector = self.activation_function(output_vector)
                        
            return output_vector
        

        def evaluate(self, data, labels):
            corrects, wrongs = 0, 0
            for i in range(len(data)):
                res = self.run(data[i])
                res_max = res.argmax()
                if res_max == labels[i].argmax():
                    corrects += 1
                else:
                    wrongs += 1
            return corrects, wrongs
