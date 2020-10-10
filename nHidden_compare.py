#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import random
import KleinFunction_DLR as Klein


image_size = 28
no_of_different_labels = 10
image_pixels = image_size*image_size
usebinQ = True
if(usebinQ is False):
    #train_data = np.loadtxt("../../../mnist_train.csv",delimiter=",")
    train_data = np.loadtxt("../../../mnist_train_100.csv",delimiter=",")
    fac = 255 # normalising data values to [0., 1.]
    train_imgs = np.asfarray(train_data[:, 1:]) / fac
    train_labels = np.asfarray(train_data[:, :1])
else:
    train_imgs = np.fromfile("../../../../mnist_train_imgs_binary.dat").reshape((60000,image_pixels))
    train_labels = np.fromfile("../../../../mnist_train_labels_binary.dat").reshape((60000,1))
test_imgs = np.fromfile("../../../../mnist_test_imgs_binary.dat").reshape((10000,image_pixels))
test_labels = np.fromfile("../../../../mnist_test_labels_binary.dat").reshape((10000,1))

nFile = 5
no_of_hidden_nodes = [45,60,80,100,120,150,200,300]
accuracy = 0.96
epochs = 30
accuracy_buffer = 5
save_frequency = 600
bias = None
learning_rate_original = [[0.04,0.05,0.06,0.07,0.08],
                          [0.07,0.08,0.09,0.1,0.11],
                          [0.14,0.16,0.18,0.2,0.22],
                          [0.14,0.16,0.18,0.2,0.22],
                          [0.18,0.2,0.22,0.24,0.26],
                          [0.18,0.2,0.22,0.24,0.26],
                          [0.18,0.2,0.22,0.24,0.26],                          
                          [0.24,0.26,0.28,0.3,0.32]]
learning_rate_factor = [[0.2,0.25,0.3],
                        [0.4,0.5,0.6],
                        [0.4,0.5,0.6],
                        [0.5,0.6,0.7],
                        [0.5,0.6,0.7],
                        [0.5,0.6,0.7],
                        [0.5,0.6,0.7],
                        [0.5,0.6,0.7]]
learning_rate_bias = [[0.2,0.3,0.4,0.5,0.6],
                      [0.3,0.4,0.5,0.6,0.7],
                      [0.3,0.4,0.5,0.6,0.7],
                      [0.3,0.4,0.5,0.6,0.7],
                      [0.3,0.4,0.5,0.6,0.7],
                      [0.3,0.4,0.5,0.6,0.7],
                      [0.3,0.4,0.5,0.6,0.7],
                      [0.3,0.4,0.5,0.6,0.7]]                     
adam_alpha = [[0.0005,0.001,0.0015,0.002],
              [0.0005,0.001,0.0015,0.002],
              [0.0005,0.001,0.0015,0.002],
              [0.0005,0.001,0.0015,0.002],
              [0.0005,0.001,0.0015,0.002],
              [0.0005,0.001,0.0015,0.002],
              [0.001,0.0015,0.002,0.0025],
              [0.001,0.0015,0.002,0.0025]]
adam_epsilon = [[1e-9,1e-8,1e-7,1e-6],
                [1e-8,1e-7,1e-6,1e-5],
                [1e-8,1e-7,1e-6,1e-5],
                [1e-9,1e-8,1e-7,1e-6],
                [1e-9,1e-8,1e-7,1e-6],
                [1e-9,1e-8,1e-7,1e-6],
                [1e-10,1e-9,1e-8,1e-7],
                [1e-10,1e-9,1e-8,1e-7]]
momentum_learning_rate = [[0.02,0.03,0.04,0.05],
                          [0.03,0.04,0.05,0.06],
                          [0.03,0.04,0.05,0.06],
                          [0.04,0.05,0.06,0.07],
                          [0.04,0.05,0.06,0.07],
                          [0.04,0.05,0.06,0.07],
                          [0.05,0.06,0.07,0.08],
                          [0.05,0.06,0.07,0.08]]
momentum_para = [[0.3,0.4,0.5,0.6],
                 [0.5,0.6,0.7,0.8],
                 [0.5,0.6,0.7,0.8],
                 [0.5,0.6,0.7,0.8],
                 [0.5,0.6,0.7,0.8],
                 [0.6,0.7,0.8,0.9],
                 [0.6,0.7,0.8,0.9],
                 [0.6,0.7,0.8,0.9]]

for iFile in range(nFile):
    for iHidden in range(len(no_of_hidden_nodes)):
        network = Klein.NeuralNetwork(no_of_in_nodes=image_pixels,
                                      no_of_out_nodes=no_of_different_labels,
                                no_of_hidden_nodes=no_of_hidden_nodes[iHidden],
                                learning_rate=learning_rate_original[0][0],
                                      data_array=train_imgs,
                                      labels=train_labels,
                                no_of_different_labels=no_of_different_labels,
                                      accuracy=accuracy,
                                      epochs=epochs,
                                      bias=bias,
                                      mean_square_error_cost=True,
                                      intermediate_results=True,
                                      data_array_for_testing=test_imgs,
                                      labels_for_testing=test_labels)
    
        bias_node = 1 if bias else 0
        
        network.wih_original = np.loadtxt("../Text/weights_original_nHidden400/wih_original_"+str(iFile)+".txt")[:no_of_hidden_nodes[iHidden]]
        network.who_original = np.loadtxt("../Text/weights_original_nHidden400/who_original_"+str(iFile)+".txt")[:,:no_of_hidden_nodes[iHidden]]
        network.order_data_array = np.loadtxt("../Text/order_data_array/order_"+str(iFile)+".txt",dtype="int32")

        # train the network with a constant learning rate
        network.learning_rate_bias = None
        arr_accuracy = np.nan * np.ones(len(learning_rate_original[iHidden]))
        arr_epoch = np.nan * np.ones(len(learning_rate_original[iHidden]))
        for iRate in range(len(learning_rate_original[iHidden])):
            if(np.isnan(learning_rate_original[iHidden][iRate])): continue
            network.learning_rate = learning_rate_original[iHidden][iRate]
            tmp_wih, tmp_who, tmp_accu = network.train_more_save("AlgoStandard", accuracy_buffer, save_frequency, randomise_patterns=False)
            print(iFile," ",iHidden," constant ",iRate)
        
            if(tmp_accu[-1]>accuracy):
                arr_accuracy[iRate] = tmp_accu[-1]
                arr_epoch[iRate] = len(tmp_accu)*save_frequency/60000.
            else:
                break
        
        np.savetxt("Text/nHidden_compare/accuracy_ConstantEta_nHidden"+str(no_of_hidden_nodes[iHidden])+"_file"+str(iFile)+".txt",arr_accuracy)
        np.savetxt("Text/nHidden_compare/epoch_ConstantEta_nHidden"+str(no_of_hidden_nodes[iHidden])+"_file"+str(iFile)+".txt",arr_epoch)


        
        # train the network with Adam
        network.learning_rate_bias = None
        network.beta_1 = 0.9
        network.beta_2 = 0.999
        arr_accuracy = np.nan * np.ones((len(adam_alpha[iHidden]),len(adam_epsilon[iHidden])))
        arr_epoch = np.nan * np.ones((len(adam_alpha[iHidden]),len(adam_epsilon[iHidden])))
        for iAlpha in range(len(adam_alpha[iHidden])):
            if(np.isnan(adam_alpha[iHidden][iAlpha])): continue            
            for iEpsi in range(len(adam_epsilon[iHidden])):
                network.alpha = adam_alpha[iHidden][iAlpha]
                network.epsilon = adam_epsilon[iHidden][iEpsi]
                tmp_wih, tmp_who, tmp_accu = network.train_more_save("AlgoAdamQ", accuracy_buffer, save_frequency, randomise_patterns=False)
                print(iFile," ",iHidden," Adam ",iAlpha," ",iEpsi)
            
                if(tmp_accu[-1]>accuracy):
                    arr_accuracy[iAlpha][iEpsi] = tmp_accu[-1]
                    arr_epoch[iAlpha][iEpsi] = len(tmp_accu)*save_frequency/60000.
            
        np.savetxt("Text/nHidden_compare/accuracy_Adam_nHidden"+str(no_of_hidden_nodes[iHidden])+"_file"+str(iFile)+".txt",arr_accuracy)
        np.savetxt("Text/nHidden_compare/epoch_Adam_nHidden"+str(no_of_hidden_nodes[iHidden])+"_file"+str(iFile)+".txt",arr_epoch)

        

        # train the network with Nesterov's Momentum
        network.learning_rate_bias = None
        arr_accuracy = np.nan * np.ones((len(momentum_learning_rate[iHidden]),len(momentum_para[iHidden])))
        arr_epoch = np.nan * np.ones((len(momentum_learning_rate[iHidden]),len(momentum_para[iHidden])))
        for iRate in range(len(momentum_learning_rate[iHidden])):
            if(np.isnan(momentum_learning_rate[iHidden][iRate])): continue
            network.learning_rate = momentum_learning_rate[iHidden][iRate]
            for iPara in range(len(momentum_para[iHidden])):
                tmp_wih, tmp_who, tmp_accu = network.train_more_save("AlgoNesterov", accuracy_buffer, save_frequency, momentum_para=momentum_para[iHidden][iPara])
                print(iFile," ",iHidden," Nesterov ",iRate," ",iPara)

                if(tmp_accu[-1]>accuracy):
                    arr_accuracy[iRate][iPara] = tmp_accu[-1]
                    arr_epoch[iRate][iPara] = len(tmp_accu)*save_frequency/60000.
            
        np.savetxt("Text/nHidden_compare/accuracy_Nesterov_nHidden"+str(no_of_hidden_nodes[iHidden])+"_file"+str(iFile)+".txt",arr_accuracy)
        np.savetxt("Text/nHidden_compare/epoch_Nesterov_nHidden"+str(no_of_hidden_nodes[iHidden])+"_file"+str(iFile)+".txt",arr_epoch)

                
        
        # train the network with dynamic learning rate
        arr_accuracy = np.nan * np.ones((len(learning_rate_factor[iHidden]),len(learning_rate_bias[iHidden])))
        arr_epoch = np.nan * np.ones((len(learning_rate_factor[iHidden]),len(learning_rate_bias[iHidden])))
        for iFac in range(len(learning_rate_factor[iHidden])):
            if(np.isnan(learning_rate_factor[iHidden][iFac])): continue
            network.learning_rate = learning_rate_factor[iHidden][iFac]
            for iBias in range(len(learning_rate_bias[iHidden])):
                network.learning_rate_bias = learning_rate_bias[iHidden][iBias]
                tmp_wih, tmp_who, tmp_accu = network.train_more_save("AlgoStandard", accuracy_buffer, save_frequency, randomise_patterns=False)
                print(iFile," ",iHidden," dynamic ",iFac," ",iBias)

                if(tmp_accu[-1]>accuracy):
                    arr_accuracy[iFac][iBias] = tmp_accu[-1]
                    arr_epoch[iFac][iBias] =len(tmp_accu)*save_frequency/60000.

            if( len( np.where(np.isnan(arr_accuracy[iFac])==False)[0] )==0 ):
                break
        np.savetxt("Text/nHidden_compare/accuracy_Dynamic_nHidden"+str(no_of_hidden_nodes[iHidden])+"_file"+str(iFile)+".txt",arr_accuracy)
        np.savetxt("Text/nHidden_compare/epoch_Dynamic_nHidden"+str(no_of_hidden_nodes[iHidden])+"_file"+str(iFile)+".txt",arr_epoch)


            
        del network

    
np.savetxt("Text/nHidden_compare/nHidden.txt",[no_of_hidden_nodes])
np.savetxt("Text/nHidden_compare/learning_rate_original.txt",learning_rate_original)
np.savetxt("Text/nHidden_compare/learning_rate_factor.txt",learning_rate_factor)
np.savetxt("Text/nHidden_compare/learning_rate_bias.txt",learning_rate_bias)
np.savetxt("Text/nHidden_compare/Adam_alpha.txt",adam_alpha)
np.savetxt("Text/nHidden_compare/Adam_epsilon.txt",adam_epsilon)
np.savetxt("Text/nHidden_compare/momentum_learning_rate.txt",momentum_learning_rate)
np.savetxt("Text/nHidden_compare/momentum_parameter.txt",momentum_para)
np.savetxt("Text/nHidden_compare/other_variable.txt",(save_frequency, accuracy, nFile))
