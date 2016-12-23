from __future__ import division  # floating point division
import numpy as np
import pandas as pd
import utilities as utils
import math;
from numpy import array;

class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """
    
    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        
    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params,parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}
            
    def getparams(self):
        return self.params
    
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        
    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """
    def __init__( self, parameters={} ):
        self.params = {'regwgt': 0.01}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        yt[yt == 0] = -1
        
        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.pinv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.params['regwgt']*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples
        
    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1     
        ytest[ytest < 0] = 0
        #print("ytest linear regression ", ytest);
        return ytest
        
class NaiveBayes(Classifier):
    """ Gaussian naive Bayes;  """
    
    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        # Assumes that a bias unit has been added to feature vector as the last feature
        # If usecolumnones is False, it ignores this last feature
        self.params = {'usecolumnones': False}
        self.reset(parameters)
        self.Xtrain_classzero = [];
        self.Xtrain_classone = [];
        self.Xtrain_classone_mean = [];
        self.Xtrain_classone_sd = [];
        self.Xtrain_classzero_mean = [];
        self.Xtrain_classzero_sd = [];
        self.predicted_y = [];
            
    def reset(self, parameters):
        self.resetparams(parameters)
        self.Xtrain_classzero = [];
        self.Xtrain_classone = [];
        self.Xtrain_classone_mean = [];
        self.Xtrain_classone_sd = [];
        self.Xtrain_classzero_mean = [];
        self.Xtrain_classzero_sd = [];
        self.predicted_y = [];
        # TODO: set up required variables for learning

    def calculate_mean(self, rows_temp_list):
        mean = sum(rows_temp_list) / len(rows_temp_list);
        return mean;
        
    def calculate_standard_deviation(self, rows_temp_list):
        mean = self.calculate_mean(rows_temp_list);
        variance = 0;
        for number in rows_temp_list:
            variance += (mean - number) ** 2;
        variance /= float(len(rows_temp_list)-1)
        sd = math.sqrt(variance)
        return sd

    def calculate_pdf(self, x, mean, sd):
        try:
            pdf_of_x = ((1.0 / (math.sqrt(2.0 * math.pi) * sd)) * np.exp(-((x - mean) * (x -  mean)) / (2.0 * sd * sd)));
            return pdf_of_x;
        except:
            return 1;
    def split_classes(self, Xtrain, ytrain):
        ignore_last_column = 0;
        if self.params['usecolumnones'] is False:
            ignore_last_column = 1;
        else:
            ignore_last_column = 0;
            
        for rows_count in range(0, Xtrain.shape[0]):
            if ytrain[rows_count] == 1:
                self.Xtrain_classone.append(Xtrain[rows_count]);
            elif ytrain[rows_count] == 0:
                self.Xtrain_classzero.append(Xtrain[rows_count]);
        
        for columns_count in range(0, len(self.Xtrain_classone[0]) - ignore_last_column):
            rows_temp_list = [];
            for rows_count in range(0, len(self.Xtrain_classone)):
                rows_temp_list.append(self.Xtrain_classone[rows_count][columns_count]);
            self.Xtrain_classone_mean.append(self.calculate_mean(rows_temp_list));    
            self.Xtrain_classone_sd.append(self.calculate_standard_deviation(rows_temp_list));

        for columns_count in range(0, len(self.Xtrain_classzero[0])  - ignore_last_column):
            rows_temp_list = [];
            for rows_count in range(0, len(self.Xtrain_classzero)):
                rows_temp_list.append(self.Xtrain_classzero[rows_count][columns_count]);
            self.Xtrain_classzero_mean.append(self.calculate_mean(rows_temp_list));    
            self.Xtrain_classzero_sd.append(self.calculate_standard_deviation(rows_temp_list));

        #print("Xtrainclasszero_sd ", self.Xtrain_classzero_sd);
            
    # TODO: implement learn and predict functions
    def learn(self, Xtrain, ytrain):
        self.split_classes(Xtrain, ytrain);

    def predict(self, Xtest):
        ignore_last_column = 0;
        if self.params['usecolumnones'] is False:
            ignore_last_column = 1;
        else:
            ignore_last_column = 0;
        #print("Naive Bayes ", self.params['usecolumnones'], "  ", ignore_last_column);
        
        for rows_count in range(0, len(Xtest)):
            individual_row = Xtest[rows_count];
            class_zero = 1.0;
            class_one = 1.0;
            for each_attribute in range(0, len(individual_row) - ignore_last_column):
                class_zero = class_zero * self.calculate_pdf(individual_row[each_attribute], self.Xtrain_classzero_mean[each_attribute], self.Xtrain_classzero_sd[each_attribute]);
                class_one = class_one * self.calculate_pdf(individual_row[each_attribute], self.Xtrain_classone_mean[each_attribute], self.Xtrain_classone_sd[each_attribute]);
            if class_zero > class_one:
                self.predicted_y.append(0);
            elif class_one > class_zero:
                self.predicted_y.append(1);
            #print("classes ", class_zero, " ", class_one);
        #print("test ", len(Xtest), " ", len(self.predicted_y));
        return self.predicted_y;
           
class LogitReg(Classifier):

    def __init__( self, parameters={} ):
        # Default: no regularization
        self.params = {'regwgt': 0.0, 'regularizer': 'None'}
        self.reset(parameters)
        self.weights = [];
        self.learningrate = 0.1;
        self.predicted_ytest = [];
        self.lambda_reg = 0.001;
        self.regularization_param = 0;

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        self.weights = [];
        self.learningrate = 0.1;
        self.predicted_ytest = [];
        self.regularization_param = 0;
        if self.params['regularizer'] is 'l1':
            self.regularizer = (utils.l1, utils.dl1)
        elif self.params['regularizer'] is 'l2':
            self.regularizer = (utils.l2, utils.dl2)
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))

    def transfer_function(self, x_value):
        estimated_y_value = (1 / (1 + np.exp(-x_value)));
        return estimated_y_value;
        
     
    # TODO: implement learn and predict functions
    def learn(self, Xtrain, ytrain):
        #print("whoa " , np.shape(Xtrain), "  ", np.shape(ytrain));
        #print("whoa " , Xtrain.shape[1]);
        #print("rows ", Xtrain[0]);
        self.weights = [0]*(Xtrain.shape[1]);
        #print("weights ", self.weights);
        for epoch_count in range(50):
            for rows in range(Xtrain.shape[0]):
                #print("rows ", Xtrain[rows]);
                #print("output rows ", ytrain[rows]);
                self.regularization_param = 0;
                each_row = Xtrain[rows];
                x_value = self.weights[0];
                for each_row_temp in range(1, len(each_row)):
                    x_value = x_value + (self.weights[each_row_temp] * each_row[each_row_temp]);
                estimated_y_value = self.transfer_function(x_value);

                #self.weights[0] = self.weights[0] + (self.learningrate * (ytrain[rows] - estimated_y_value) * estimated_y_value * (1- estimated_y_value) * 1.0);

                if self.params['regularizer'] is 'l1':
                        for weights_count in range(0, len(self.weights)):
                            self.regularization_param = self.regularization_param + (self.lambda_reg * self.weights[weights_count]);
                elif self.params['regularizer'] is 'l2':
                        for weights_count in range(0, len(self.weights)):
                            if self.weights[weights_count] != 0:
                                self.regularization_param = self.regularization_param + (self.lambda_reg * (self.weights[weights_count] / abs(self.weights[weights_count])));
                elif self.params['regularizer'] is 'other':
                    for weights_count in range(0, len(self.weights)):
                            if self.weights[weights_count] != 0:
                                self.regularization_param = self.regularization_param + (self.lambda_reg * self.weights[weights_count]) + (self.lambda_reg * (self.weights[weights_count] / abs(self.weights[weights_count])));
                else:
                    self.regularization_param = 0;

                #print("error ", ytrain[rows] - estimated_y_value);
                if abs(ytrain[rows] - estimated_y_value) < 0.0005:
                    break;

                #self.weights[0] = self.weights[0] + (self.learningrate * (ytrain[rows] - estimated_y_value) * estimated_y_value * (1- estimated_y_value) * 1.0);
                
                for individual_weights in range(0, len(self.weights)):
                    #code to implement the l1 regularization:
                    self.weights[individual_weights] = self.weights[individual_weights] + (self.learningrate * ((ytrain[rows] - estimated_y_value) + self.regularization_param)  * estimated_y_value
                                                                                           * (1- estimated_y_value) * each_row[individual_weights]);
                    
            #print("weights: ", self.weights);

    def predict(self, Xtest):
        #print("regularizer parameter ", self.params['regularizer']);
        #print("predict ", Xtest);
        for rows in range(Xtest.shape[0]):
            x_value = self.weights[0];
            each_row = Xtest[rows];
            for each_row_temp in range(1, len(each_row)):
                x_value = x_value + (self.weights[each_row_temp] * each_row[each_row_temp]);
            estimated_y_value = self.transfer_function(x_value);
            if estimated_y_value > 0.5:
                estimated_y_value = 1;
            else:
                estimated_y_value = 0;
            self.predicted_ytest.append(estimated_y_value);
        #print("ytest ", self.predicted_ytest);
        return self.predicted_ytest;
            
                

class NeuralNet(Classifier):

    def __init__(self, parameters={}):
        self.params = {'nh': 4,
                       'regwgt': 0.01,
                        'transfer': 'sigmoid',
                        'stepsize': 0.01,
                        'epochs': 10}
        self.reset(parameters)
        self.learningrate = 0.01;
        self.predicted_ytest = [];

    def reset(self, parameters):
        self.resetparams(parameters)
        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')      
        self.wi = None
        self.wo = None
        self.learningrate = 0.01;
        self.predicted_ytest = [];
        self.weights = None;

    def transfer_function(self, x_value):
        estimated_y_value = (1 / (1 + np.exp(-x_value)));
        return estimated_y_value;
        
    # TODO: implement learn and predict functions
    def learn(self, Xtrain, ytrain):
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        yt[yt == 0] = -1
        
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.pinv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.params['regwgt']*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples

        for epoch_count in range(self.params['epochs']):
            for rows in range(Xtrain.shape[0]):
                #print("rows ", Xtrain[rows]);
                #print("output rows ", ytrain[rows]);
                
                each_row = Xtrain[rows];
                x_value = self.weights[0];
                for each_row_temp in range(1, len(each_row)):
                    x_value = x_value + (self.weights[each_row_temp] * each_row[each_row_temp]);
                estimated_y_value = self.transfer_function(x_value);

                #self.weights[0] = self.weights[0] + (self.learningrate * (ytrain[rows] - estimated_y_value) * estimated_y_value * (1- estimated_y_value) * 1.0);

                for individual_weights in range(0, len(self.weights)):
                    self.weights[individual_weights] = self.weights[individual_weights] + (self.learningrate * (ytrain[rows] - estimated_y_value) * estimated_y_value
                                                                                           * (1- estimated_y_value) * each_row[individual_weights]);
            #print("neural network weights ", self.weights);

    
    def _evaluate(self, inputs):
        """ 
        Returns the output of the current neural network for the given input
        The underscore indicates that this is a private function to the class NeuralNet
        """
        if inputs.shape[0] != self.ni:
            raise ValueError('NeuralNet:evaluate -> Wrong number of inputs')
        
        # hidden activations
        ah = self.transfer(np.dot(self.wi,inputs))  

        # output activations
        ao = self.transfer(np.dot(self.wo,ah))
        
        return (ah, ao)

    def predict(self, Xtest):
        #print("predict ", Xtest);
        for rows in range(Xtest.shape[0]):
            x_value = self.weights[0];
            each_row = Xtest[rows];
            for each_row_temp in range(1, len(each_row)):
                x_value = x_value + (self.weights[each_row_temp] * each_row[each_row_temp]);
            estimated_y_value = self.transfer_function(x_value);
            if estimated_y_value > 0.5:
                estimated_y_value = 1;
            else:
                estimated_y_value = 0;
            self.predicted_ytest.append(estimated_y_value);
        #print("ytest ", self.predicted_ytest);
        return self.predicted_ytest;

    

class LogitRegAlternative(Classifier):

    def __init__( self, parameters={} ):
        self.reset(parameters)
        self.weights = [];
        self.learningrate = 0.1;
        self.predicted_ytest = [];

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        self.weights = [];
        self.learningrate = 0.1;
        self.predicted_ytest = [];

    def transfer_function(self, x_value):
        #estimated_y_value = (1 / (1 + np.exp(-x_value)));
        estimated_y_value = ((1.0/2.0)*(1.0+(x_value / (math.sqrt(1.0 + (x_value*x_value))))));
        return estimated_y_value;
        
    def learn(self, Xtrain, ytrain):
        #print("whoa " , np.shape(Xtrain), "  ", np.shape(ytrain));
        #print("whoa " , Xtrain.shape[1]);
        #print("rows ", Xtrain[0]);
        self.weights = [0]*Xtrain.shape[1];
        #print("weights ", self.weights);
        for epoch_count in range(50):
            for rows in range(Xtrain.shape[0]):
                #print("rows ", Xtrain[rows]);
                #print("output rows ", ytrain[rows]);
                each_row = Xtrain[rows];
                x_value = self.weights[0];
                for each_row_temp in range(1, len(each_row)):
                    x_value = x_value + (self.weights[each_row_temp] * each_row[each_row_temp]);
                estimated_y_value = self.transfer_function(x_value);

                self.weights[0] = self.weights[0] + (self.learningrate * (ytrain[rows] - estimated_y_value) * estimated_y_value * (1- estimated_y_value) * 1.0);

                for individual_weights in range(1, len(self.weights)):
                    self.weights[individual_weights] = self.weights[individual_weights] + (self.learningrate * (ytrain[rows] - estimated_y_value) * estimated_y_value
                                                                                       * (1- estimated_y_value) * each_row[individual_weights]);
                
        #print("weights: ", self.weights);

    def predict(self, Xtest):
        #print("predict ", Xtest);
        for rows in range(Xtest.shape[0]):
            x_value = self.weights[0];
            each_row = Xtest[rows];
            for each_row_temp in range(1, len(each_row)):
                x_value = x_value + (self.weights[each_row_temp] * each_row[each_row_temp]);
            estimated_y_value = self.transfer_function(x_value);
            if estimated_y_value > 0.5:
                estimated_y_value = 1;
            else:
                estimated_y_value = 0;
            self.predicted_ytest.append(estimated_y_value);
        #print("ytest ", self.predicted_ytest);
        return self.predicted_ytest;
            
                
         
        
           
    
