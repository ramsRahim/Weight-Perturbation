import sys
sys.path.append('../Project_Clustering')

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
import math
import matplotlib.pyplot as plt

def singleLayer(data, en_act, de_act, op , n_enco, epoch, Mask, mtype = None, Limit = False):
    
  
    input_data = scale_0_1(data)
    output_data = scale_0_1(data)


    # Autoencoder with 1 hidden layer
    n_samp, n_input = input_data.shape 
    n_hidden = n_enco

    x = tf.placeholder("float", [None, n_input])

    ## setting random seed
    #tf.random.set_random_seed(10)

    # Weights and biases to hidden layer
    Wh = tf.Variable(tf.random_uniform((n_input, n_hidden), 0, 1.0 / math.sqrt(n_input),seed = 10)) 
    bh = tf.Variable(tf.zeros([n_hidden]))
    
    if en_act == "tanh":
        h = tf.nn.tanh(tf.matmul(x,Wh) + bh)
    elif en_act == "relu":
        h = tf.nn.relu(tf.matmul(x,Wh) + bh)
    elif en_act == "sigmoid":
        h = tf.nn.sigmoid(tf.matmul(x,Wh) + bh)

    # Weights and biases to hidden layer
    Wo = tf.transpose(Wh) # tied weights
    bo = tf.Variable(tf.zeros([n_input]))
    
    if de_act == "tanh":
        y = tf.nn.tanh(tf.matmul(h,Wo) + bo)
    elif de_act == "relu":
        y = tf.nn.relu(tf.matmul(h,Wo) + bo)
    elif de_act == "sigmoid":
        y = tf.nn.sigmoid(tf.matmul(h,Wo) + bo)


    # Objective functions
    y_ = tf.placeholder("float", [None,n_input])

    meansq = tf.reduce_mean(tf.square(y_- y))

    if op == "grad" :
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(meansq)
    else :
        train_step = tf.train.RMSPropOptimizer(0.01).minimize(meansq)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    n_rounds = epoch
    batch_size = 50
    perc_weight = []
    loss = []
    i = 0
    #num_batch = n_samp/batch_sizes
    for n_epoch in range(n_rounds):

        for k in range(0, n_samp, batch_size):

            batch_xs = input_data[k:k + batch_size][:]
            batch_ys = output_data[k:k + batch_size][:]

            sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys}) 

        loss.append(sess.run(meansq, feed_dict={x: batch_xs, y_:batch_ys})) 

        if (Mask and Limit) :
            if n_epoch % 50 == 10 and i<3:
                #print (n_epoch)
                
           #     print ('Masking now with limit')
                W =  sess.run(Wh)
                d, m = np.histogram (np.ravel(W), bins= 100)
                if i == 0:
                    Wold = np.ones(W.shape)
                    
                Wold, perc = weight_perc(Wold,W,m,d, mtype=mtype)
                perc_weight.append (perc)
                wm = np.multiply(Wold,W)
                sess.run(Wh.assign( wm))
                i = i+1
                
        if (Mask and not Limit) :
            if n_epoch % 50 == 10:
                
             #   print ('Masking now without limit')
                #print (n_epoch)
                W =  sess.run(Wh)
                d, m = np.histogram (np.ravel(W), bins= 100)
                
                if i == 0:
                    Wold = np.ones(W.shape)
                    
                Wold, perc = weight_perc(Wold,W,m,d, mtype = mtype)
                perc_weight.append (perc)
                wm = np.multiply(Wold,W)
                sess.run(Wh.assign( wm))
                i = i+1    

    hidden_outputs = sess.run(h,feed_dict={x: input_data})

    
 
       
    w_out = sess.run(Wh)
 
    return loss,perc_weight, w_out

  
    
# extreme low value masking
def maskExtLow(W): 
    
        lim_upper =(W.max())*0.05
        lim_lower = (W.min())*0.05
        B1 = W > lim_upper
        B2 = W < lim_lower
        B = np.logical_or(B1, B2)
        #B = np.logical_not(B)
        B.astype(np.int)
        Wm = np.multiply (W, B)
        return Wm
    
    
# Expo masking

def maskExpo(W):
    
    w_max = np.max(W)
    alpha = np.divide (2.3, w_max) # constrains all weight values between 0 and 10
    Wm = np.exp(alpha*W)
    
    return Wm

    
#extreme boundary value masking
def maskExtBound(W):
    
    lower_limit = np.mean(W) - 3*np.std(W)
    upper_limit = np.mean(W) + 3*np.std(W)

    B1 = W > upper_limit
    B2 = W < lower_limit
    B = np.logical_or(B1, B2)
    B = np.logical_not(B)
    B.astype(np.int)
    Wm = np.multiply (W, B)
    
    return Wm
        
       
    
# negative weight masking
def maskNeg(W):
        
        B = W > 0.00
        B.astype(np.int)
        Wm = np.multiply (W, B)
        return Wm
    
#histogram based masking
def histMask(W,m,d):
        
        ind = m[:-1]
        lim = d.mean()
        m = ind[d>=lim]
        lim_lower = m.min()
        lim_upper = m.max()
        B1 = W > lim_upper
        B2 = W < lim_lower
        B = np.logical_or(B1, B2)
        #B = np.logical_not(B)
        B.astype(np.int)
        Wm = np.multiply (W, B)
        return Wm
    
def weight_perc(Wold,W,m,d, mtype):
        
        if mtype =='expo':
            
            Wn = maskExpo(W)
        
        if mtype == 'Low':
            
                Wn = maskExtLow(W)
                
        if mtype == 'Bound':
             
                Wn = maskExtBound(W)
                  
        if mtype == 'Neg':
             
                Wn = maskNeg(W)
                
        Wt = np.multiply(Wn,Wold)
        Wold = Wt.astype(np.bool) 
        Wold = Wold.astype(np.int)
        cnt_zero = len(np.ravel(Wold))-np.count_nonzero(Wold)
        perc = (( cnt_zero)*100)/len(np.ravel(Wold))
      #  print( "% of weight masked", perc)  
       # perc_weights.append(perc)
        return Wold, perc
        
def scale_0_1 (data):

        scaled_data = np.divide((data-data.min()), (data.max()-data.min()))
        return scaled_data

def scale_1_1 (data):

        norm_data = scale_0_1(data)
        scaled_data = (norm_data*2)-1
        return scaled_data

def plotPercMask (perc_weight, fileName):
      
        fig = plt.figure()
        plt.plot([*range(len(perc_weight))], perc_weight,'b.-',linewidth=2,                                                                               markersize=12)
    
      #  plt.scatter([*range(len(perc_weight))],perc_weight)
        plt.ylim(0,30)
        plt.xlim(0,len(perc_weight))
        
        plt.tick_params(axis='x', labelsize=14, labelcolor='k')
        plt.tick_params(axis='y', labelsize=14, labelcolor='k')
        plt.xlabel ('Number of perturbations',color='k',fontsize=14)
        plt.ylabel ('% of Weight Masked',color='k', fontsize=14)
    
        plt.savefig('Figure/Perc_Weight/'+fileName+ 'wt_drop.pdf', bbox_inches='tight')
   #     plt.clf()
        plt.close(fig)
        
def plotLossCurve (loss_1, loss_2, fileName):
    
    fig = plt.figure()
    plt.plot(-np.log(loss_1[:1000]),'r.-',label='with masking')
    plt.plot(-np.log(loss_2[:1000]),'g.-',label="without masking")
    plt.legend(fontsize=14)

        
    plt.xlim((0,1000))
    plt.tick_params(axis='x', labelsize=14, labelcolor='k')
    plt.tick_params(axis='y', labelsize=14, labelcolor='k')
    plt.xlabel ('Number of epochs',color='k',fontsize=14)
    plt.ylabel ('Negative log loss',color='k', fontsize=14)
    
    plt.savefig('Figure/LossCurves/'+fileName+ 'loss.pdf', bbox_inches='tight')
  #  plt.clf()
    plt.close(fig)

def activationMap (dBName):
    
    if (dBName == 'ad_data' or dBName == 'heart_failure'):
        
        en_act = 'relu'
        de_act = 'relu'
        
    elif (dBName == 'gene_seq'):
        
        en_act = 'sigmoid'
        de_act = 'relu'
        
    else: 
    
        en_act = 'sigmoid'
        de_act = 'sigmoid'
        
        
    return en_act, de_act
    
    
        
    
