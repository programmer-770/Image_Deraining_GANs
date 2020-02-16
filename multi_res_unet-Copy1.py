
# coding: utf-8

# In[2]:


CUDA_VISIBLE_DEVICES = 1


# In[3]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


trainA = []
trainB = []
for i in range(1,701):
  img = cv2.imread('rain/{}clean.jpg'.format(i))
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  img = cv2.resize(img,(256,256))
  trainA.append(img)
  img = cv2.imread('rain/{}bad.jpg'.format(i))
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  img = cv2.resize(img,(256,256))
  trainB.append(img)
trainA = np.array(trainA)
trainB = np.array(trainB)
trainA = (trainA - 127.5)/127.5
trainB = (trainB - 127.5)/127.5


# In[ ]:


from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.utils.vis_utils import plot_model


def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
    '''
    2D Convolutional layers
    
    Arguments:
        x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
    
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(1, 1)})
        activation {str} -- activation function (default: {'relu'})
        name {str} -- name of the layer (default: {None})
    
    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    if(activation == None):
        return x

    x = Activation(activation, name=name)(x)

    return x


def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):
    '''
    2D Transposed Convolutional layers
    
    Arguments:
        x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
    
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(2, 2)})
        name {str} -- name of the layer (default: {None})
    
    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3, scale=False)(x)
    
    return x


def MultiResBlock(U, inp, alpha = 1.67):
    '''
    MultiRes Block
    
    Arguments:
        U {int} -- Number of filters in a corresponding UNet stage
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''

    W = alpha * U

    shortcut = inp

    shortcut = conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) +
                         int(W*0.5), 1, 1, activation=None, padding='same')

    conv3x3 = conv2d_bn(inp, int(W*0.167), 4, 4,
                        activation='relu', padding='same')

    conv5x5 = conv2d_bn(conv3x3, int(W*0.333), 4, 4,
                        activation='relu', padding='same')

    conv7x7 = conv2d_bn(conv5x5, int(W*0.5), 4, 4,
                        activation='relu', padding='same')

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    out = BatchNormalization(axis=3)(out)

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    return out


def ResPath(filters, length, inp):
    '''
    ResPath
    
    Arguments:
        filters {int} -- [description]
        length {int} -- length of ResPath
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''


    shortcut = inp
    shortcut = conv2d_bn(shortcut, filters, 1, 1,
                         activation=None, padding='same')

    out = conv2d_bn(inp, filters, 4, 4, activation='relu', padding='same')

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    for i in range(length-1):

        shortcut = out
        shortcut = conv2d_bn(shortcut, filters, 1, 1,
                             activation=None, padding='same')

        out = conv2d_bn(out, filters, 4, 4, activation='relu', padding='same')
        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis=3)(out)

    return out


def MultiResUnet(height, width, n_channels):
    '''
    MultiResUNet
    
    Arguments:
        height {int} -- height of image 
        width {int} -- width of image 
        n_channels {int} -- number of channels in image
    
    Returns:
        [keras model] -- MultiResUNet model
    '''


    inputs = Input((height, width, n_channels))

    mresblock1 = MultiResBlock(32, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
    mresblock1 = ResPath(32, 4, mresblock1)

    mresblock2 = MultiResBlock(32*2, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
    mresblock2 = ResPath(32*2, 3, mresblock2)

    mresblock3 = MultiResBlock(32*4, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
    mresblock3 = ResPath(32*4, 2, mresblock3)

    mresblock4 = MultiResBlock(32*8, pool3)


    up5 = concatenate([Conv2DTranspose(
        32*4, (2, 2), strides=(2, 2), padding='same')(mresblock4), mresblock3], axis=3)
    mresblock6 = MultiResBlock(32*4, up5)

    up6 = concatenate([Conv2DTranspose(
        32*2, (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock2], axis=3)
    mresblock7 = MultiResBlock(32*2, up6)

    up7 = concatenate([Conv2DTranspose(
        32, (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock1], axis=3)
    mresblock8 = MultiResBlock(32, up7)
    g = Conv2DTranspose(3, (4,4), strides=(1,1), padding='same')(mresblock8)
    output1 = Activation('tanh')(g)    

    #second encoder-decoder ##############################################################################################
    
    mresblock10 = MultiResBlock(32, output1)
    pool10 = MaxPooling2D(pool_size=(2, 2))(mresblock10)
    mresblock10 = ResPath(32, 4, mresblock10)

    bridge1 = concatenate([Conv2D(32*2,(2,2),strides=(2,2),padding='same')(mresblock10), mresblock7],axis=3)
    mresblock11 = MultiResBlock(32*2, bridge1)
    pool11 = MaxPooling2D(pool_size=(2, 2))(mresblock11)
    mresblock11 = ResPath(32*2, 3, mresblock11)
    
    bridge2 = concatenate([Conv2D(32*4,(2,2),strides=(2,2),padding='same')(mresblock11), mresblock6],axis=3)
    mresblock12 = MultiResBlock(32*4, bridge2)
    pool12 = MaxPooling2D(pool_size=(2, 2))(mresblock12)
    mresblock12 = ResPath(32*4, 2, mresblock12)

    bridge3 = concatenate([Conv2D(32*8,(2,2),strides=(2,2),padding='same')(mresblock12), mresblock4],axis=3)
    mresblock13 = MultiResBlock(32*8, pool12)

    up16 = concatenate([Conv2DTranspose(
        32*4, (2, 2), strides=(2, 2), padding='same')(mresblock13), mresblock12, mresblock3], axis=3)
    mresblock16 = MultiResBlock(32*4, up16)

    up17 = concatenate([Conv2DTranspose(
        32*2, (2, 2), strides=(2, 2), padding='same')(mresblock16), mresblock11, mresblock2], axis=3)
    mresblock17 = MultiResBlock(32*2, up17)

    up18 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(mresblock17), mresblock10, mresblock1], axis=3)
    mresblock18 = MultiResBlock(32, up18)
    g = Conv2DTranspose(3, (4,4), strides=(1,1), padding='same')(mresblock18)
    output = Activation('tanh')(g)       
    model = Model(inputs,output)

    return model
   


def main():

    # Define the model

    model = MultiResUnet(256, 256,3)
    model.summary()



if __name__ == '__main__':
    main()

# plot the model
#plot_model(model, to_file='generator_model_plot.png', show_shapes=True, show_layer_names=True)


# In[ ]:


from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, add, Concatenate
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.utils.vis_utils import plot_model
from keras.initializers import RandomNormal
from tensorflow.keras.losses import BinaryCrossentropy
def define_discriminator(image_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_src_image = Input(shape=image_shape)
	# target image input
	in_target_image = Input(shape=image_shape)
	# concatenate images channel-wise
	merged = Concatenate()([in_src_image, in_target_image])
	# C64
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer
	d = Conv2D(256, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# patch output
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('sigmoid')(d)
	# define model
	model = Model([in_src_image, in_target_image], patch_out)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss= BinaryCrossentropy(from_logits=True), optimizer=opt, loss_weights=[0.5])
	return model 
# define image shape
image_shape = (256,256,3)
# create the model
model = define_discriminator(image_shape)
# summarize the model
model.summary()
# plot the model
#plot_model(model, to_file='/content/drive/My Drive/test/discriminator_model_plot.png', show_shapes=True, show_layer_names=True)


# In[ ]:


from tensorflow.keras.losses import BinaryCrossentropy
def define_gan(g_model, d_model, image_shape):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # define the source image
    in_src = Input(shape=image_shape)
    in_target = Input(shape = image_shape)
    # connect the source image to the generator input
    gen_out = g_model(in_src)
    # connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src, gen_out])
    # src image as input, generated image and classification output
    model = Model([in_src,in_target], [dis_out, gen_out])
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=[BinaryCrossentropy(from_logits=True), 'mae'], optimizer=opt, loss_weights=[1,100])
    return model


# In[ ]:


def generate_real_samples(n_samples, patch_shape):
	# unpack dataset

	# choose random instances
	ix = np.random.randint(0, trainA.shape[0], n_samples)
	# retrieve selected images
	X1, X2 = trainA[ix], trainB[ix]
	# generate 'real' class labels (1)
	y = np.ones((n_samples, patch_shape, patch_shape, 1))
	return [X1, X2], y


# In[1]:


def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
	X = g_model.predict(samples)
	# create 'fake' class labels (0)
	y = np.zeros((len(X), patch_shape, patch_shape, 1))
	return X, y


# In[18]:


# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, d_model, gan_model, n_samples=1):
	# select a sample of input images
	[X_realA, X_realB], _ = generate_real_samples(n_samples, 1)
	# generate a batch of fake samples
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	# scale all pixels from [-1,1] to [0,1]
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0
	X_fakeB = 255 * X_fakeB	
	# plot generated target image
	for i in range(n_samples):
		plt.subplot(3, n_samples, 1 + n_samples + i)
		plt.axis('off')
		plt.imshow(X_fakeB[i])
	# save plot to file
	filename1 = 'test1/plot_%06d.png' % (step+1)
	cv2.imwrite(filename1,X_fakeB[0])
	# save the generator, discriminator and gan models
	filename2 = 'test1/g_model_%06d.h5' % (step+1)
	g_model.save(filename2)
	#filename3 = 'test/d_model_%06d.h5' % (step+1)
	#d_model.save(filename3)
	#filename4 = 'test/gan_model_%06d.h5' % (step+1)
	#gan_model.save(filename4)
	print('>Saved: %s and %s' % (filename1, filename2))


# In[32]:


def train(d_model, g_model, gan_model, n_epochs=200, n_batch=1, n_patch=32):
	# unpack dataset
  
	# calculate the number of batches per training epoch
	bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# manually enumerate epochs
	for i in range(n_steps):
		# select a batch of real samples
		[X_realA, X_realB], y_real = generate_real_samples( n_batch, n_patch)
		# generate a batch of fake samples
		X_fakeB, y_fake = generate_fake_samples(g_model, X_realB, n_patch)
		# update discriminator for real samples
		d_loss1 = d_model.train_on_batch([X_realB, X_realA], y_real)
		# update discriminator for generated samples
		d_loss2 = d_model.train_on_batch([X_realB, X_fakeB], y_fake)
		# update the generator
		g_loss, _, _ = gan_model.train_on_batch([X_realB,X_realA], [y_fake, X_realA])
		# summarize performance
		print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
    # summarize model performance
		if (i+1) % (bat_per_epo * 1) == 0:
			summarize_performance(i, g_model,d_model, gan_model) 


# In[ ]:


image_shape = (256,256,3)
# define the models
d_model = define_discriminator(image_shape)
g_model = MultiResUnet(256,256,3)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)
# train model
train(d_model, g_model, gan_model)

