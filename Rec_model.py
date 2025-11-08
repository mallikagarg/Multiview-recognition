from keras.layers import Dense, Dropout, Activation, \
                         Flatten, Convolution2D, MaxPooling2D, \
                         BatchNormalization, Conv2D,Conv3D, Input,AveragePooling2D,concatenate
from keras.models import Model
from keras import optimizers
import keras

from Segmentation.Seg_Model import SegModel


class RecModel(object):
        def __init__(self, input_size,num_class):
            self.input_size=input_size
            self.num_class=num_class

            self._build_model()

        def _build_model(self):
                SegM=SegModel(self.input_size)
                Smodel=SegM.model
                Smodel.load_weights('Seg_weight.hdf5')
                l=len(Smodel.layers)
                for layer in Smodel.layers[:l]:
                    layer.trainable = False

                inp = Input(shape=self.input_size)
                inp_stream1=Smodel.input
                inp_stram2 = Smodel.output

                ###Stream1------------------

                x = Conv2D(8, 3, activation = 'relu', padding = 'same' , dilation_rate=1,name='CV1')(inp_stream1)
                #x=keras.regularizers.l1(0.01)(x)
                x = MaxPooling2D(pool_size=(3, 3))(x)
                
                
                x = Conv2D(16, 3, activation = 'relu', padding = 'same',dilation_rate=2,name='CV2')(x)
                x = MaxPooling2D(pool_size=(3, 3))(x)

                x = Conv2D(32, 3, activation = 'relu', padding = 'same',name='CV4')(x)
                x = MaxPooling2D(pool_size=(3, 3))(x)

                x1 = Conv2D(64, 3, activation = 'relu', padding = 'same',name='CV41')(x)
                x1 = MaxPooling2D(pool_size=(3, 3))(x1)
                
                x1 = Conv2D(128, 3, activation = 'relu', padding = 'same',name='CV42')(x1)
                x1 = MaxPooling2D(pool_size=(3, 3))(x1)
                x1 = Dropout(0.2)(x1)
                
                x2 = Conv2D(16, 3, activation = 'relu', padding = 'same',dilation_rate=1,name='CV112')(inp_stream1)
                x2 = MaxPooling2D(pool_size=(3, 3))(x2)

                x2 = Conv2D(16, 5, activation = 'relu', padding = 'same',dilation_rate=2,name='CV21')(x2)
                x2 = MaxPooling2D(pool_size=(3, 3))(x2)

                x2 = Conv2D(16, 7, activation = 'relu', padding = 'same',name='CV411')(x2)
                x2 = MaxPooling2D(pool_size=(3, 3))(x2)
                x2 = Conv2D(16, 9, activation = 'relu', padding = 'same' ,name='CV33')(x2)
                x2 = MaxPooling2D(pool_size=(3, 3))(x2)
                
                x2 = Conv2D(16, 11, activation = 'relu', padding = 'same' ,name='CV331')(x2)
                x2 = MaxPooling2D(pool_size=(3, 3))(x2)
                x2 = Dropout(0.2)(x2)
                f=concatenate([x1,x2],axis = 3) 
                
                
                ###Stream2--------------------
                """
                x1 = Conv2D(16, 3, activation = 'relu', padding = 'same',dilation_rate=1,name='CV11')(inp_stram2)
                x1 = MaxPooling2D(pool_size=(3, 3))(x1)

                x1 = Conv2D(32, 3, activation = 'relu', padding = 'same',dilation_rate=2,name='CV421')(x1)
                x1 = MaxPooling2D(pool_size=(3, 3))(x1)
                              
                x1 = Conv2D(64, 3, activation = 'relu', padding = 'same',dilation_rate=2 ,name='CV31')(x1)
                x1 = MaxPooling2D(pool_size=(3, 3))(x1)

                
                x1 = Conv2D(128, 3, activation = 'relu', padding = 'same',dilation_rate=2,name='CV412')(f)
                xf2 = MaxPooling2D(pool_size=(3, 3))(x1)
                #xf2 = Dropout(0.45)(xf2)

                ####--------
                #f=concatenate([xf2,xf1],axis = 3) 
                #f = MaxPooling2D(pool_size=(3, 3))(xf1)
                """

                f = Flatten()(f)

                f = Dropout(0.45)(f)

                prediction = Dense(self.num_class, activation="softmax")(f)
                model_final = Model(inputs = Smodel.input, outputs = prediction)
                print(model_final.summary())
                                                  
                self.model_F=model_final

