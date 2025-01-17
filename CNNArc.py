### a proposal arc based on cnns for image Classification

## keras model

input00 = keras.layers.Input(shape=(img_h , img_w , 3) , name='input')
conv00 = keras.layers.Conv2D( 128, (5,5) , activation = 'relu' , padding='same' )(input00)
batchNorm00 = keras.layers.BatchNormalization()(conv00)
conv01 = keras.layers.Conv2D( 256, (3,3) , activation = 'relu' , padding='same' )(batchNorm00)
batchNorm001 = keras.layers.BatchNormalization()(conv01)
pool0 = keras.layers.MaxPooling2D()(batchNorm001)
conv02 = keras.layers.Conv2D( 128 , (3,3) , activation = 'relu' , padding='same' )(pool0)
conv12 = keras.layers.Conv2D( 64 , (5,5) , activation = 'relu' , padding='same' )(pool0)
conv13 = keras.layers.Conv2D( 64 , (1,1) , activation = 'relu' , padding='same' )(pool0)
concat1 = keras.layers.Concatenate(axis=-1)([  conv02 , conv12 , conv13])
batchNorm01 = keras.layers.BatchNormalization()(concat1)
pool1 = keras.layers.MaxPooling2D()(batchNorm01)
conv21 = keras.layers.Conv2D( 32 , (5,5) , activation= 'relu' , padding='same' )(pool1)
conv22 = keras.layers.Conv2D( 64 , (3,3) , activation= 'relu' , padding='same' )(pool1)
conv23 = keras.layers.Conv2D( 64 , (1,1) , activation= 'relu' , padding='same' )(pool1)
concat2 = keras.layers.Concatenate(axis=-1)([ conv21 , conv22 , conv23 ])
batchNorm02 = keras.layers.BatchNormalization()(concat2)
pool2 = keras.layers.MaxPooling2D()(batchNorm02)
convf1 = keras.layers.Conv2D( 32 , (5,5) , activation= 'relu' , padding='same' )(pool2)
batchNorm03 = keras.layers.BatchNormalization()(convf1)
pool3 = keras.layers.MaxPooling2D()(batchNorm03)
convf2 = keras.layers.Conv2D( 16 , (3,3) , activation= 'relu' , padding='valid' )(pool3)
batchNorm04 = keras.layers.BatchNormalization()(convf2)
pool4 = keras.layers.MaxPooling2D()(batchNorm04)
convf3 = keras.layers.Conv2D( 16 , (3,3) , activation= 'relu' , padding='valid' )(pool4)
pool5 = pool4 = keras.layers.MaxPooling2D()(convf3)
flatten = keras.layers.Flatten()(pool5)
fc1 = keras.layers.Dense( 256 , activation= 'relu')(flatten)
drop1 = keras.layers.Dropout(0.6)(fc1)
fc2 = keras.layers.Dense( 256 , activation= 'relu')(drop1)
drop3 = keras.layers.Dropout(0.5)(fc2)
fcfinal = keras.layers.Dense( 2 , activation = 'softmax')(drop3)
model = keras.models.Model(inputs=[input00] , outputs=[fcfinal])
model.summary()