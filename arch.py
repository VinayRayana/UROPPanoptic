import tensorflow as tf
def architecture(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
#Build the model
	inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
	s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

	#Contraction path
	c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
	c1 = tf.keras.layers.Dropout(0.1)(c1)
	c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
	p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

	c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
	c2 = tf.keras.layers.Dropout(0.1)(c2)
	c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
	p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
	 
	c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
	c3 = tf.keras.layers.Dropout(0.2)(c3)
	c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
	p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
	 
	c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
	c4 = tf.keras.layers.Dropout(0.2)(c4)
	c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
	p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
	 
	c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
	c5 = tf.keras.layers.Dropout(0.3)(c5)
	c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
	p5 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c5)
	 
	c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p5)
	c6 = tf.keras.layers.Dropout(0.3)(c6)
	c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
	 
	#Expansive path 
	u7 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
	u7 = tf.keras.layers.concatenate([u7, c5])
	c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
	c7 = tf.keras.layers.Dropout(0.2)(c7)
	c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
	 
	u8 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
	u8 = tf.keras.layers.concatenate([u8, c4])
	c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
	c8 = tf.keras.layers.Dropout(0.2)(c8)
	c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
	 
	u9 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
	u9 = tf.keras.layers.concatenate([u9, c3])
	c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
	c9 = tf.keras.layers.Dropout(0.2)(c9)
	c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
	 
	u10 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c9)
	u10 = tf.keras.layers.concatenate([u10, c2])
	c10 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u10)
	c10 = tf.keras.layers.Dropout(0.1)(c10)
	c10 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c10)
	 
	u11 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c10)
	u11 = tf.keras.layers.concatenate([u11, c1], axis=3)
	c11 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u11)
	c11 = tf.keras.layers.Dropout(0.1)(c11)
	c11 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c11)
	 
	outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c11)
	 
	model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	return model