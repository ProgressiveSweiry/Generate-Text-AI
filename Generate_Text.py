# Import Required Libraries
import tensorflow as tf
import numpy as np
import os
import time
import codecs

class Generate_Text(object):
    
        def __init__(self, start_string, num_runs=1):
            #Settings
            #train the model?
            self.train = False
            self.generate_t = True
            
            #setting epochs
            self.EPOCHS=int(num_runs)
            
            #How many char generate ?
            self.num_generate = 1000
            
            #=========================================================
            self.start_string = start_string
            if(start_string == "train"):
                self.train = True
                self.generate_t = False   
                
            #============================
            # 'tx.txt' is the file from which the program trains.
            text = open('tx.txt', 'rb').read().decode(encoding='utf-8')
            
            # Setting Offset,filtering and appending into $new_text
            new_text = [""]
            for t in text:
                new_text.append(t[17:])
            new_text = [_ for i in range(len(text)) for _ in text[i]]
            
           #============================
           
           # Sorting text into readable dataset for the Neural Network to process
            vocab = sorted(set(text))
            char2idx = {u:i for i, u in enumerate(vocab)}
            idx2char = np.array(vocab)
            text_as_int = np.array([char2idx[c] for c in text])
            self.char2idx = char2idx
            self.idx2char = idx2char
            self.text_as_int = text_as_int
            seq_length = 100
            examples_per_epoch = len(text)//(seq_length+1)
            char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
            sequences = char_dataset.batch(seq_length+1,drop_remainder=True)
            dataset = sequences.map(self.split_input_target)
            
            # NN Model Settings
            BATCH_SIZE= 64
            BUFFER_SIZE = 295
            dataset = dataset.shuffle(BUFFER_SIZE,reshuffle_each_iteration=True).batch(BATCH_SIZE, drop_remainder=True)
            self.dataset = dataset
            self.vocab_size = len(vocab)
            self.embedding_dim = 256
            self.rnn_units = 1024
            
            #Build The Model
            self.model = self.build_model(
                vocab_size = self.vocab_size,
                embedding_dim = self.embedding_dim,
                rnn_units = self.rnn_units,
                batch_size = BATCH_SIZE)
            self.model.compile(optimizer='adamax', loss=self.loss)
            self.train_model()
            if(self.generate_t): 
               print("\n \n \n" + self.generate(start_string= start_string))
            
            
            # Training the model with Saving/Loading checkpoints.
        def train_model(self):
            checkpoint_dir = './training_checkpoints'
            checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
            self.checkpoint_dir = checkpoint_dir
            #Loading ckpt
            try:
                self.model.load_weights(tf.train.latest_checkpoint(self.checkpoint_dir))
            except:
                print("couldnt load......")
            #Creating Checkpoints
            checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_prefix,
                save_weights_only=True)
            if(self.train): 
               history = self.model.fit(self.dataset, epochs=self.EPOCHS, callbacks=[checkpoint_callback])
            
            
            
            
        def split_input_target(self, chunk):
            input_text = chunk[:-1]
            target_text = chunk[1:]
            return input_text, target_text
        
        # Model Settings (Returns NN Model)
        def build_model(self, vocab_size, embedding_dim, rnn_units, batch_size):
            model = tf.keras.Sequential([
                tf.keras.layers.Embedding(vocab_size, embedding_dim, 
                                          batch_input_shape=[batch_size, None]),
                tf.keras.layers.GRU(rnn_units,
                                    return_sequences=True,
                                    stateful=True,
                                    recurrent_initializer='glorot_uniform'),
                tf.keras.layers.Dense(vocab_size)
                ])
            return model
        
        # NN Loss Function
        def loss(self, labels, logits):
            return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        
        # Function to generate text using the model (argument is the first char for the model to regenrate text RNN)
        def generate(self, start_string):
           model = self.build_model(self.vocab_size, self.embedding_dim, self.rnn_units,
                                    batch_size=1)
           model.load_weights(tf.train.latest_checkpoint(self.checkpoint_dir))
           model.build()
           
           input_eval = [self.char2idx[s] for s in start_string]
           input_eval = tf.expand_dims(input_eval, 0)
           
           text_generated = []
           
           #Networks Temperature
           temperature = 1.0
           #reset RNN 
           model.reset_states()
           for i in range(self.num_generate):
               predictions = model(input_eval)
               predictions = tf.squeeze(predictions,0)
               predictions = predictions / temperature
               predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
               
               input_eval = tf.expand_dims([predicted_id], 0)
               text_generated.append(self.idx2char[predicted_id])
               
           return(start_string + ''.join(text_generated))
           
#Number Of Runs
runs = 1

#Train Model Or Generate Text?
start_string = input("Enter Start String or (train): ")
if(start_string == "train"):
    runs = input("epochs: ")

#Calling The "Generate Text" Function With Initial Arguments
if __name__ == "__main__":
    Generate_Text(start_string, runs)



































