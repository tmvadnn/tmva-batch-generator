import ROOT
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Flatten
import tensorflow as tf

class Generator:
    def __init__(self,batch_size, x_rdf, y_rdf, nevt):
        self.batch_size = batch_size
        self.x_rdf = x_rdf
        self.y_rdf = y_rdf
        self.nevt =  nevt
        self.x_generator = ROOT.Generator_t()
        self.y_generator = ROOT.Generator_t()

    def generator_batches(self):
        '''
        Calls C++ Generator and returns numpy array.
        RDF -> RTensor -> C++ Array -> Numpy Array
        '''        

        ## x_batch creation
        x_batch = self.x_generator(self.batch_size, self.x_rdf, self.nevt)
        # print(batch)
        x_batch_shape = list(x_batch.GetShape()) ##RTensor shape in cppyy.gbl.std.vector<unsigned long>
        # print(batch_shape)
        x_batch_data = x_batch.GetData() ## RTensor Data in cppyy.LowLevelView
        # print(batch_data)
        x_batch_data.reshape((int(x_batch_shape[0]*x_batch_shape[1]),)) ## RTensor Data in cppyy.LowLevelView
        # print(np.asarray(batch_data))
        x_reshaped_batch_data = np.asarray(x_batch_data).reshape(x_batch_shape)
        # print("x_reshaped_batch_data", x_reshaped_batch_data)

        ## y_batch creation
        y_batch = self.y_generator(self.batch_size, self.y_rdf, self.nevt)
        # print(batch)
        y_batch_shape = list(y_batch.GetShape()) ##RTensor shape in cppyy.gbl.std.vector<unsigned long>
        # print(batch_shape)
        y_batch_data = y_batch.GetData() ## RTensor Data in cppyy.LowLevelView
        # print(batch_data)
        y_batch_data.reshape((int(y_batch_shape[0]*y_batch_shape[1]),)) ## RTensor Data in cppyy.LowLevelView
        # print(np.asarray(batch_data))
        y_reshaped_batch_data = np.asarray(y_batch_data).reshape(y_batch_shape)

        # if len(x_reshaped_batch_data) == 0:
        #     return None, None

        return x_reshaped_batch_data,y_reshaped_batch_data

x_rdf = ROOT.RDataFrame("bkg_tree", "./Higgs_data.root", ["jet1_phi", "jet1_eta", "jet1_pt", "jet2_phi", "jet2_pt"])
y_rdf = ROOT.RDataFrame("bkg_tree", "./Higgs_data.root", ["jet1_phi", "jet1_eta", "jet1_pt", "jet2_phi", "jet2_pt"])

print("compiling Generator functor....")
ROOT.gInterpreter.ProcessLine('#include "batch_generator_functor.h"')

batch_size = 4
batch_is_empty = 0
batch_num = 0
nevt = 16

generator_class = Generator(batch_size, x_rdf, y_rdf, nevt)

model = Sequential()
model.add(Dense(12, input_dim=3, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.summary()
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


model.fit(generator_class.generator_batches(),steps_per_epoch=100/batch_size ,epochs=20)