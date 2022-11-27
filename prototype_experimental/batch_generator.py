import ROOT
import numpy as np


class Generator:
    def __init__(self,batch_size, x_rdf, nevt):
        self.batch_size = batch_size
        self.x_rdf = x_rdf
        self.nevt =  nevt
        self.generator = ROOT.Generator_t()

    def generator_functor(self):
        '''
        Calls C++ Generator and returns numpy array.
        RDF -> RTensor -> C++ Array -> Numpy Array
        '''        
        batch = self.generator(self.batch_size, self.x_rdf, self.nevt)
        # print(batch)
        batch_shape = list(batch.GetShape()) ##RTensor shape in cppyy.gbl.std.vector<unsigned long>
        # print(batch_shape)
        batch_data = batch.GetData() ## RTensor Data in cppyy.LowLevelView
        # print(batch_data)
        batch_data.reshape((int(batch_shape[0]*batch_shape[1]),)) ## RTensor Data in cppyy.LowLevelView
        # print(np.asarray(batch_data))
        reshaped_batch_data = np.asarray(batch_data).reshape(batch_shape)
        # print("reshaped_batch_data", reshaped_batch_data)
        return reshaped_batch_data

x_rdf = ROOT.RDataFrame("sig_tree", "http://root.cern.ch/files/Higgs_data.root", ["jet1_phi", "jet1_eta", "jet1_pt", "jet2_phi", "jet2_pt"])

print("compiling Generator functor....")
ROOT.gInterpreter.ProcessLine('#include "batch_generator_functor.h"')

batch_size = 4
batch_num = 0
nevt = 16

generator_class = Generator(batch_size, x_rdf, nevt)

while True:
    batch = generator_class.generator_functor()
    if (len(batch) == 0):
        break
    print("Batch No.", batch_num)
    print("Generator: ", batch)
    batch_num+=1