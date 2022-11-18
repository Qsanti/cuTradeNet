import numpy as np
from numba import cuda
from .Kernels.kC import gpu_MCS,gpu_MCSfollow,gpu_MCSepoch
from .Model import NetModel
from . Utils import ExceptionsManager as EM
import warnings


class DYNetModel(NetModel):
    '''Class for a Drăgulescu and Yakovenko model on complex networks.
    Runs transactions in GPU using numba, multiple graphs can be used running in parallel.
    '''
    
    def __init__(self,G,wmin=1e-17):
        '''Create a new Constant trade model with the given graph or list of graphs and f value
        G: igraph/ntworkx graph or list of igraph/networkx graphs
        wmin: minimum wealth an agent has to have to be able to transact'''
        

        #Create the base network model of wealths 
        super().__init__(G,wmin)

        #Set default state for the model
        self.reset()

    def __str__(self) -> str:
        return f'Drăgulescu and Yakovenko model: \nGraph: {self._NetModel__Na} graphs of {self._NetModel__Nnet} agents \nMinimum wealth to transact: {self._NetModel__wmin}'
    


    def reset(self,wealth_type='uniform'):
        '''
        Reset the model to random state in risks and wealths. 
        wealth_type: 'uniform' or 'equal'
        '''
        super().reset(wealth_type)
        

    def termalize(self,M : int):
        '''Termalize the model for M montecarlo steps
        M: number of montecarlo steps'''

        EM.check_MCS(M)
        warnings.simplefilter('ignore')
        gpu_MCS[self._NetModel__blockspergrid,self._NetModel__threadsperblock](
        self._NetModel__d_Nwealths,
        self._NetModel__d_SI,self._NetModel__d_SJ,
        self._NetModel__wmin,
        self._NetModel__d_L1, self._NetModel__d_L2,
        self._NetModel__rng_states,M,
        self._NetModel__Nnet,self._NetModel__Na)
        cuda.synchronize()
        warnings.simplefilter('default')

    def epoch(self,M : int):
        '''Make an epoch of M montecarlo steps returning the mean temporal wealths in each agent
        M: number of montecarlo steps'''

        EM.check_MCS(M)
        Nwi=np.zeros(self._NetModel__N)
        Nwi=Nwi.astype(np.float32)
        cuda.to_device(Nwi,to=self._NetModel__d_Nwi)
        warnings.simplefilter('ignore')
        gpu_MCSepoch[self._NetModel__blockspergrid,self._NetModel__threadsperblock](
        self._NetModel__d_Nwealths,
        self._NetModel__d_SI,self._NetModel__d_SJ,
        self._NetModel__wmin,
        self._NetModel__d_L1,self._NetModel__d_L2,
        self._NetModel__rng_states,M,
        self._NetModel__Nnet,self._NetModel__Na,
        self._NetModel__d_Nwi)
        cuda.synchronize()
        warnings.simplefilter('default')
        return self._NetModel__d_Nwi.copy_to_host()/M
    
    def follow(self,M : int ,agent : int):
        '''Make an epoch of M montecarlo steps returning the wealths of the agent in each step
        M: number of montecarlo steps
        agent: index of the agent'''

        EM.check_MCS(M)
        Wi=np.zeros(M)
        Wi=Wi.astype(np.float32)
        d_Wi=cuda.device_array(M,dtype=np.float32)
        cuda.to_device(Wi,to=d_Wi)
        warnings.simplefilter('ignore')
        gpu_MCSfollow[self._NetModel__blockspergrid,self._NetModel__threadsperblock](
        self._NetModel__d_Nwealths,
        self._NetModel__d_SI,self._NetModel__d_SJ,
        self._NetModel__wmin,
        self._NetModel__d_L1,self._NetModel__d_L2,
        self._NetModel__rng_states,M,
        self._NetModel__Nnet,self._NetModel__Na,
        d_Wi,agent)
        cuda.synchronize()
        warnings.simplefilter('default')
        d_Wi.copy_to_host(Wi)
        del d_Wi

        return Wi

   