import numpy as np
from numba import cuda
from .Kernels.kYS import gpu_MCS,gpu_MCSfollow,gpu_MCSepoch
from .Model import NetModel
import warnings


class YSNetModel(NetModel):
    '''Class for a Yard Sale model on complex networks.
    Runs transactions in GPU using numba, multiple graphs can be used running in parallel.
    '''
    
    def __init__(self,G,f: float,wmin=1e-17):
        '''Create a new YS model with the given graph or list of graphs and f value
        G: igraph/ntworkx graph or list of igraph/networkx graphs
        f: social protection factor
        wmin: minimum wealth an agent has to have to be able to transact'''

        if f>0.5 or f<0:
            raise Exception('social protection factor f must be between 0 and 0.5')
        self.__f=f

        #Create the base network model of wealths 
        super().__init__(G,wmin)

        #Allocated GPU memory for risks
        self.__d_Nrisks=cuda.device_array(self._NetModel__N,dtype=np.float32)

        #Set default state for the model
        self.reset()

    def __str__(self) -> str:
        return f'Yard Sale model: \nGraph: {self._NetModel__Na} graphs of {self._NetModel__Nnet} agents \nSocial protection factor f: {self.__f}\nMinimum wealth to transact: {self._NetModel__wmin}'
      

    @property
    def f(self):
        '''Return the social protection factor f'''
        return self.__f

    @f.setter
    def f(self,f):
        '''Set the social protection factor f'''
        if f>0.5 or f<0:
            raise Exception('social protection factor f must be between 0 and 0.5')
        self.__f=f

    @property
    def risks(self):
        '''Return the risks of the agents'''
        return self.__d_Nrisks.copy_to_host()

    @risks.setter
    def risks(self,R):
        '''Set the risks of the agents
        R: array of risks in the same order as the agents'''
        if len(R)!=self._N:
            raise Exception(f'Number of risks must be equal to the number of agents ({self._NetModel__N})')

        if np.any(R>1) or np.any(R<0):
            raise Exception('All risks must be between 0 and 1')

        cuda.to_device(R.astype(np.float32),to=self.__d_Nrisks)

    def set_risk_by_idx(self,A,r):
        '''Set the risk of the agents indexed by A to r
        A: indexes of the agents Ex: [1,2,3]
        r: risk to set or array of risks Ex: 0.1 or [0.1,0.2,0.3]''' 

        if np.any(r>1) or np.any(r<0):
            raise Exception('All risks must be between 0 and 1')

      
        Nrisks=self.__d_Nrisks.copy_to_host()
        Nrisks[A]=r
        Nrisks=Nrisks.astype(np.float32)
        cuda.to_device(Nrisks,to=self.__d_Nrisks)


    def reset(self,wealth_type='uniform',risk_type='hetereogeneous',r=0.1):
        '''
        Reset the model to random state in risks and wealths. 
        wealth_type: 'uniform' or 'equal'
        risk_type:  'hetereogeneous' or 'homogeneous'
        r: if risk_type is 'homogeneous' this is the risk for all agents
        '''
        super().reset(wealth_type)

        if risk_type=='hetereogeneous':
            Nrisks=np.random.uniform(0,1,self._N)
        elif risk_type=='homogeneous':
            Nwealths=np.ones(self._N)
            Nrisks=np.ones(self._N)*r
        else:
            raise Exception('''Unsupported risk type. Use 'hetereogeneous' or 'homogeneous'.''')

        Nrisks=Nrisks.astype(np.float32)    
        cuda.to_device(Nrisks,to=self.__d_Nrisks)
        

    def termalize(self,M : int):
        '''Termalize the model for M montecarlo steps
        M: number of montecarlo steps'''
        warnings.simplefilter('ignore')
        gpu_MCS[self._NetModel__blockspergrid,self._NetModel__threadsperblock](
        self._NetModel__d_Nwealths,self.__d_Nrisks,
        self._NetModel__d_SI,self._NetModel__d_SJ,
        self.__f,self._NetModel__wmin,self._NetModel__d_L1,
        self._NetModel__d_L2,self._NetModel__rng_states,M,
        self._NetModel__Nnet,self._NetModel__Na)
        cuda.synchronize()
        warnings.simplefilter('default')

    def epoch(self,M : int):
        '''Make an epoch of M montecarlo steps returning the mean temporal wealths in each agent
        M: number of montecarlo steps'''
        Nwi=np.zeros(self._NetModel__N)
        Nwi=Nwi.astype(np.float32)
        cuda.to_device(Nwi,to=self._NetModel__d_Nwi)
        warnings.simplefilter('ignore')
        gpu_MCSepoch[self._NetModel__blockspergrid,self._NetModel__threadsperblock](
        self._NetModel__d_Nwealths,self.__d_Nrisks,
        self._NetModel__d_SI,self._NetModel__d_SJ,
        self.__f,self._NetModel__wmin,
        self._NetModel__d_L1,self._NetModel__d_L2,
        self._NetModel__rng_states,M,self._NetModel__Nnet,self._NetModel__Na,self._NetModel__d_Nwi)
        cuda.synchronize()
        warnings.simplefilter('default')
        return self._NetModel__d_Nwi.copy_to_host()/M
    
    def follow(self,M : int ,agent : int):
        '''Make an epoch of M montecarlo steps returning the wealths of the agent in each step
        M: number of montecarlo steps
        agent: index of the agent'''
        Wi=np.zeros(M)
        Wi=Wi.astype(np.float32)
        d_Wi=cuda.device_array(M,dtype=np.float32)
        cuda.to_device(Wi,to=d_Wi)
        warnings.simplefilter('ignore')
        gpu_MCSfollow[self._NetModel__blockspergrid,self._NetModel__threadsperblock](self._NetModel__d_Nwealths,self.__d_Nrisks,self._NetModel__d_SI,self._NetModel__d_SJ,self.__f,self._NetModel__wmin,self._NetModel__d_L1,self._NetModel__d_L2,self._NetModel__rng_states,M,self._NetModel__Nnet,self._NetModel__Na,d_Wi,agent)
        cuda.synchronize()
        warnings.simplefilter('default')
        d_Wi.copy_to_host(Wi)
        del d_Wi

        return Wi

   