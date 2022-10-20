import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
from time import time 
from . Utils import GraphManager as gm
from . Utils import ExceptionsManager as EM
import igraph as ig
from networkx import Graph as nxGraph



class NetModel:
    '''Class for a generic model of wealth exchange in a network'''
    
    def __init__(self,G,wmin=1e-17):
        '''G: igraph/ntworkx graph or list of igraph/networkx graphs
        wmin: minimum wealth an agent has to have to be able to transact'''


        EM.check_wmin(wmin)

        if type(G)==list:
            if type(G[0])==nxGraph:
                G=[ig.Graph.from_networkx(g) for g in G]
            Na,Nnet,L1,L2=gm.getBigGraph(G)


        elif type(G)==ig.Graph:
            L1,L2=gm.toLL(G)
            Nnet=L2.size-1
            Na=1

        #if its networkx graph convert to igraph
        elif type(G)==nxGraph:
            G=ig.Graph.from_networkx(G)
            L1,L2=gm.toLL(G)
            Nnet=L2.size-1
            Na=1

        else:
            raise Exception('Unknown graph type. Use igraph or networkx graph.')

        if Nnet>1024:
            raise Exception('Graphs cannot be bigger than 1024 nodes for GPU compatibility')
            
        self.__Nnet=Nnet
        self.__Na=Na
        self.__tL1=L1
        self.__tL2=L2
        self._N=L2.size-1
        self.__N=L2.size-1
        self.__wmin=wmin
        self.__threadsperblock=1024
        self.__blockspergrid=self.__Na

        #Alocamos memoria en la GPU para riquezas, riesgos, listas, y semaforos i y j
        self.__d_Nwealths=cuda.device_array(self.__N,dtype=np.float32)
        self.__d_Nwi=cuda.device_array(self.__N,dtype=np.float32)
        self.__d_L1=cuda.device_array(self.__tL1.size,dtype=np.int32)
        self.__d_L2=cuda.device_array(self.__tL2.size,dtype=np.int32)
        self.__d_SI=cuda.device_array(self.__N,dtype=np.int32)
        self.__d_SJ=cuda.device_array(self.__N,dtype=np.int32)

        #Disponemos los vecinos en la GPU (son fijas para todos los f)
        cuda.to_device(self._NetModel__tL1.astype(np.int32),to=self._NetModel__d_L1)
        cuda.to_device(self._NetModel__tL2.astype(np.int32),to=self._NetModel__d_L2)

        #semaforos en estado inicial
        SI=np.ones(self.__N,dtype=np.int32)
        cuda.to_device(SI,to=self.__d_SI)
        SJ=np.ones(self.__N,dtype=np.int32)
        cuda.to_device(SJ,to=self.__d_SJ)

        #Alocamos estado para números aleatoprios
        self.__rng_states = create_xoroshiro128p_states(self.__threadsperblock*self.__blockspergrid, seed=time())

    
    @property
    def wmin(self):
        '''Return the minimum wealth to transact'''
        return self.__wmin

    @wmin.setter
    def wmin(self,wmin):
        '''Set the minimum wealth to transact'''
        EM.check_wmin(wmin)
        self.__wmin=wmin

    @property
    def wealths(self):
        '''Return the wealths of the agents'''
        return self.__d_Nwealths.copy_to_host()

    @wealths.setter
    def wealths(self,W):
        '''Set the wealths of the agents'''
        if len(W)!=self.__N:
            raise Exception('Wealths must be a list of length N')

        EM.check_wealths(W)
        
        cuda.to_device(W.astype(np.float32),to=self.__d_Nwealths)

    def set_wealth_by_idx(self,A,w):
        '''Set the wealth of the agents indexed by A to w
        A: indexes of the agents Ex: [1,2,3]
        w: wealth to set or array of wealths Ex: 0.1 or [0.1,0.2,0.3]''' 
        
        EM.check_wealths(w)

        Nwealths=self.__d_Nwealths.copy_to_host()
        Nwealths[A]=w
        Nwealths=Nwealths.astype(np.float32)
        cuda.to_device(Nwealths,to=self.__d_Nwealths)

    def get_graphSize(self) -> tuple:
        '''Return a tuple with the number of graphs and the number of agents per graph'''
        return (self.__Nnet,self.__Na)

    def get_graph(self):
        '''Return the graph of the model as igraph graph'''
        return gm.toGraph(self.__tL1,self.__tL2)

    def get_nxGraph(self):
        '''Return the graph of the model as networkx graph'''
        return gm.toGraph(self.__tL1,self.__tL2).to_networkx()

    def __getGraphs(self): #todo
        '''Return the graphs of the model as list of igraph graphs'''
        #return gm.getGraphs(self.__tL1,self.__tL2)
        pass

    def __modifyGraph(self,G):
        '''Modify the graph of the model'''
        pass
      
    def reset(self,wealth_type):

        if wealth_type=='uniform':
            Nwealths=np.random.uniform(0,1,self._N)
        elif wealth_type=='equal':
            Nwealths=np.ones(self._N)
        else:
            raise Exception('''Unsupported wealth type. Use 'uniform' or 'equal'.''')

        #Set the risks and wealths
        for l in range(self._NetModel__Na):
            Nwealths[l*self._NetModel__Nnet:(l+1)*self.__Nnet]=Nwealths[l*self.__Nnet:(l+1)*self.__Nnet]/np.sum(Nwealths[l*self.__Nnet:(l+1)*self.__Nnet])
        Nwealths=Nwealths.astype(np.float32)

        cuda.to_device(Nwealths,to=self.__d_Nwealths)

    def Gini(self):
        '''Return the Gini coefficient of the actual wealth distribution'''
        x=self.__d_Nwealths.copy_to_host()
        #divide x in self.__Na parts
        x=x.reshape(self.__Na,self.__Nnet)

        #sort each part
        sorted_x = np.sort(x,axis=1)
        #calculate the cumulative sum of each part
        cumx = np.cumsum(sorted_x, dtype=np.float32, axis=1)


        return (self.__Nnet + 1 - 2 * np.sum(cumx,axis=1) / cumx[:,-1]) / self.__Nnet

        