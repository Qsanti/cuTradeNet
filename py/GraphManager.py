import igraph as ig
import numpy as np

def getBigGraph(AA,path):
    
    Ai=AA[0] #iniciamos con una red

    #Cargamos las putilistas de adjacency
    L1=np.load(path+f'Graphs/L1{int(Ai)}.npy')
    L2=np.load(path+f'Graphs/L2{int(Ai)}.npy')

    #calculamos el n de las redes y la cantida de ellas
    Nnet=L2.size-1 
    Na=len(AA)

    #Unimos las listas de adyacencia de las demás redes
    for Ai in AA[1:]:
        nL1=np.load(path+f'Graphs/L1{int(Ai)}.npy')
        nL2=np.load(path+f'Graphs/L2{int(Ai)}.npy')
        n=L2.size-1
        L1=np.concatenate((L1,nL1+n))
        L2=np.concatenate((L2,nL2[1:]+L2[-1]))

    return L1,L2



def graph_enseble(L1,L2,n=10):
    Nnet=L2.size-1

    eL1=L1.copy()
    eL2=L2.copy()

    for i in range(n-1):
        Nnet=eL2.size-1
        eL1=np.concatenate((eL1,L1+Nnet))
        eL2=np.concatenate((eL2,L2[1:]+eL2[-1]))

    return eL1,eL2



def toGraph(L1,L2):
    g=ig.Graph()
    g.add_vertices(L2.size-1)
    for i in range(L2.size-1):
        for j in L1[L2[i]:L2[i+1]]:
            if(i<1000 and j>1000):
                print(i,j)
            if(i>j):
                g.add_edge(i,j)
    return g


def toLL(g):
    L1=[]
    L2=[0]
    for i in range(g.vcount()):
        L1.extend(g.neighbors(i))
        L2.append(len(L1))
    return np.array(L1),np.array(L2)

class DinamicGraph:
    def __init__(self,L1,L2,Nnet=1000,kmax=10):
        self.L1=L1
        self.L2=L2
        self.L10=L1.copy()
        self.L20=L2.copy()
        self.Nnet=Nnet
        self.N=L2.size-1
        self.kmax=kmax

    def reset(self):
        self.L1=self.L10.copy()
        self.L2=self.L20.copy()

    def getLs(self):
        return self.L1.copy(),self.L2.copy()

    def getGraph(self):
        return toGraph(self.L1,self.L2)

    def getNeigh(self,i):
        return self.L1[self.L2[i]:self.L2[i+1]]

    def getK(self,i):
        return self.L2[i+1]-self.L2[i]

    def getsK(self,i):
        k=self.L2[i+1]-self.L2[i]
        if k<self.kmax:
            return k
        return self.kmax-1

    def getsKs(self,I):
        return np.array([self.getsK(i) for i in I])

    def take_action(self,i,dk):
        "Ejecuta la accion tomando tomando el agente y parametros"

        if dk==0:
            if self.L2[i+1]-self.L2[i]==0:
            #No hay vecinos
                return 0

            #seleccionamos un vecino al azar
            j=self.L1[self.L2[i]:self.L2[i+1]][np.random.randint(self.L2[i+1]-self.L2[i])]

            self.L1=np.delete(self.L1,self.L2[i]+np.argmax(self.L1[self.L2[i]:self.L2[i+1]]==j))
            self.L2[i+1:]-=1
            self.L1=np.delete(self.L1,self.L2[j]+np.argmax(self.L1[self.L2[j]:self.L2[j+1]]==i))
            self.L2[j+1:]-=1

       

            return j
            
        if dk==1:

            if self.L2[i+1]-self.L2[i]==self.kmax-1:
            #saturo
                return self.kmax



            while True:
                j=np.random.randint(0,self.Nnet)+i//self.Nnet*self.Nnet
                if j not in self.L1[self.L2[i]:self.L2[i+1]]:
                    break

            self.L1=np.insert(self.L1,self.L2[i+1],j)
            self.L2[i+1:]=self.L2[i+1:]+1
            self.L1=np.insert(self.L1,self.L2[j+1],i)
            self.L2[j+1:]+=1



            return j