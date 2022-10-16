import igraph as ig
import numpy as np

def getBigGraph(Graphs):

    Na=len(Graphs)
    L1,L2=toLL(Graphs[0])
    Nnet=L2.size-1 

    if Na==1:
        return 1,Nnet,L1,L2

    #check maximum number of nodes
    if Na>1000:
        raise Exception('Too many graphs. Max 1000 graphs supported.')

    for g in Graphs[1:]:
        nL1,nL2=toLL(g)
        #chekc if grpahs have the same number of nodes
        if nL2.size-1!=Nnet:
            raise Exception('Graphs cannot have different number of nodes.')

        n=L2.size-1
        L1=np.concatenate((L1,nL1+n))
        L2=np.concatenate((L2,nL2[1:]+L2[-1]))

    return Na,Nnet,L1,L2



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

