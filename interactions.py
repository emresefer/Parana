import itertools
from util import *

class KeyObj(object) :
    '''
    Holds a key for entries into the dynamic programming
    table.  Currently, the key is an undirected edge, along
    with a value for the flip function.
    '''
    def __init__( self, u, v, f, r) :
        self._tup = (u,v) if u < v else (v,u)
        self._fw = f
        self._rev = r

    def __hash__(self) :
        frtup = (self._fw, self._rev)
        return (self._tup.__hash__() + frtup.__hash__())

    def __eq__(self, other) :
        if type(other) is str :
            return False
        else :
            return self._tup[0] == other._tup[0] and self._tup[1] == other._tup[1] and self._fw == other._fw and self._rev == other._rev

    def __str__(self):
        return "({0},{1},{2},{3})".format(self._tup[0], self._tup[1], self._fw, self._rev)

    def __repr__(self):
        return self.__str__()

    def arity(self) :
        return 1 if self._tup[0] == self._tup[1] else 2

def flipForward( key ) :
    u, v = key._tup
    f, r = key._fw, key._rev
    return KeyObj( u, v, 1-f, r )

def flipReverse( key ) :
    u, v = key._tup
    f, r = key._fw, key._rev
    return KeyObj( u, v, f, 1-r )

def flipBoth( key ) :
    u, v = key._tup
    f, r = key._fw, key._rev
    return KeyObj( u, v, 1-f, 1-r )

class BlockingLoop(object) :
    def __init__( self, verts, nonTreeEdges ) :
        self._verts = verts
        self._elist = [ (verts[i], verts[i+1]) for i in xrange(len(verts)-1)] + [ (verts[-1],verts[0]) ]
        edges = set( self._elist )
        nte = set([ (e[0], e[1]) for e in nonTreeEdges.keys() ] + [ (e[1],e[0]) for e in nonTreeEdges.keys() ])
        self._offendingEdges = set( filter( lambda x: x[0] != x[1], nte & set(self._elist) ) )
        
    def size(self) :
        return len(self._offendingEdges)

    def offendingEdges(self):
        return self._offendingEdges

    def __str__(self):
        s = "Blocking Loop : {"
        s += ", ".join( [ "{0} => {1}".format(v0, v1) for v0, v1 in self._elist] )
        s += "}"
        return s


def isInvalidCycle(l, treeEdges, nonTreeEdges):
    '''
    To be an invalid cycle, the cycle must contain at least one tree
    edge, and no two consecutive edges can be non-tree edges.
    '''

    # must contain at least one tree edge to be invalid
    if not treeEdges.isdisjoint(l):
        
        # if any consecutive pair of edges in the cycle
        # are nonTree edges, then it is not a blocking loop
        for x,y in pairs(l):
            if x in nonTreeEdges and y in nonTreeEdges:
                return False
        # If the loop contains at least one tree edge,
        # and no two consecutive edges are non-tree edges
        # then it is a blocking loop.
        return True
    else: # cycles of all non-tree edges are valid
        return False
        
    
def isBlockingLoop(l, T, rv=None) :
    '''
    Checks if the loop l is blocking according to the duplications
    indicated in the tree T. For a loop to be a blocking loop, it must
    have the following structure x0, a(x1), x1, a(x2), . . . , a(xk),
    xk, a(x0), where a(x) denotes a node that is an ancestor of
    x. This means that, starting from some node x0 in the loop, the
    next node must be an ancestor of another node, which must again
    link to the ancestor of yet another node, and so forth, until we
    return to the ancestor of our initial node x0.  Because we don't
    know where the loop begins, we must check both the initial ordering
    and a single "shift" to assure that we detect such a cycle.
    '''
    def checkBlocking(loop) :
        n = len(loop)
        startNode = loop[0]
        for i in xrange(1,n-1,2) :
            j = i + 1
            pj = set(pathToRoot(loop[j], rv, T )) - set([loop[j]])
            # if i is not an ancestor of j, then
            # this cannot be a blocking loop
            if loop[i] not in pj :
                return False

        # check that the last node in the loop
        # is an ancestor of the first
        ps = set(pathToRoot(startNode, rv, T)) - set([startNode])
        if loop[-1] not in ps :
            return False
        else :
            return True

    # if the user didn't pass in a root vertex, then find one
    if rv == None : rv = findRoot(T)

    # must check two initial offsets
    for offset in [0, 1] :
        loop = l[offset:] + l[:offset]
        if checkBlocking(loop) :
            return True
    return False
 
