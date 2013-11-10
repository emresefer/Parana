#! /usr/bin/env python

'''
PARANA is open-source software which is free for personal and academic
use.  It is licensed under a Creative Commons license with commercial
usage and attribution restrictions.  Full details of the license can
be found in the file LICENSE.txt.
'''

# Python modules
import argparse
import random
import itertools
import cPickle
import heapq
import logging
from collections import deque, OrderedDict

# 3rd party modules
import cogent
import networkx as nx

# Our modules
# from my_simple_cycles import my_simple_cycles
import first_cycle
from interactions import KeyObj, flipForward, flipReverse, flipBoth, BlockingLoop, isBlockingLoop
from util import allPairs, swap, edgeCost, treeToGraph, pathToRoot, pairs, findRoot, prepareTree
import evaluation

## Simple convenience functions for pairty checks
isOdd = lambda x : (x % 2) != 0
isEven = lambda x : not isOdd(x)

## Are the target networks directed or undirected ?
undirected = False 

## Relative creation and deletion costs
cc = 1.0
dc = 1.0

def leaves(T):
    """
    Return a list of all of the leaves in the tree T.
    """
    return [ n for n in T.nodes_iter() if len(T.successors(n)) == 0 ]


def sol(k, T, slnDict, rl, constraints, selfLoops, d) :
    '''
    In accordance with Guillaume's most recent recurrence, call the apropriate
    version of the solution function based on the arity of the argument.
    '''
    if k.arity() == 1 :
        return solArity1(k, T, slnDict, rl, constraints, selfLoops, d)
    else :
        return solAirty2(k, T, slnDict, rl, constraints, selfLoops, d)

def solArity1(k, T, slnDict, rl, constraints, selfLoops, d) :
    '''
    S(u, f) = S(Lu,f) + S(Ru,f) + S(Lu, Ru, f)
    '''
    # convenience functions
    # check the node type
    isInternal = lambda n : len(T.successors(n)) == 2
    isLeaf = lambda n : not isInternal(n)
    
    u, v, f, r = k._tup[0], k._tup[1], k._fw, k._rev

    # Don't recompute a previously stored solution
    if k in slnDict :
        return slnDict[k]

    # The subtrees into which we will recurse
    try:
        LU, RU = T.successors(u)
    except ValueError:
        logging.info("can't descend into {0}".format(u))
        logging.info("means I didn't have key {0}".format(k))

    noLoopL, noLoopR, noLoopLR = KeyObj( LU, LU, f, r ), KeyObj( RU, RU, f, r), KeyObj(LU, RU, f, r)
    loopL, loopR, loopLR = flipBoth(noLoopL), flipBoth(noLoopR), flipBoth(noLoopLR)
    
    # Here are the recursive options
    costFlipNone = 0.0
    costFlipBoth = cc if f + r == 0 else dc 
    recOptions = []
    if selfLoops : recOptions.append( (costFlipBoth, loopL, loopR, loopLR, '##fb') )
    recOptions.append((costFlipNone, noLoopL, noLoopR, noLoopLR, '##fn'))

    # Compute the cost of each
    cases = {}
    for cost, leftRec, rightRec, selfRec, caseKey in recOptions :
        costLU, costLV = sol(leftRec, T, slnDict, rl, constraints, selfLoops, d+1), sol(rightRec, T, slnDict, rl, constraints, selfLoops, d+1) 
        costLULV = sol(selfRec, T, slnDict, rl, constraints,selfLoops, d+1)
        cases[caseKey] = (cost + costLU + costLV + costLULV, leftRec, rightRec, selfRec)

    # The solution to this problem is the minimum over all
    # of the evaluated subproblems
    minSln = ['', float('inf'), None, None, None]
    for caseKey, caseValue in cases.iteritems():
        cost, leftChild, rightChild, selfRec = caseValue
        if cost < minSln[1]:
            minSln = [caseKey, cost, leftChild, rightChild, selfRec]

    # Put the minimum solution in the table
    slnDict[ k ] = minSln[ 1 ]

    # Put the backtracking solution in the table
    if k not in rl:
        whichFlip = minSln[0][2:4]      
        rl[k] = (whichFlip, minSln[2], minSln[3], minSln[4])

    return slnDict[k] 

def solAirty2(k, T, slnDict, rl, constraints, selfLoops, d) : 
    """
    Solves the dynamic programming recurrence in the tree T for subtrees
    u and v, and flip function f.  The slnDict parameter is assumed, already,
    to be populated with the solutions to the necessary subproblems.
    """
    directed = not undirected

    # convenience functions check the node type
    isInternal = lambda n : len(T.successors(n)) == 2
    isLeaf = lambda n : not isInternal(n)
    
    u, v, f, r = k._tup[0], k._tup[1], k._fw, k._rev

    # Don't re-compute the solution to an already-solved 
    # subproblem
    if k in slnDict :
        return slnDict[k]
    
    # Tests if a given edge (or it's inverse) is forbidden by constraints
    respectsConstraints  = lambda u,v : not ( (u,v) in constraints or (v,u) in constraints )
    cases = {}

    # Compute the solutions of subproblems where we recurse on rnode
    def recurseOn( rnode, onode, prefix ) :
        if isInternal(rnode) :
            LRN, RRN = T.successors(rnode)
            noFlipL, noFlipR = KeyObj(LRN, onode, f, r), KeyObj(RRN, onode, f, r)
            dualFlipL, dualFlipR = flipBoth(noFlipL), flipBoth(noFlipR)

            noFlipCost = 0.0
            if undirected :
                # If we are flipping from of to on, we pay the creation cost. Otherwise we pay the deletion cost
                dualFlipCost = cc if f+r == 0 else dc
            else :
                # We pay the creation cost for any edge we turn on, and the deletion cost for any edge we turn off
                dualFlipCost = cc if f == 0 else dc
                dualFlipCost += cc if r == 0 else dc

            recOptions = [ (dualFlipCost, dualFlipL, dualFlipR, prefix+'fb'), (noFlipCost, noFlipL, noFlipR, prefix+'fn') ]

            if directed: 
                # We pay the cost for creating an edge if none exists; otherwise we pay the cost for deleting an edge
                fwFlipCost = cc if f == 0 else dc
                revFlipCost = cc if r == 0 else dc
                # In the directed case we add the recursions for the directed edges
                fwFlipL, fwFlipR = flipForward(noFlipL), flipForward(noFlipR)
                revFlipL, revFlipR = flipReverse(noFlipL), flipReverse(noFlipR)
                recOptions += [(fwFlipCost, fwFlipL, fwFlipR, prefix+'ff'), (revFlipCost, revFlipL, revFlipR, prefix+'fr')]
            ctr = 0
            for cost, leftRec, rightRec, caseKey in recOptions :
                flipCase = caseKey[-2:]
                if respectsConstraints( rnode, onode ) or flipCase == 'fn': 
                    ctr += 1
                    cases[caseKey] = (cost + sol(leftRec, T, slnDict, rl, constraints, selfLoops, d+1) +\
                                          sol(rightRec, T, slnDict, rl, constraints, selfLoops, d+1), leftRec, rightRec)
            if ctr == 0 : raise Exception( "Couldn't follow any sub-problem for {0}!".format((rnode, onode)) )
                
    recurseOn(u, v, 'ru')
    recurseOn(v, u, 'rv')
   
    # The solution to this problem is the minimum over all
    # of the evaluated subproblems
    c = 0 if differentExtantNetworks(T,u,v) else float('inf')
    minSln = ['####', c, None, None]
    for caseKey, caseValue in cases.iteritems():
        cost, leftChild, rightChild = caseValue
        if cost < minSln[1]:
            minSln = [caseKey, cost, leftChild, rightChild]

    # Put the minimum solution in the table
    slnDict[k] = minSln[ 1 ]

    # Put the backtracking solution in the table
    if k not in rl :
        whichFlip = minSln[0][2:4]      
        rl[k] = (whichFlip, minSln[2], minSln[3])

    return slnDict[k]

 
def initf(T, rv, G, constraints):
    '''
    Given the duplication tree T, the root vertex rv, the extant graph G and
    the constraints, fill in the base case of the recurrence (i.e. the parsimony
    score of entering any pair of leaves in any of the potential states of f).
    '''
    rl = {}
    slnDict = {}
    inf = float('inf')
    directed = not undirected
    for u,v in allPairs( T.node[rv]['leaves'] ):
        if differentExtantNetworks(T, u, v):
            continue
        fn = KeyObj(u,v,0,0)
        fb = KeyObj(u,v,1,1)
        if directed: 
            fr = KeyObj(u,v,0,1)
            ff = KeyObj(u,v,1,0)
        
        # If u and v are different protein
        if u != v :
            d_f = 1 if G.has_edge(u,v) else 0
            d_r = 1 if G.has_edge(v,u) else 0 
            # If the nodes have not been lost, then assign the regular costs
            if u in G.nodes() and v in G.nodes() :
                if undirected:
                    slnDict[fn] = 0 if d_f + d_r == 0 else cc 
                    slnDict[fb] = 0 if d_f + d_r == 2 else dc 
                else: 
                    slnDict[fn] = d_f * cc + d_r * cc
                    slnDict[fb] = (1 - d_f) * dc + (1 - d_r) * dc
                    slnDict[fr] = d_f * cc + (1 - d_r) * dc
                    slnDict[ff] = (1 - d_f) * dc + d_r * cc
        else :
            hasSelfLoop = G.has_edge(u,v)
            if u in G.nodes() and v in G.nodes() :
                # A self loop always costs cc
                slnDict[fn] = cc if hasSelfLoop else 0 
                slnDict[fb] = 0 if hasSelfLoop else dc
                if directed:
                    slnDict[fr] = 0 if hasSelfLoop else dc
                    slnDict[ff] = 0 if hasSelfLoop else dc

        if not( u in G.nodes() and v in G.nodes() ):
            # Costs to lost nodes are always 0
            lostCost = 0
            slnDict[fn] = lostCost; slnDict[fb] = lostCost
            if directed: 
                slnDict[fr] = lostCost; slnDict[ff] = lostCost; 

        ## The base cases for the optimal solutions
        if slnDict[fn] == 0 :
            rl[fn] = ('fn', None, None)
            rl[fb] = ('fb', None, None)
            if directed:
                rl[ff] = ('ff', None, None)
                rl[fr] = ('fr', None, None)

        if directed and slnDict[ff] == 0 :
            rl[ff] = ('fn', None, None)
            rl[fr] = ('fb', None, None)
            rl[fn] = ('ff', None, None)
            rl[fb] = ('fr', None, None)

        if directed and slnDict[fr] == 0 :
            rl[fn] = ('fr', None, None)
            rl[ff] = ('fb', None, None)
            rl[fr] = ('fn', None, None)
            rl[fb] = ('ff', None, None)

        if slnDict[fb] == 0 :
            rl[fn] = ('fb', None, None)
            rl[fb] = ('fn', None, None)            
            if directed:
                rl[ff] = ('fr', None, None)
                rl[fr] = ('ff', None, None)

    return slnDict, rl

def differentExtantNetworks(T, u, v) :
    '''
    Are the descendants of u and v strictly in different extant networks
    '''
    return (u != v) and len( T.node[u]['enets'] & T.node[v]['enets'] ) == 0

def gatherNonTreeEdges(T, rootKey, optPath, slnDict) :
    '''
    Use optPath to recover the non-tree edges added to T starting with rootKey
    '''
    directed = not undirected
    nonTreeEdges = {}
    explored = set([])
    cnt = 0
    toProcess = [ rootKey ] 
    while len(toProcess) > 0 :
         
        cnode = toProcess.pop()
        u, v, f, r = cnode._tup[0], cnode._tup[1], cnode._fw, cnode._rev
                
        edge = (u,v)

        # If we've already looked at this guy, don't do it again
        if edge in explored:
            continue
        explored.add( edge )

        try :
            sln = optPath[ cnode ]
            cnt += 1
        except KeyError as e:
            if differentExtantNetworks(T,u,v) :
                # It's okay not to have a key in this case; just continue
                continue
            else :
                # Otherwise this should not happen . . . error
                logging.error("NO KEY {0}".format(cnode))
                exit(0)

        addedEdge = False
        addedSelfLoop = False

        if u != v :
            if sln[0] == 'fb' : nonTreeEdges[edge] = 'b'
            if directed:
                if sln[0] == 'ff' : nonTreeEdges[edge] = 'f' 
                if sln[0] == 'fr' : nonTreeEdges[edge] = 'r' 
        else :
            if sln[0] == 'fb' : nonTreeEdges[edge] = 'b'; addedSelfLoop = True

        for slnc in xrange(1,len(sln)):
            if sln[slnc] != None:
                child = sln[slnc]
                toProcess.append( child )
       
    return nonTreeEdges

def impliedEdges(u, v, rv, T, nonTreeEdges) :
    '''
    Given the set of non-tree edges (nonTreeEdges), the duplication history (T), the root vertex (rv),
    and a pair of nodes u and v, determine the set of edges between u and v (i.e. forward, reverse, etc.)
    implied by the tree topology and the set of nonTreeEdges.
    '''
    pa = pathToRoot(u, rv, T)
    pb = pathToRoot(v, rv, T)
    pab = 0
    pba = 0

    # Count non-tree edges
    if u != v :
        for a in pa:
            for b in pb:
                edge = (a,b) if a < b else (b,a)
                if edge in nonTreeEdges: 
                    dir = nonTreeEdges[edge]
                    if dir == 'f' : pab += 1
                    elif dir == 'r' : pba += 1
                    elif dir == 'b' : pab += 1; pba += 1

        edges = [ ]
        # "forward" edges
        if isOdd(pab) :
            edges.append( (u,v) )
        # "reverse" edges
        if isOdd(pba) :
            edges.append( (v,u) )

    ## Count self loops
    else :
        pa = pathToRoot(u, rv, T)
        paa = 0
        for a in pa :
            edge = (a,a)
            if edge in nonTreeEdges :
                dir = nonTreeEdges[edge]
                if dir == 'b' : paa += 1
        edges = [ (u,v) ] if isOdd(paa) else []

    return edges


def parRecon( G, T, rv, lostNodes, constraints ):
    '''
    Run the main parsimony reconstruction algorithm on the input and return
    the most parsimonious solution.
    '''
    directed = not undirected
    lc, rc = T.successors(rv)[0], T.successors(rv)[1]
    
    kn, kb, kf, kr = KeyObj(rv, rv, 0, 0), KeyObj(rv, rv, 1, 1), KeyObj(lc, rc, 1, 0), KeyObj(lc, rc, 0, 1)
    slnDictG, optPathG = initf(T, rv, G, [])

    # Hash from the key to the non-tree edges induced by this choice
    # If GU == GV we only consider no flips or a self loop
    GKeys = OrderedDict()
    
    # At the top level, we either create a self loop or don't
    GKeys[kn] = None; GKeys[kb] = None

    # Hash holds the solution dictionary and optimal path backtracking
    # for all initial conditions
    GOpts = OrderedDict()
    SolCosts = OrderedDict()
    # Cost of a self loop is cc, cost of no self loop is 0
    SolCosts[kn] = 0; SolCosts[kb] = cc

    # Solution dictionaries for two top-level options
    GOpts[kn]={ 'sln' : slnDictG, 'opt' : optPathG }
    GOpts[kb]={ 'sln' : slnDictG, 'opt' : optPathG }; GOpts[kb]['opt'][kb] = ('fb', KeyObj(lc,lc,1,1), KeyObj(rc,rc,1,1), KeyObj(lc,rc,1,1))
 
    # Compute solutions for all initial conditions 
    for k in GKeys :
        SolCosts[k] += sol(k, T, GOpts[k]['sln'], GOpts[k]['opt'], constraints, True, 0) 

    # For every solution we computed, calculate its implied set of non-tree edges
    for k in GKeys :
        GKeys[k] = gatherNonTreeEdges(T, k, GOpts[k]['opt'], GOpts[k]['sln'])
        recon, ds = constructingHistory(T, rv, G, GKeys[k], lostNodes)
        if not len(ds) == 0:
            raise Exception("Computed a parsimonious history that does not reconstruct G! \nThe root key is {0} \nThe differences are {1}".format(k,ds) )

    parSln, slnCost = None, float('inf')
    for sln, cost in SolCosts.iteritems():
        if cost <= slnCost:
            slnCost = cost
            parSln = sln
            
    return slnCost, GKeys[parSln]


def impliedNetworkOn( X, T, rv, nte ) :
    '''
    Given a set of nodes representing a cut through the tree (X), the duplication history (T),
    the root vertex (rv) and the set of non-tree edges (nte), compute the network implied on the
    nodes of X.
    '''
    recon = nx.Graph() if undirected else nx.DiGraph()
    for u,v in allPairs(X.nodes()):
        e = impliedEdges(u,v,rv,T,nte)
        recon.add_edges_from( [ edge for edge in e if not differentExtantNetworks(T, u, v) ] )
    return recon

def constructingHistory(T, rv, G, nte, lostNodes):
   '''
   Test if the given set of non-tree edges (nte) reconstructs the extant network (G).
   This function computes the reconstructed newtork (recon) as well as the set of edges (ds)
   that are different between (G) and (recon).  The history given by nte reconstructs
   G <==> (len(ds) == 0)
   '''
   recon = nx.Graph() if undirected else nx.DiGraph()
   if undirected:
       G = G.to_undirected()
   for u,v in allPairs(G.nodes()):
       e = impliedEdges(u, v, rv, T, nte) 
       recon.add_edges_from(  [ edge for edge in e if isEffectiveEdge(T, u, v, lostNodes) ] )
       
   ds = set([])
   ds |= set( [ e for e in G.edges() if not recon.has_edge(e[0],e[1]) ] )
   ds |= set( [ e for e in recon.edges() if not G.has_edge(e[0],e[1]) ] )
   return recon, ds 

def isEffectiveEdge(T, u, v, lostNodes):
    '''
    An edge u,v in the tree is "effective" if and only if it is between
    two nodes in the same organism, and neither of the nodes are "LOST".
    '''
    return not differentExtantNetworks(T, u, v) and not u in lostNodes and not v in lostNodes

def main():
    '''
    Main method that sets up logging and creates and invokes the command line parser
    '''
    ## Set the logging level
    logging.basicConfig(level=logging.DEBUG)

    ## Parse the command line arguments and set the relevant
    ## program options.
    parser = createParser()
    options = parser.parse_args()
    options.func(options)
    
def inferMain( options ):
    '''
    Main method for the subcommand 'infer'.  Performs the parsimonious reconstruction by finding
    the minimum set of non-tree edges in the duplication history which explains the extant networks.
    '''
    ## The global variables we will need to set based on the
    ## command line arguments.
    global undirected
    global cc
    global dc

    undirected = options.undirected
    directed = not undirected
    dc = 1.0
    cc = options.cost_ratio

    # Read in the extant networks
    G = nx.read_adjlist( options.target, create_using = nx.Graph() if undirected else nx.DiGraph() )

    # Read in the gene tree and convert it to a NetworkX graph
    T = treeToGraph( cogent.LoadTree( options.duplications ) )
    rv = findRoot(T)

    # Compute some important sets of nodes from the extant network and the tree
    # specifically, the set of lost and extant nodes
    leaves = set( [n for n in T.nodes() if len(T.successors(n)) == 0] )
    lostNodes = set( filter( lambda x: x.find('LOST') != -1, leaves ) )
    extantNodes = leaves - lostNodes

    # Add back any isolated extant nodes (nodes which have no interactions) to the extant network
    isolatedNodes = extantNodes - set(G.nodes())
    if len(isolatedNodes) > 0:
        logging.info( "Adding isolated extant nodes {0} to the extant network".format( isolatedNodes ) )
        G.add_nodes_from( isolatedNodes ) 

    prepareTree(T, rv, lostNodes)

    blockingLoops = []
    constraints = []
    hasBlockingLoops = True
    SolutionGraph = nx.DiGraph()
    loopCtr = 0
    t = 0

    while (hasBlockingLoops) :
        
        if not len(blockingLoops) == 0:
            l = blockingLoops.pop()
            for e in l.offendingEdges():
                if not ( e[0] in leaves and e[1] in leaves ) :
                    constraints.append(e)
                    break
                
        logging.info("Round {0}".format(t))
        logging.info("CONSTRAINTS = {0}".format(constraints))
        logging.info("Computing maximally parsimonious network history . . . ")
        cost, nonTreeEdges = parRecon(G, T, rv, lostNodes, constraints)

        nonTreeEdges = { e:d for e,d in nonTreeEdges.iteritems() if isEffectiveEdge(T, e[0], e[1], lostNodes) }
        
        logging.info("Cost = {0}".format(cost))

        # Augment the tree with auxiliary edges pointing from a
        # node to all of its children
        if options.loop:
            augmentedT = T.copy()
            augmentedT.add_edges_from( [ (nte[0],nte[1]) for nte in nonTreeEdges ] + [ (nte[1],nte[0]) for nte in nonTreeEdges ] ) 
       
            # Take the first blocking loop
            cycle = first_cycle.find_cycle( augmentedT, T )
            blockingLoops = [ BlockingLoop(cycle, nonTreeEdges) ] if len(cycle) > 0 else []
            nbl = len(blockingLoops)
            hasBlockingLoops = nbl > 0 

            if hasBlockingLoops:
                logging.info("Fast cycle checker found blocking loop {0}".format(cycle) )
        else :
            hasBlockingLoops = False
        t += 1


    if options.nonTreeEdges is not None:
        with open(options.nonTreeEdges, 'wb') as ofile:
            for nte,d in nonTreeEdges.iteritems():
                u,v = nte
                # We only care about those non-tree edges that are between ancestors of extant nodes
                # which have not been lost
                ofile.write('{0}\t{1}\t{2}\n'.format(u,v,d))
                    
    exit(0)
 
def extractMain( options ):
    '''
    Given a duplication tree, a set of nodes, N, representing an ancestral network and
    a set of non-tree edges computed with parana's "infer" command, extract the ancestral
    network implied on N by the set of non-tree edges.
    '''
    ## The global variables we will need to set based on the
    ## command line arguments.
    global undirected

    undirected = options.undirected

    # Read in the gene tree and convert it to a NetworkX graph
    T = treeToGraph( cogent.LoadTree( options.duplications ) )
    rv = findRoot(T)
    leaves = set( [n for n in T.nodes() if len(T.successors(n)) == 0] )
    lostNodes = filter( lambda x: x.find('LOST') != -1, leaves )
    prepareTree(T, rv, lostNodes)

    # The cut through the duplication tree that defines the ancestral network's nodes
    cutNetwork = nx.Graph() if undirected else nx.DiGraph()
    with open(options.source) as ifile:
        for l in ifile:
            cutNetwork.add_node(l.rstrip())

    nonTreeEdges = {}
    # Read in the non-tree edges
    with open(options.nonTreeEdges,'rb') as ifile:
        for l in ifile:
            u, v, d = l.rstrip().split('\t')
            edge = (u,v) if u < v else (v,u)
            nonTreeEdges[ edge ] = d

    A = impliedNetworkOn( cutNetwork, T, rv, nonTreeEdges )
    nx.write_adjlist(A, options.output)
    #with open(options.output, 'wb') as ofile:
    #    for u,v in A.edges_iter():
    #        ofile.write("{0}\t{1}\n".format(u,v))

    exit(0)
    
def compareMain( options ):
    '''
    Compare a ground truth and reconstructed ancestral network
    '''
    ## The global variables we will need to set based on the
    ## command line arguments.
    global undirected

    undirected = options.undirected

    groundTruthAncestor = nx.read_adjlist( options.ground_truth,  create_using = nx.Graph() if undirected else nx.DiGraph() )
    reconstructedAncestor = nx.read_adjlist( options.recon, create_using = nx.Graph() if undirected else nx.DiGraph() )
 
    precision, recall, f1 = evaluation.calcStats(groundTruthAncestor, reconstructedAncestor)
    print("\nprec = {0}, rec = {1}, F1 = {2}".format(precision, recall, f1))

    if options.duplications:
        
        T = treeToGraph( cogent.LoadTree(options.duplications) )
        rv = findRoot(T)
        leaves = set( [n for n in T.nodes() if len(T.successors(n)) == 0] )
        lostNodes = set(filter( lambda x: x.find('LOST') != -1, leaves ))
        extantNodes = leaves - lostNodes

        prepareTree(T, rv, lostNodes)
        validNodes = filter( lambda x: len(T.node[x]['leaves'] & extantNodes) > 0, groundTruthAncestor.nodes_iter())

        groundTruthSubgraph = groundTruthAncestor.subgraph(validNodes)
        reconstructedSubgraph = reconstructedAncestor.subgraph(validNodes)
 
        precision, recall, f1 = evaluation.calcStats(groundTruthSubgraph, reconstructedSubgraph)
        print("Loss Free")
        print("prec = {0}, rec = {1}, F1 = {2}".format(precision, recall, f1))

        
    exit(0)
        
def createParser():
    '''
    This function creates the command line parser for parana.  It creates subparsers for the three main commands --- infer,
    extract and compare.  Based on the command given, it invokes the appropriate main method of parana with the provided
    command line arguments.
    '''
    
    parser = argparse.ArgumentParser(prog="./parana.py")
    subParsers = parser.add_subparsers(help="=== Commands accepted by PARANA ===")
    parserInfer = subParsers.add_parser('infer', help='compute parsimonious interaction history')
    parserExtract = subParsers.add_parser('extract', help='extract the implied network at a given set of nodes from a previously computed set of non-tree edges')
    parserCompare = subParsers.add_parser('compare', help='compare a reconstructed parsimonious ancestral network with some \"ground truth\" ancestral network')

    # parserInfer.add_argument("-r", "--rule", default='sum',\
    #                        help="{sum|or}")
    parserInfer.add_argument("-u", "--undirected", action="store_true", default=False,\
                            help="target networks are undirected (e.g. ppi)")
    parserInfer.add_argument("-l", "--loop", action="store_true", default=False,\
                            help="check for and break loops")
    parserInfer.add_argument("-c", "--cost_ratio", type=float, default=1.0,\
                            help="ratio of creation cost to deletion cost (i.e. creation costs \'c\' times as much as deletion")
    parserInfer.add_argument("-n", "--nonTreeEdges", default=None,\
                            help="where to write non-tree edges (optional)")
    parserInfer.add_argument("-t", "--target", default=None,\
                            help="edges from extant networks")
    parserInfer.add_argument("-d", "--duplications", default=None,\
                            help="duplication history")
    parserInfer.set_defaults(func=inferMain)


    
    parserExtract.add_argument("-u", "--undirected", action="store_true", default=False,\
                                   help="target networks are undirected (e.g. ppi)")
    parserExtract.add_argument("-n", "--nonTreeEdges", default=None, required=True,\
                                   help="where to write non-tree edges")
    parserExtract.add_argument("-s", "--source", default=None, required=True,\
                                   help="ancestor network")
    parserExtract.add_argument("-d", "--duplications", default=None, required=True,\
                                   help="duplication history")
    parserExtract.add_argument("-o", "--output", default=None, required=True,\
                                   help="file where extracted ancestral interactions should be written")
    parserExtract.set_defaults(func=extractMain)

    
    parserCompare.add_argument("-u", "--undirected", action="store_true", default=False,\
                                   help="target networks are undirected (e.g. ppi)")
    parserCompare.add_argument("-g", "--ground_truth", default=None, required=True,\
                                   help="\"ground truth\" ancestor network")
    parserCompare.add_argument("-r", "--recon", default=None, required=True,\
                                   help="reconstructed ancestor network")
    parserCompare.add_argument("-d", "--duplications", default=None, required=False,\
                                   help="duplication history")
    parserCompare.set_defaults(func=compareMain)
    
    return parser

    
if __name__ == "__main__" : main()
