Using PARANA
============

PARANA is a Python based tool, so no real setup or compilation is
required. However, to run PARANA, you will need a recent distribution
of Python 2 (>= 2.7) as well as the PyCogent
(pycogent.sourceforge.net) and NetworkX (networkx.lanl.gov) Python
packages.

PARANA has three main modes of operation:

1) Ancestral interaction inference ("infer") -- In this mode of operation,
PARANA uses the dynamic program described in the paper to reconstruct
a parsimonious set of non-tree edges (which represent the gain or loss
of ancestral interactions).

2) Ancestral network extraction ("extract") -- In this mode of
operation, PARANA uses the non-tree edges inferred via the "infer"
command to reconstruct the state of an ancestral network represented
by a particular cut through the duplication tree.  The user provides
the inferred non-tree edges, a set of node names representing the
nodal content of the ancestral network and the duplication history.
The program will output an adjacency list file containing the inferred
interactions amongst the nodes of the ancestral network.

3) Network comparison ("compare") -- In this mode of operation, PARANA
will compare the reconstructed ancestral network with some "ground truth"
ancestral network.  This mode of operation is useful mainly for testing
accuracy on synthetic data or data where, through some other means, a
high-confidence ancestral interaction network is believed to be known.

To display usage information, PARANA can be invoked with the '-h' option.
For further usage information regarding any specific mode of operation
(i.e. {infer, extract, compare}), PARANA can be invoked as follows:

./parana.py [mode] -h

A typical usage scenario would look something like what follows:

./parana.py infer -u -l -c 1.0 -d dupHistory.nwk -t extantNetworks.adj -n nte.edg

and then

./parana.py extract -u -d dupHistory.nwk -s ancestralNodes.txt -n nte.edg -o ancestralNetwork.adj

Here geneTree.nwk contains a NEWICK formatted tree that encodes the
duplication history of the proteins being analyzed, while
extantNetworks.adj contains an adjacency list; the entries of which
constitute the extant biological networks.  The '-u' denotes that the
extant networks should be considered undirected and the '-l' denotes
that blocking loops should be checked for and broken.  Finally, '-c
1.0' denotes that the ratio of the cost of creating an ancestral
interaction to deleting an ancestral interaction should be 1.

The second command uses the non-tree edges extracted via the first
command to reconstruct the ancestral network on the node set provided
in 'ancestralNodes.txt'.  The inferred edges are written out to
'ancestralNetwork.adj'.  The meaning of all other command line
arguments is the same.

Example
=======

The source code distribution comes with a sample dataset.  It's a synthetic regulatory network dataset
contained in the data subdirectory.  To test that PARANA is working correctly on your machine, you
can execute the following commands ('>' represents the terminal prompt, and '|' precedes the output
you should expect from PARANA):

> ./parana.py infer -d data/dup.nwk -t data/extant.adj -n output/nte.txt -c 1.0

|INFO:root:Round 0
|INFO:root:CONSTRAINTS = []
|INFO:root:Computing maximally parsimonious network history . . . 
|INFO:root:Cost = 56.0

> ./parana.py extract -d data/dup.nwk -s data/ancestor_nodes.txt -n output/nte.txt -o output/recon.adj

> ./parana.py compare -g data/ancestor.adj -r output/recon.adj
 |INFO:root:===== False Pos =====
 |INFO:root:set([(u'+146____', u'+179'), (u'+146____', u'+199'), (u'+194', u'+179'), (u'+146____', u'+120___'), (u'+194', u'+120___'), (u'+88__', u'+189'), (u'+194', u'+199')])
 |INFO:root:===== False Neg =====
 |INFO:root:set([(u'+194', u'+177'), (u'+177', u'+189')]) 
 |
 |prec = 0.95, rec = 0.985185185185, F1 = 0.967272727273

We can see that in this dataset, PARANA reconstructed an ancestral history requiring 56 'flip' operations,
and that the inferred ancestral network has an F1 score of 0.96.

 
Data Format
===========

== [Inputs] ==

* Duplication history --- The duplication history should be provided as a
a binary phylogenetic tree tree in NEWICK format.  There are no specific
restrictions on how internal nodes should be named.  However, leaf nodes
(representing either lost proteins, or proteins in extant species) should
adhere to the following naming convention:

Any leaf node corresponding to an existing protein in species 'X'
should have a name *ending* with the string '_X'.

Any leaf node associated with with a protein that is lost in species
'X' should have a name *beginning* with the string 'X*LOST'.

* Extant networks --- The content of the extant networks should be provided
in a simple adjacency list format.  The adjacencies for all extant networks should
be provided in a single file.

* Ancestral node content --- The file provided to the "extract" command which
contains the ancestral node content should list all nodes of the ancestral
network -- one per line.

== [Outputs] ==

* Non-tree edges --- The list of non-tree edges output by the "infer" command
are given in a simple "augmented" edge list format.  Each line contains a single
non-tree edge, and is of the format:

n1	 n2    dir

which denotes a non-tree edge between nodes n1 and n2, with the direction given
by dir (i.e. one of {f, r, b} -- forward, reverse or both).

* Ancestral adjacencies --- The interactions extracted by the "extract" command are
given in the adjacency list format.


License
=======

Copyright 2011 

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

