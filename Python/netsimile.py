import sys
import getopt
import glob
import zipfile
import pandas as pd
from igraph import *
import scipy.spatial.distance as dis
from itertools import combinations
from itertools import product

# read the zipped files in given directory and
# generate graph list for each graph as separate files
def read_collate_graph_files(path_name):
    str = path_name+"*.zip"
    print "Collating files from ", path_name, " ..."
    graphsEdgeLists = []
    fileList = glob.glob(str)                           # for example "C:\\graphs\\*.zip"
    for name in fileList:                               # list of zip files in current directory
        zfile = zipfile.ZipFile(name)                   # for each zip file
        zipName = os.path.basename(name)
        rest = zipName.split('.', 1)[0]                 # convert to g1 as file name, remove after dot
        graphsEdgeLists.append(rest)
        funzip = open(rest,'w')                         # open g1
        for fileName in zfile.namelist():               # files inside the zip files
            with zfile.open(fileName) as f:             # read file line by line
                for line in f:
                    funzip.write(line)
    funzip.close()
    return graphsEdgeLists

# create graph objects from generated files
def create_graph_objects(graphsEdgeLists):
    print "Creating graph objects ... "
    df=[]
    for i , v in enumerate(graphsEdgeLists):
        y = pd.read_csv(v,sep=" ", names=["Edge1", "Edge2"])
        n_vertex, n_edge = y.irow(0)
        y = y.drop(0)
        df.append(y)
    graphs = []
    for i, v in enumerate(df):
        graph = Graph(edges=[(x[1]["Edge1"], x[1]["Edge2"]) for x in df[i].iterrows()], directed=False)
        graph.vs.select(_degree = 0).delete()
        graph = graph.simplify()
        graph.vs["original_index"] = graph.vs.indices     # assign original index in attribute original index attribute
        graphs.append(graph)
        print "Number of edges : ", graph.ecount(), " Number of vertices : ", graph.vcount()
        print graph.get_edgelist()
    return graphs

# find the average number of neighbour neighbour for each node
def avg_number_2_hop_away_neighbours(neighbours_of_v):
    sumv = 0
    for vv in neighbours_of_v:
        sumv += len(vv.neighbors())
    return float(sumv)/len(neighbours_of_v)

# finds the number of edges in ego net,
# i.e. which is number of neighbours plus possible edges between nodes in ego net
def number_of_edges_in_egonet(neighbours_of_v, graph):
    v_neighbour_list = [nei['original_index'] for nei in neighbours_of_v]
    possible_neighbour_pairs = [[comb[0], comb[1]] for comb in combinations(v_neighbour_list, 2)]
    edges_in_egonet = 0
    for pair in possible_neighbour_pairs:
        check = graph.get_eid(pair[0], pair[1], directed=False, error=False)
        if(check!=-1):
            edges_in_egonet += 1
    edges_in_egonet += len(neighbours_of_v)
    return edges_in_egonet

#finds the neighbours of the identified ego net
def neighbours_of_egonet(neighbours_of_v, v, ego_net):
    neighbour_nodes_list = []
    for node in ego_net:                                            # for all nodes in ego net find the neighbours
        for node1 in node.neighbors():
            if node1['original_index'] not in neighbour_nodes_list:
                neighbour_nodes_list.append(node1['original_index'])

    neighbour_nodes_index = [nei['original_index'] for nei in neighbours_of_v]
    neighbours_of_ego_net = [x for x in neighbour_nodes_list if x not in neighbour_nodes_index]
    return neighbours_of_ego_net

#finds the outgoing edges of the egonet
def out_edges_of_egonet(neighbours_of_ego_net, ego_net_index, graph):
    out_neigh = 0
    nnpairs = [[prod[0], prod[1]]for prod in product(neighbours_of_ego_net, ego_net_index)]
    for pair in nnpairs:
        check = graph.get_eid(pair[0], pair[1], directed=False, error=False)
        if(check!=-1):
            out_neigh += 1
    return out_neigh

#generate data frame of features of a particular graph which is passed
def generate_dataframe(graph, di, ci, dni, cni, eegoi, neegoi, outeegoi):
    dataFrm = pd.DataFrame(index=range(graph.vcount()))
    dataFrm['f1'] = di
    dataFrm['f2'] = ci
    dataFrm['f3'] = dni
    dataFrm['f4'] = cni
    dataFrm['f5'] = eegoi
    dataFrm['f6'] = neegoi
    dataFrm['f7'] = outeegoi
    print "For graph 7 features for all nodes :"
    print dataFrm
    return dataFrm

#Generate Feature Matrices
def create_feature_matrix(graphs):
    print "Creating feature matrices ... "
    features = []
    for i, graph in enumerate(graphs):
        di = graph.degree(graph.vs)                             # 1st feature (val for small)
        ci = graph.transitivity_local_undirected(graph.vs)      # 2nd feature (implemented based on http://igraph.sourceforge.net/doc/python/igraph.GraphBase-class.html#transitivity_local_undirected)
        dni, cni, eegoi, neegoi, outeegoi = [], [], [], [], []
        for v in graph.vs:
            neighbours_of_v = v.neighbors()                                 # repeatedly used methods:
            dni.append(avg_number_2_hop_away_neighbours(neighbours_of_v))   # 3rd feature (val for small)
            x = graph.transitivity_local_undirected(neighbours_of_v)        # 4th feature (val for small)
            cni.append(sum(x)/len(neighbours_of_v))
            eegoi.append(number_of_edges_in_egonet(neighbours_of_v, graph)) # 5th feature (val for small)
            ego_net = neighbours_of_v                                       #get the neighbor of the vertex
            ego_net.append(v)                                               # we now have the node list in ego_net
            neighbours_of_ego_net = neighbours_of_egonet(neighbours_of_v, v, ego_net)
            ego_net_index = [x['original_index'] for x in ego_net]
            neegoi.append(len(neighbours_of_ego_net))                                           # 7th feature - neighbours of ego net
            outeegoi.append(out_edges_of_egonet(neighbours_of_ego_net, ego_net_index, graph))   #6th feature - out going edges from ego net
        #enf for
        features.append(generate_dataframe(graph, di, ci, dni, cni, eegoi, neegoi, outeegoi))
    return features

#aggregate the results for each feature matrix corresponding to a graph into a signature vector
def aggregator(features):
    print "Aggregating features ... "
    sglist_concated = []
    for feature in features:
        sglist = []                                                 #signature vector list for all vertices of a graph
        for a, b, c, d, e in zip(feature.mean().tolist(), feature.median().tolist(), feature.std().tolist(), feature.skew().tolist(), feature.kurt().tolist()):
            sglist.append([a, b, c, d, e])                          # append mean median standard deviation skew and kurtosis for kth feature
        sglist_concated.append([item for sublist in sglist for item in sublist]) #append signature vector of the graph to list
    return sglist_concated

#pairwise compare the signature vector representing the graphs
def pairwise_compare(signature_vectors):
    print "Pairwise Comparison of graphs ... "
    for i in range(0, len(signature_vectors)):
        for j in range(i+1, len(signature_vectors)):
            print "\t Distance between graph", i, "and graph ", j, dis.canberra(signature_vectors[i],signature_vectors[j])

#driver method
def process(path_name):
    print "Begin Program ... "
    graphList = read_collate_graph_files(path_name)
    graphs = create_graph_objects(graphList)
    features = create_feature_matrix(graphs)
    signature_vectors = aggregator(features)
    pairwise_compare(signature_vectors)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print >> sys.stderr, "ERROR: Program requires single argument: path to graph file."
    else:
        process(sys.argv[1])