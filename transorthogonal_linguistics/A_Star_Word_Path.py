# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 17:06:43 2015

@author: peter
"""
import logging
import numpy
import scipy
import scipy.spatial
import word_path

_default_feature_file = "db/features.npy"
_default_vocab_file = "db/vocab.npy"

class A_Star_Word_Path(word_path.Features):
    """A class to calulate the A* path between two words"""
    def __init__(self,f_features=_default_feature_file,
                 f_vocab=_default_vocab_file,
                 empty=False):
        """Loads the data and computes the nearest-neighbour relationships"""
        super(word_path.Features,self).__init__(f_features,f_vocab,empty)
        Hull=scipy.spatial.ConvexHull(self.features)
        self.neighbours=[set() for i in xrange(len(self))]
        for facet in Hull.simplices.tolist():
            for vertex in facet:
                self.neighbours[vertex]|={point for point in facet}
                
    def __call__(self,start,end):
        """Finds the A* path between start and end"""
        start_index=self.inv_index[start]
        end_index=self.inv_index[end]
        result=None
        frontier=[{'path':[start_index],
                   'travelled':0.0,
                   'heuristic':numpy.linalg.norm(self.features[start_index]-self.features[end_index])}]
        while result is None:
            candidate=frontier.pop(0)
            last=candidate['path'][-1]
            if last==end_index:
                result=candidate['path']
            else:
                for n in self.neighbours[last]:
                    if n not in candidate['path']:
                        frontier.append({'path':candidate['path']+[n],
                                         'travelled':candidate['travelled']+numpy.linalg.norm(self.features[last]-self.features[n]),
                                         'heuristic':numpy.linalg.norm(self.features[n]-self.features[end_index])})
                frontier.sort(key=lambda x:x['travelled']+x['heuristic'])
        return [self.index[i] for i in result]
        
if __name__ == "__main__":

    import argparse

    desc = '''
    A* path

    Finds the route shortest across a graph formed by connecting each node to 
    its n nearest neighbours in n-dimensional space
    '''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--f_features",
                        help="numpy feature matrix",
                        default=_default_feature_file)
    parser.add_argument("--f_vocab",
                        help="numpy vocab vector",
                        default=_default_vocab_file)
    parser.add_argument("--word_cutoff", '-c',
                        help="Number of words to select",
                        type=int, default=25)
    parser.add_argument("words",
                        nargs="*",
                        help="Space separated pairs of words example: "
                        "python word_path.py boy man mind body")

    args = parser.parse_args()

    if not args.words:
        msg = "You must either pick at least two words!"
        raise SyntaxError(msg)

    if len(args.words) % 2 != 0:
        msg = "You input an even number of words!"
        raise SyntaxError(msg)

    word_pairs = [[w1, w2] for w1, w2 in zip(args.words[::2],
                                             args.words[1::2])]

    network = A_Star_Word_Path(args.f_features,
                               args.f_vocab)

    for k, (w1, w2) in enumerate(word_pairs):
            
        result = network(w1,w2)

        print_result(result)

        if k: print
            