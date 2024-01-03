from argparse import ArgumentParser as ap
import h5py as h5
import numpy as np
import BeamFeatures

#def get_x_norm(bf, index):

def get_nedge_attrs(bf):
  return bf[0].edge_attr.shape[1]

if __name__ == '__main__':
  parser = ap()
  parser.add_argument('-i', required=True)
  parser.add_argument('-o', default='norm_vals.h5')
  args = parser.parse_args()
  bf = BeamFeatures.BeamFeatures(args.i) #'/home/jake/GNN_work/data/merged.h5')

  xs = []
  nentries = len(bf)


  for a, bfi in enumerate(bf):
    x = bfi.x.numpy()
    xs += [i for i in x]
    print(f'{a}/{nentries}', end='\r')
    if a > 20000: break
  x_means, x_stds = np.mean(xs, axis=0), np.std(xs, axis=0)

  del xs
  print()

  nedge_attrs = get_nedge_attrs(bf)
  edge_means = np.zeros(nedge_attrs)
  edge_stds = np.zeros(nedge_attrs)
  for i in range(0, nedge_attrs, 3):
    edges = [] 
    for a, bfi in enumerate(bf):
      e = bfi.edge_attr.numpy()
      edges += [ei[i:i+3] for ei in e]
      print(f'{a}/{nentries}', end='\r')
      if a > 20000: break
    edge_means[i:i+3], edge_stds[i:i+3] = np.mean(edges, axis=0), np.std(edges, axis=0)
    print()


  with h5.File(args.o, "w") as outfile:
    outfile['x_means'] = x_means
    outfile['x_stds'] = x_stds

    outfile['edge_means'] = edge_means
    outfile['edge_stds'] = edge_stds
