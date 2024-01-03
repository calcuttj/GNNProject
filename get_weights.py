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
  parser.add_argument('-o', default='weights.h5')
  args = parser.parse_args()
  bf = BeamFeatures.BeamFeatures(args.i)

  ys = np.zeros(6) 
  nentries = len(bf)


  for a, bfi in enumerate(bf):
    ys += bfi.y.numpy()[0]

    print(f'{a}/{nentries}', end='\r')
    #if a > 1000: break
  print(ys)

  with h5.File(args.o, "w") as outfile:
    outfile['weights'] = np.sum(ys)/ys 
