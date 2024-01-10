import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as GraphData
from torch_geometric.data import Dataset as GraphDataset

class BeamFeatures(GraphDataset):
    """
    TODO ADD DESCRIPTION
    """
    def __init__(self, file_path, norm_path=None, style="interaction", ave_charge=False):
        """
        Args: file_path ..... path to the HDF5 file that contains the feature data
        """
        # Initialize a file handle, count the number of entries in the file
        self._file_path = file_path
        self._file_handle = None
        self.ave_charge = ave_charge

        if style not in ["interaction", "beam_frac", "pdgs"]:
          raise Exception(f"Unknown beam features style {style}")
        self.style=style

        with h5py.File(self._file_path, "r", swmr=True) as data_file:
            self._entries = len(data_file['node_features'])

        if norm_path is not None:
          self.has_norm = True
          with h5py.File(norm_path, 'r', swmr=True) as norm_file:
            self.x_means = norm_file['x_means'][:]
            self.x_stds = norm_file['x_stds'][:]

            self.edge_means = norm_file['edge_means'][:]
            self.edge_stds = norm_file['edge_stds'][:]
        else:
          self.x_means = 0.
          self.x_stds = 1.
          self.edge_means = 0.
          self.edge_stds = 1.
          self.has_norm = False


    def __del__(self):
        
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
            
    def __len__(self):
        return self._entries

    def len(self):
        return len(self._entries)

    def __getitem__(self, idx):
        
        # Get the subset of node and edge features that correspond to the requested event ID
        if self._file_handle is None:
            self._file_handle = h5py.File(self._file_path, "r", swmr=True)
            
        vals = self._file_handle['node_features'][idx].reshape(-1, 9)
        if self.ave_charge:
          ave_charge = np.mean(vals[:,3:6], axis=1)
          new_vals = np.zeros((len(vals), 4))
          new_vals[:,:3] = vals[:,:3]
          new_vals[:,-1] = np.mean(vals[:,3:6], axis=1)
          vals = new_vals
        node_features = torch.tensor(
            (vals - self.x_means)/self.x_stds,
            dtype=torch.float32
        )
        #print(node_features.shape)

        vals = self._file_handle['edge_features'][idx].reshape(-1, 14)


        edge_index = torch.tensor(vals[:,-2:], dtype=torch.long).t()

        if self.ave_charge:
          pass
        edge_features = torch.tensor(
            (vals[:,:-2] - self.edge_means)/self.edge_stds,
            dtype=torch.float32
        )
        #print(edge_index.shape, edge_features.shape)

        if self.style == "beam_frac":
          truth = torch.tensor(
            self._file_handle['beam_fraction'][idx].reshape(-1,1),
            dtype=torch.float32
          )
        elif self.style == "pdgs":
          truth = torch.tensor(
            self._file_handle['truth_pdg'][idx].reshape(-1,4),
            dtype=torch.float32
          )
        else:
          truth = torch.tensor(
              self._file_handle['truth'][idx].reshape(-1,6),
              dtype=torch.float32
          )
        return GraphData(x = node_features,
                         edge_index = edge_index,
                         edge_attr = edge_features,
                         y = truth,
                         edge_label = None,
                         index = idx)

    def get(self,idx):
        return self[idx]
