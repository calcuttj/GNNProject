import h5py
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as GraphData
from torch_geometric.data import Dataset as GraphDataset

class BeamFeatures(GraphDataset):
    """
    TODO ADD DESCRIPTION
    """
    def __init__(self, file_path):
        """
        Args: file_path ..... path to the HDF5 file that contains the feature data
        """
        # Initialize a file handle, count the number of entries in the file
        self._file_path = file_path
        self._file_handle = None
        with h5py.File(self._file_path, "r", swmr=True) as data_file:
            self._entries = len(data_file['node_features'])

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
            
        node_features = torch.tensor(
            self._file_handle['node_features'][idx].reshape(-1, 9),
            dtype=torch.float32
        )
        #node_features, group_ids, node_labels = (
        #    node_info[:,:-3], node_info[:,-2].long(), node_info[:,-1].long()
        #)
        
        edge_info = torch.tensor(
            self._file_handle['edge_features'][idx].reshape(-1, 14),
            dtype=torch.float32
        )

        #edge_features, edge_index, edge_labels = (
        edge_features, edge_index = (
            edge_info[:,:-2],
            edge_info[:,-2:].long().t(),
        #    edge_info[:,-1].long()
        )
        #edge_index = self.file_handle['edge_indices'][idx].reshape(-1, 2)
        
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
