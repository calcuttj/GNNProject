from argparse import ArgumentParser as ap
import h5py as h5
import torch
import model, BeamFeatures
from torch_geometric.loader import DataLoader

def run_test(test_loader, device):
  print('Running test')
  #losses = []
  running_loss = 0.
  with torch.no_grad():
    for batchnum, batch in enumerate(test_loader):
      batch.to(device)
      pred = net(batch, batch.batch)
      loss = loss_fn(pred, batch.y)
      theloss = loss.item()
      running_loss += theloss
      if not batchnum % 10: print(f'{batchnum}')
      #losses.append(theloss)
    running_loss /= len(test_loader)
    return {
      #'losses':losses,
      'ave_loss':running_loss,
    }


if __name__ == '__main__':
  parser = ap()
  parser.add_argument('--train', required=True, type=str)
  parser.add_argument('--batch', type=int, default=16)
  parser.add_argument('--epoch', type=int, default=1)
  parser.add_argument('--test', default=None, type=str)
  parser.add_argument('--save', default=None, type=str)
  args = parser.parse_args()

  bf_train = BeamFeatures.BeamFeatures(args.train)
  train_loader = DataLoader(
      bf_train,
      shuffle=True,
      num_workers=0,
      batch_size=args.batch,
  )

  if args.test is not None:
    bf_test = BeamFeatures.BeamFeatures(args.test)
    test_loader = DataLoader(
        bf_test,
        shuffle=False,
        num_workers=0,
        batch_size=args.batch,
    )

  net = model.GNNModel()

  loss_fn = torch.nn.BCELoss(reduction='mean')
  optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
  
  # check if a GPU is available. Otherwise run on CPU
  device = 'cpu'
  args_cuda = torch.cuda.is_available()
  if args_cuda: device = "cuda:0"
  print('device : ',device)
  net.to(device)
  
  losses = []
  #test_losses = []
  test_ave_losses = []

  if args.test is not None:
    test_results = run_test(test_loader, device)
    test_ave_losses.append(test_results['ave_loss'])

  net.train()
  for i in range(args.epoch):
    print(f'EPOCH {i}')
    running_loss = 0.
    for batchnum, batch in enumerate(train_loader):
      optimizer.zero_grad()
      batch.to(device)
      pred = net(batch, batch.batch)
      loss = loss_fn(pred, batch.y)
      loss.backward()
      optimizer.step()
      theloss = loss.item()
      running_loss += theloss
      if not batchnum % 10: print(f'{batchnum}')
      if not batchnum % 100 and batchnum > 0:
        print(f'\n(Batch {batchnum}) Loss: {running_loss / 100.}')
        running_loss = 0.
      losses.append(theloss)
    if args.test is not None:
      test_results = run_test(test_loader, device)
      #test_losses.append(test_results['losses'])
      test_ave_losses.append(test_results['ave_loss'])

  if args.save is not None:
    with h5.File(args.save, 'w') as fsave:
      fsave['losses'] = losses

      #Write out test losses
      if args.test is not None:
        fsave['test_ave_losses'] = test_ave_losses
