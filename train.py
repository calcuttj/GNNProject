from argparse import ArgumentParser as ap
import yaml
import numpy as np
import h5py as h5
import torch
import model, BeamFeatures
from torch_geometric.loader import DataLoader

def get_confusion(data, x, n=4):
  confusion = np.zeros((n, n))
  for i in range(n):
      for j in range(n):
          indices = np.where(x.argmax(axis=1) == i)
          confusion[i,j] = (
              data.y.argmax(axis=1)[indices] == j
          ).sum()
  return confusion

def check_batchnum(maxnum, batchnum):
  return (maxnum > 0 and batchnum >= maxnum)

def run_test(net, test_loader, device, loss_fn, ndim=4):
  print('Running test', len(test_loader), 'batches')
  #losses = []
  running_loss = 0.
  ngood = 0
  nevents = 0
  nbatches = 0
  
  total_confusion = np.zeros((ndim, ndim))
  net.eval()
  with torch.no_grad():
    for batchnum, batch in enumerate(test_loader):
      if check_batchnum(args.max_test_batch, batchnum): break
      batch.to(device)
      pred = net(batch, batch.batch)
      loss = loss_fn(pred, batch.y.argmax(axis=1))
      theloss = loss.item()
      running_loss += theloss
      if not batchnum % 10: print(f'{batchnum}')
      #losses.append(theloss)
      ngood += (pred.argmax(axis=1) == batch.y.argmax(axis=1)).sum()
      nevents += len(batch)
      nbatches += 1
      total_confusion += get_confusion(batch, pred, ndim)
    running_loss /= nbatches 
    accuracy = ngood/nevents
    return {
      #'losses':losses,
      'ave_loss':running_loss,
      'accuracy':accuracy,
    }


if __name__ == '__main__':
  parser = ap()
  parser.add_argument('--train', required=True, type=str)
  parser.add_argument('--batch', type=int, default=16)
  parser.add_argument('--max_train_batch', type=int, default=-1)
  parser.add_argument('--max_test_batch', type=int, default=-1)
  parser.add_argument('--epoch', type=int, default=1)
  parser.add_argument('--test', default=None, type=str)
  parser.add_argument('--save', default=None, type=str)
  parser.add_argument('--weights', default=None, type=str)
  parser.add_argument('--norm', default=None, type=str)
  parser.add_argument('--style', default='interaction', type=str)
  parser.add_argument('--checkpoint', default=None, type=str)
  parser.add_argument('--ave_charge', action='store_true')
  parser.add_argument('--learning_rate', '--lr', type=float, default=1.e-2)
  args = parser.parse_args()

  good_styles = ['interaction', 'beam_frac', 'pdgs']
  if args.style not in good_styles:
    combined = ', '.join(good_styles)
    raise Exception(f'Must supply --style as one of {combined}')

  bf_train = BeamFeatures.BeamFeatures(args.train, args.norm, args.style, ave_charge=args.ave_charge)
  train_loader = DataLoader(
      bf_train,
      shuffle=True,
      num_workers=0,
      batch_size=args.batch,
  )

  if args.test is not None:
    bf_test = BeamFeatures.BeamFeatures(args.test, args.norm, args.style, ave_charge=args.ave_charge)
    test_loader = DataLoader(
        bf_test,
        shuffle=False,
        num_workers=0,
        batch_size=args.batch,
    )

  # check if a GPU is available. Otherwise run on CPU
  device = 'cpu'
  args_cuda = torch.cuda.is_available()
  if args_cuda: device = "cuda:0"
  print('device : ',device)

  weights = None
  if args.weights is not None:
    #TODO -- finish setting up weights from yaml
    with h5.File(args.weights, 'r') as fweight:
      weights = torch.tensor(fweight['weights']).float().to(device)
      

  net = model.GNNModel(
      (args.style == 'beam_frac'),
      outdim=(6 if args.style == 'interaction' else 4),
      node_input=(4 if args.ave_charge else 9),
      edge_input=(8 if args.ave_charge else 12),
  )

  #loss_fn = torch.nn.BCELoss(reduction='mean', weight=weights)
  if args.style == 'beam_frac':
    loss_fn = torch.nn.MSELoss()
  else:
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean', weight=weights)
  optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=5e-4)
  
  net.to(device)
  
  losses = []
  accuracies = []
  #test_losses = []
  test_ave_losses = []
  test_accuracies = []

  if args.test is not None:
    test_results = run_test(net, test_loader, device, loss_fn)
    test_ave_losses.append(test_results['ave_loss'])
    test_accuracies.append(test_results['accuracy'])

  for i in range(args.epoch):
    net.train()
    print(f'EPOCH {i}', len(train_loader), 'batches')
    running_loss = 0.
    for batchnum, batch in enumerate(train_loader):
      if check_batchnum(args.max_train_batch, batchnum): break
      optimizer.zero_grad()
      batch.to(device)
      pred = net(batch, batch.batch)
      loss = loss_fn(pred, batch.y) if (args.style == 'beam_frac') else loss_fn(pred, batch.y.argmax(axis=1))
      loss.backward()
      optimizer.step()
      theloss = loss.item()
      running_loss += theloss
      if not batchnum % 10: print(f'{batchnum}')
      if not batchnum % 100 and batchnum > 0:
        print(f'\n(Batch {batchnum}) Loss: {running_loss / 100.}')
        running_loss = 0.
      losses.append(theloss)

      accuracies.append(
        (pred.argmax(axis=1) == batch.y.argmax(axis=1)).sum() / len(batch)
      )


    if args.test is not None:
      test_results = run_test(net, test_loader, device, loss_fn)
      #test_losses.append(test_results['losses'])
      test_ave_losses.append(test_results['ave_loss'])
      test_accuracies.append(test_results['accuracy'])

  if args.save is not None:
    with h5.File(args.save, 'w') as fsave:
      fsave['losses'] = losses
      fsave['accuracies'] = accuracies

      #Write out test losses
      if args.test is not None:
        fsave['test_ave_losses'] = test_ave_losses
        fsave['test_accuracies'] = test_accuracies

  if args.checkpoint is not None:
    torch.save({
      'model_state_dict': net.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
    }, args.checkpoint)
