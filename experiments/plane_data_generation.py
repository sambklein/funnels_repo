from nflows.utils import get_num_parameters
from nflows import transforms
from nflows import flows
import nflows

import torch
import numpy as np
import matplotlib.pyplot as plt

import time

from funnels.data.plane import load_plane_dataset
from funnels.models.flows import get_transform
from funnels.models.sur_flows import SurNSF
from funnels.data.hyper_dim import HyperCheckerboardDataset

import argparse

from funnels.utils.io import save_object
from funnels.utils.plotting import getCrossFeaturePlot, plot2Dhist, plot_likelihood
from funnels.utils.torch_utils import tensor2numpy


def parse_args():
    parser = argparse.ArgumentParser()

    # Saving
    parser.add_argument('-d', '--outputdir', type=str, default='plane_images_local',
                        help='Choose the base output directory')
    parser.add_argument('-n', '--outputname', type=str, default='local',
                        help='Set the output name directory')
    parser.add_argument('--load', type=int, default=0,
                        help='Load a model?')

    # Model set up
    parser.add_argument('--inp_dim', type=int, default=2,
                        help='The dimension of the input data.')
    parser.add_argument('--nodes', type=int, default=64,
                        help='The number of nodes in each layer used to learn the flow parameters.')
    parser.add_argument('--num_blocks', type=int, default=2,
                        help='The number of layers used to learn the flow placements.')
    parser.add_argument('--nstack', type=int, default=2,
                        help='The number of flow layers.')
    parser.add_argument('--tails', type=str, default='linear',
                        help='The tail function to apply.')
    parser.add_argument('--tail_bound', type=float, default=4.,
                        help='The tail bound.')
    parser.add_argument('--num_bins', type=int, default=10,
                        help='The number of bins to use in the RQ-NSF.')
    parser.add_argument('--num_add', type=int, default=1,
                        help='The number of additional layers to add.')
    parser.add_argument('--add_sur', type=int, default=0,
                        help='Whether to make the additional layers funnels layers.')
    parser.add_argument('--splines', type=int, default=1,
                        help='Use RQ-NSF if true, else Real NVP.')

    # Dataset and training parameters
    parser.add_argument('--dataset', type=str, default='checkerboard',
                        help='The name of the plane dataset on which to train.')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Whether to make the additional layers funnels layers.')
    parser.add_argument('--n_epochs', type=int, default=10,
                        help='Whether to make the additional layers funnels layers.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Whether to make the additional layers funnels layers.')
    parser.add_argument('--ndata', type=int, default=int(1e5),
                        help='Whether to make the additional layers funnels layers.')
    parser.add_argument('--n_val', type=int, default=int(1e3),
                        help='Whether to make the additional layers funnels layers.')
    parser.add_argument('--n_test', type=int, default=int(1e4),
                        help='Whether to make the additional layers funnels layers.')
    parser.add_argument('--gclip', type=float, default=5.,
                        help='Whether to make the additional layers funnels layers.')
    parser.add_argument('--monitor_interval', type=int, default=100,
                        help='Whether to make the additional layers funnels layers.')
    parser.add_argument('--bnorm', type=int, default=0,
                        help='Apply batch normalisation?')

    return parser.parse_args()


def checkerboard_test():
    # Parse args
    args = parse_args()

    # Set up saving and change the name of some args
    svo = save_object(args.outputdir, args.outputname, args=args)
    inp_dim = args.inp_dim
    out_dim = inp_dim - args.num_add * args.add_sur
    nstack = args.nstack + args.num_add * (1 - args.add_sur)

    spline = args.splines

    # Set up and define the model
    transform_list = [get_transform(inp_dim, nodes=args.nodes,
                                    nstack=nstack,
                                    num_blocks=args.num_blocks,
                                    tail_bound=args.tail_bound,
                                    num_bins=args.num_bins,
                                    tails=args.tails,
                                    spline=spline)]

    # Add the funnels layers
    dim = inp_dim
    sur_layers = []
    for _ in range(args.num_add * args.add_sur):
        sur_layers += [SurNSF(dim, args.nodes,
                              num_blocks=args.num_blocks,
                              tail_bound=args.tail_bound,
                              num_bins=args.num_bins,
                              tails=args.tails,
                              spline=spline)
                       ]
        dim -= 1
        if args.bnorm:
            sur_layers += [
                transforms.BatchNorm(dim)
            ]
        # transform_list += [transforms.ReversePermutation(dim)]
        sur_layers += [transforms.LULinear(dim)]

    transform_list += sur_layers[:-1]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Running on {device}')
    transform = transforms.CompositeTransform(transform_list)
    base_dist = nflows.distributions.StandardNormal([out_dim])
    flow = flows.Flow(transform, base_dist).to(device)
    print(f'There are {get_num_parameters(flow)} params')

    # Set up the dataset and training parameters
    val_batch_size = 1000

    optimizer = torch.optim.Adam(flow.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.ndata / args.batch_size * args.n_epochs, 0)

    # Train the model
    train_save = []
    val_save = []

    if args.load:
        transform.load_state_dict(torch.load(svo.save_name('model', extension='')))
    else:
        for epoch in range(args.n_epochs):
            start_time = time.time()
            trainset = load_plane_dataset(args.dataset, args.ndata)
            training_data = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                                        num_workers=0)
            validset = load_plane_dataset(args.dataset, args.n_val)
            valid_data = torch.utils.data.DataLoader(validset, batch_size=val_batch_size, shuffle=True, num_workers=0)

            # Training
            running_loss = []
            for i, lo_data in enumerate(training_data, 0):
                if spline:
                    lo_data = lo_data * args.tail_bound
                data = lo_data.to(device)
                # zero the parameter gradients before calculating the losses
                optimizer.zero_grad()
                loss = -flow.log_prob(data).mean()
                loss.backward()
                if args.gclip > 0:
                    torch.nn.utils.clip_grad_value_(flow.parameters(), args.gclip)
                optimizer.step()
                scheduler.step()

                if i % args.monitor_interval == 0:
                    losses = -flow.log_prob(data).mean()
                    running_loss += [losses.item()]
                    s = '[{}, {}] {}'.format(epoch + 1, i + 1, running_loss[-1])

            # Update training loss trackers
            train_save += [np.mean(running_loss, 0)]

            # Validation
            val_loss = np.zeros((int(args.n_val / val_batch_size)))
            for i, data in enumerate(valid_data):
                val_loss[i] = -flow.log_prob(data.to(device)).mean().item()

            val_save += [np.mean(val_loss, 0)]

            with open(svo.save_name('timing', extension='txt'), 'w') as f:
                f.write('{}\n'.format(time.time() - start_time))

        torch.save(transform.state_dict(), svo.save_name('model', extension=''))

        print('Finished Training')

        # Plot the training information
        fig, ax = plt.subplots(1, 1, figsize=(20, 5))
        ax.plot(train_save, label='Train')
        ax.plot(val_save, '--', label='validation')
        ax.legend()
        ax.set_ylabel("loss")
        ax.set_xlabel("epoch")
        plt.savefig(svo.save_name('training'))

    flow.eval()

    # Plot the model density
    test_bs = int(1e5)
    n_batches = int(np.ceil(args.n_test / test_bs))
    scores_uniform = torch.empty((n_batches, test_bs))
    encoding = torch.empty((n_batches, test_bs, dim))
    samples = torch.empty((n_batches, test_bs, inp_dim))
    uniform_sample = torch.empty((n_batches, test_bs, inp_dim))

    with torch.no_grad():
        for i in range(n_batches):

            # Plot the distribution of the encoding
            cpu = torch.device("cpu")
            testset = load_plane_dataset(args.dataset, test_bs)
            test_data = testset.data.to(device)
            if spline:
                test_data = test_data * args.tail_bound
            encoding[i] = flow.transform_to_noise(test_data).to(cpu)

            # Plot a selection of generated samples
            samples[i] = flow.sample(test_bs)

            uniform_sample[i] = torch.distributions.uniform.Uniform(torch.zeros(inp_dim) - 1,
                                                                    torch.ones(inp_dim),
                                                                    validate_args=None).sample([test_bs])
            if spline:
                uniform_sample[i] *= args.tail_bound
            scores_uniform[i] = flow.log_prob(uniform_sample[i].to(device)).to(cpu)

    encoding = encoding.view(-1, dim)
    samples = samples.view(-1, inp_dim).cpu().numpy()
    scores_uniform = scores_uniform.view(-1)
    uniform_sample = uniform_sample.view(-1, inp_dim)

    plt.figure()
    getCrossFeaturePlot(encoding, svo.save_name('encoding'))

    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    plot2Dhist(samples, axs)
    fig.savefig(svo.save_name('samples'))

    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    plot_likelihood(uniform_sample, scores_uniform, ax)
    fig.tight_layout()
    plt.savefig(svo.save_name('likelihood'))
    plt.clf()

    with open(svo.save_name('', extension='npy'), 'wb') as f:
        np.save(f, encoding)
        np.save(f, samples)
        np.save(f, tensor2numpy(scores_uniform))
        np.save(f, tensor2numpy(uniform_sample))


if __name__ == '__main__':
    checkerboard_test()
