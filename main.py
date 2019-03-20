import argparse
import os
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
from fake_data_loader import get_fake_loader


def main(config):
    cudnn.benchmark = True
    # train dataloader
    data_loader = get_loader(image_path=config.image_path,
                             image_size=config.image_size,
                             batch_size=config.batch_size,
                             num_workers=config.num_workers)
    # load fake data for training
    fake_data_loader = get_fake_loader(image_path=config.fake_images_path,
                                       image_size=config.image_size,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers)
    # Load test data
    test_data_loader = get_fake_loader(image_path=config.test_images_path,
                                       image_size=config.image_size,
                                       batch_size=1,
                                       num_workers=config.num_workers)
    # base object for training nets
    solver = Solver(config, data_loader, fake_data_loader, test_data_loader)

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.sample_path):
        os.makedirs(config.sample_path)

    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'sample':
        solver.sample()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=32)

    # training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--sample_size', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr_g', type=float, default=0.0002)
    parser.add_argument('--lr_d', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.9)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.9)  # momentum2 in Adam

    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--sample_path', type=str, default='./samples')
    parser.add_argument('--image_path', type=str,
                        default='./train_mnist')
    parser.add_argument('--fake_images_path', type=str,
                        default='./train_mnist')
    parser.add_argument('--test_images_path', type=str,
                        default='./test_mnist_matrix')
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=50)

    # load sample data / load model weight from this number
    parser.add_argument('--load_sample', type=str, default='1-100')
    parser.add_argument('--save_test', type=int, default=1)
    # use dcgan generator as default if 1 and otherwise use unet
    parser.add_argument('--gen_dcgan', type=int, default=0)

    config = parser.parse_args()
    print(config)
    main(config)
