import torchvision
import os
from torch import optim
from torch.autograd import Variable
from model import Generator
from model import U_Generator
from model import Alex_disc
from unet.unet_parts import *
import numpy as np
from sklearn.metrics import confusion_matrix
from PIL import Image


class Solver(object):
    def __init__(self, config, data_loader, fake_data_loader, test_data_loader):
        self.generator = None
        self.discriminator = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.z_dim = config.z_dim
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.image_size = config.image_size
        self.data_loader = data_loader
        self.fake_data_loader = fake_data_loader
        self.test_data_loader = test_data_loader
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.sample_size = config.sample_size
        self.lr_g = config.lr_g
        self.lr_d = config.lr_d
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.fake_images_path = config.fake_images_path
        self.sample_path = config.sample_path
        self.model_path = config.model_path
        self.gen_dcgan = config.gen_dcgan
        self.build_model()
        self.load_sample = config.load_sample
        self.save_test = config.save_test

    def build_model(self):
        """Build generator and discriminator."""
        ## choosing generator between DCGAN or Unet
        if self.gen_dcgan:
            self.generator = Generator(z_dim=self.z_dim,
                                       image_size=self.image_size,
                                       conv_dim=self.g_conv_dim)
            # define generator (using Unet as Generator)
        else:
            self.generator = U_Generator(1, 1)
        self.discriminator = Alex_disc()  # our_Discriminator()
        self.g_optimizer = optim.Adam(self.generator.parameters(),
                                      self.lr_g, [self.beta1, self.beta2])
        self.d_optimizer = optim.Adam(self.discriminator.parameters(),
                                      self.lr_d, [self.beta1, self.beta2])
        # use gpue if it's available
        if torch.cuda.is_available():
            self.generator.cuda()
            self.discriminator.cuda()

    def to_variable(self, x):
        """Convert tensor to variable."""
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.discriminator.zero_grad()
        self.generator.zero_grad()

    def denorm(self, x):
        """Convert range (-1, 1) to (0, 1)"""
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def train(self):

        """Train generator and discriminator."""
        fixed_noise = self.to_variable(torch.randn(self.batch_size, self.z_dim))
        total_step = len(self.data_loader)

        meanImage = Image.open("mean_image.jpg")
        meanImage = np.array(meanImage, dtype="f")
        meanImage = torch.from_numpy(meanImage)
        meanImage = torch.FloatTensor(meanImage)
        meanImage = torch.unsqueeze(meanImage, 0)

        for epoch in range(self.num_epochs):
            for i, (images, noisi_image) in enumerate(zip(self.data_loader, self.fake_data_loader)):
                images = images[0]
                noisi_image = noisi_image[0]
                # ===================== Train D =====================#
                images = self.to_variable(images)
                noisi_image = self.to_variable(noisi_image)
                batch_size = images.size(0)
                noise = self.to_variable(torch.randn(batch_size, self.z_dim))

                # Train D to recognize real images as real.
                outputs = self.discriminator(images)

                real_loss = torch.mean((
                                               outputs - 1) ** 2)  # L2 loss instead of Binary cross entropy loss (this is optional for stable training)

                # Train D to recognize fake images as fake.
                if self.gen_dcgan:
                    fake_images = self.generator(noise)
                    outputs = self.discriminator(fake_images)
                else:
                    fake_images = self.generator(noisi_image)
                    outputs = self.discriminator(fake_images)

                fake_loss = torch.mean(outputs ** 2)

                # Backprop + optimize
                d_loss = real_loss + fake_loss
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()
                # ===================== Train G =====================#
                noise = self.to_variable(torch.randn(batch_size, self.z_dim))

                # Train G so that D recognizes G(z) as real.
                if self.gen_dcgan:
                    fake_images = self.generator(noise)
                else:
                    fake_images = self.generator(noisi_image)

                fake_images = fake_images.cuda()
                outputs = self.discriminator(fake_images)
                g_loss = torch.mean((outputs - 1) ** 2)

                # add reconstruct loss if True
                use_reconstruct_loss = False
                if use_reconstruct_loss:
                    reconstruct_loss = torch.mean((images - fake_images) ** 2)
                    g_loss_withreconstruct = g_loss + reconstruct_loss
                # Backprop + optimize
                self.reset_grad()
                if use_reconstruct_loss:
                    g_loss_withreconstruct.backward()
                else:
                    g_loss.backward()

                self.g_optimizer.step()

                # print the log info
                if (i + 1) % self.log_step == 0:
                    print('Epoch [%d/%d], Step[%d/%d], d_real_loss: %.4f, '
                          'd_fake_loss: %.4f, g_loss: %.4f'
                          % (epoch + 1, self.num_epochs, i + 1, total_step,
                             real_loss.data, fake_loss.data, g_loss.data))

                # save the sampled images
                if (i + 1) % self.sample_step == 0:
                    # masoud fake_images = self.generator(fixed_noise)
                    fake_images = self.generator(noisi_image)
                    torchvision.utils.save_image(self.denorm(fake_images.data),
                                                 os.path.join(self.sample_path,
                                                              'fake_samples-%d-%d.png' % (epoch + 1, i + 1)))

                    # save the model parameters for each epoch
                    g_path = os.path.join(self.model_path, 'generator-%d-%d.pkl' % (epoch + 1, i + 1))
                    d_path = os.path.join(self.model_path, 'discriminator-%d-%d.pkl' % (epoch + 1, i + 1))
                    torch.save(self.generator.state_dict(), g_path)
                    torch.save(self.discriminator.state_dict(), d_path)

                    # ======================================save test images after each epoch
                    if self.save_test == 1:
                        for j, test_image in enumerate(self.test_data_loader):
                            test_image_save = test_image[1]
                            test_image_save = os.path.basename(test_image_save[0])
                            test_image_save = test_image_save[:-4]
                            test_image = test_image[0]
                            test_image = self.to_variable(test_image)
                            g_directory_test = os.path.join(self.sample_path, 'generator-%d-%d/' % (epoch + 1, i))
                            d_directory_test = os.path.join(self.sample_path, 'discriminator-%d-%d/' % (epoch + 1, i))
                            if not os.path.exists(g_directory_test):
                                os.makedirs(g_directory_test)
                                os.makedirs(d_directory_test)

                            fake_images = self.generator(test_image)
                            d_outputs = self.discriminator(fake_images)
                            sample_path = os.path.join(g_directory_test, '%s.png' % test_image_save)
                            d_sample_path = os.path.join(d_directory_test, '%s.png' % test_image_save)
                            pure_data = d_outputs.view(-1, 11, 11)
                            pure_data = pure_data.unsqueeze(0)
                            pure_data = self.denorm(pure_data.data)
                            threshold = 0.72
                            pure_data[pure_data < threshold] = 0
                            pure_data[pure_data >= threshold] = 1
                            torchvision.utils.save_image(self.denorm(fake_images.data), sample_path, nrow=1)
                            torchvision.utils.save_image(pure_data, d_sample_path, nrow=1)
                            # ==========================   save test images after each epoch

    def sample(self):
        step = np.arange(0, 1, 0.01)
        roc_list = []
        TPRList = []
        FPRList = []
        roc_mode = True
        if roc_mode:
            for gam in step:
                test004 = np.zeros(180, int)
                test004[95:180] = 1
                predicted = np.zeros(180, int)
                for i, test_image in enumerate(self.test_data_loader):

                    test_image_save = test_image[1]
                    test_image_save = os.path.basename(test_image_save[0])
                    test_image_save = test_image_save[:-4]
                    test_image_number = test_image_save.split("_")[1]
                    test_image = test_image[0]
                    test_image = self.to_variable(test_image)
                    # Load trained parameters
                    g_path = os.path.join(self.model_path, 'generator-%s.pkl' % (self.load_sample))
                    d_path = os.path.join(self.model_path, 'discriminator-%s.pkl' % (self.load_sample))
                    self.generator.load_state_dict(torch.load(g_path))
                    self.discriminator.load_state_dict(torch.load(d_path))
                    self.generator.eval()
                    self.discriminator.eval()

                    # Sample the images
                    fake_images = self.generator(test_image)
                    d_outputs = self.discriminator(fake_images)
                    d_outputs = self.denorm(d_outputs.data)
                    d_outputs = d_outputs.view(-1, 11, 11)
                    d_outputs = d_outputs.unsqueeze(0)

                    sample_path = os.path.join(self.sample_path, 'TestImages/gen/' + test_image_save + '.png')
                    d_sample_path = os.path.join(self.sample_path, 'TestImages/disc/' + test_image_save + '.png')
                    d_outputs[d_outputs > gam] = 1
                    d_outputs[d_outputs <= gam] = 0
                    check = d_outputs[0][0].cpu()
                    check = check.numpy()
                    if 0 in check:
                        predicted[int(test_image_number) - 1] = 1

                CM = confusion_matrix(test004, predicted)
                TN = CM[0][0]
                FN = CM[1][0]
                TP = CM[1][1]
                FP = CM[0][1]

                TPR = TP / (TP + FN)
                FPR = FP / (TN + FP)

                roc_list.append([TPR, FPR])
                TPRList.append(TPR)
                FPRList.append(FPR)

            np.save("roc_list.npy", roc_list)
            np.save("TPRList.npy", TPRList)
            np.save("FPRList.npy", FPRList)

            print("Saved sampled images to '%s'" % sample_path)

        else:

            for i, test_image in enumerate(self.test_data_loader):
                imagePath = test_image[1]
                save_test_path = os.path.basename(imagePath[0])
                save_test_path = save_test_path[:-4]
                test_image = test_image[0]
                test_image = self.to_variable(test_image)
                # Load trained parameters
                g_path = os.path.join(self.model_path, 'generator-%s.pkl' % (self.load_sample))
                d_path = os.path.join(self.model_path, 'discriminator-%s.pkl' % (self.load_sample))
                self.generator.load_state_dict(torch.load(g_path))
                self.discriminator.load_state_dict(torch.load(d_path))
                self.generator.eval()
                self.discriminator.eval()

                # Sample the images
                fake_images = self.generator(test_image)
                d_outputs = self.discriminator(fake_images)
                pure_data = d_outputs.view(-1, 11, 11)
                pure_data = pure_data.unsqueeze(0)
                pure_data = self.denorm(pure_data.data)

                sample_path = os.path.join(self.sample_path, 'TestImages/gen/' + save_test_path + '.png')
                d_sample_path = os.path.join(self.sample_path, 'TestImages/disc/' + save_test_path + '.png')

                threshold = 0.7507
                pure_data[pure_data < threshold] = 0
                pure_data[pure_data >= threshold] = 1

                torchvision.utils.save_image(self.denorm(fake_images.data), sample_path, nrow=8)
                torchvision.utils.save_image(pure_data, d_sample_path, nrow=12)
                print("save test frame in " + str(i))
