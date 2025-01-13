import numpy as np
import os
import ntpath
import time
from . import util
#from . import html


class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id
        self.use_html = opt.isTrain
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.opt = opt
        self.saved = False
        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port=opt.display_port)

        if self.use_html:
            self.img_dir = os.path.join(opt.checkpoints_dir, opt.name, 'images')
            print('create web directory %s...' % self.img_dir)
            util.mkdirs(self.img_dir)
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, times):
        for label, image_numpy in visuals.items():
            img_path = os.path.join(self.img_dir, 'epoch%.3d_%s_%.4d.png' % (epoch, label, times))
            util.save_image(image_numpy, img_path)

    # errors: dictionary of error labels and values
    def plot_current_errors(self, epoch, counter_ratio, opt, errors):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = ''
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)
        message += '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        print(message)


    def print_epoch_errors(self, epoch, total, errors, t):
        message= ''
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)
        message += '(epoch %d / %d \t Time Taken: %d sec)' % (epoch, total, t) + '\n'
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.jpg' % (image_path[0], label)
            save_path = os.path.join(image_dir, image_name)
            print(save_path)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
