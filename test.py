import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import time
from tqdm import tqdm

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1   # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))

webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))


model = model.eval()
print(model.training)

opt.how_many = 999999
# test
for i, data in enumerate(dataset):
    with tqdm(total=len(dataset),desc=f'process img',unit='img',position=0) as pbar:
    # print(' process %d/%d img ..'%(i,len(dataset)))
        if i >= opt.how_many:
            break
        model.set_input(data)
        startTime = time.time()
        model.test()
        endTime = time.time()
        # print(endTime-startTime)
        # visuals = model.get_current_visuals()
        visuals, input_P2, fake_p2 = model.get_current_image()
        img_path = model.get_image_paths()
        # img_path = [img_path]
        # print(img_path)
        visualizer.save_images(webpage, visuals, img_path,fake_p2)
        pbar.update(i)

        pbar.set_postfix({'img_path': img_path})

webpage.save()




