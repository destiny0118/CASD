import time
from collections import OrderedDict

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import os
import wandb

# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"
# use_wandb = True
# if use_wandb:
#     run=wandb.init(project='MotionTransfer',notes="CASD",tags=["baseline"], resume=True, anonymous='must')


opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)


visualizer = Visualizer(opt)
total_steps = 0

# wandb.config=opt

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0
    # 添加计算epoch损失
    epoch_loss=OrderedDict()


    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        visualizer.reset()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)

        # model.optimize_parameters()
        model.optimize_parameters()

        for key, value in model.get_current_errors().items():
            epoch_loss[key] = epoch_loss.get(key, 0) + value

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.display_current_results(model.get_current_visuals(), epoch, total_steps / opt.print_freq)

            # if use_wandb:
            #     run.log({
            #         "iter":total_steps/opt.print_freq,
            #         "iter_loss":errors
            #     })

    for key, value in epoch_loss.items():
        epoch_loss[key] = epoch_loss.get(key, 0) * opt.batchSize / epoch_iter

    visualizer.print_epoch_errors(epoch, opt.niter + opt.niter_decay, epoch_loss, time.time() - epoch_start_time)
    #
    # if use_wandb:
    #     run.log(
    #         {
    #             "epoch": epoch,
    #             "epoch_loss": epoch_loss,
    #         }
    #     )
    # 多少次epoch保存一次模型
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        # model.save('latest')
        model.save(epoch)

    model.update_learning_rate()
