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
    # epoch_loss=OrderedDict()


    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        visualizer.reset()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)

        # model.optimize_parameters()
        model.optimize_parameters()


        if total_steps % opt.display_freq == 0:
            save_result = total_steps % opt.update_html_freq == 0
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result,total_steps / opt.display_freq)
        # for key,value in model.get_current_errors().items():
        #     epoch_loss[key]=epoch_loss.get(key,0)+value

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            if total_steps % opt.display_freq == 0:
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            else:
                message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, epoch_iter, t)
                for k, v in errors.items():
                    message += '%s: %.3f ' % (k, v)
                print(message)

            # if use_wandb:
            #     run.log({
            #         "iter":total_steps/opt.print_freq,
            #         "iter_loss":errors
            #     })



            # if opt.display_id > 0:
            #     visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        # if total_steps % opt.save_latest_freq == 0:
        #     print('saving the latest model (epoch %d, total_steps %d)' %
        #           (epoch, total_steps))
        #     model.save('latest')

    # for key,value in epoch_loss.items():
    #     epoch_loss[key]=epoch_loss.get(key,0)/epoch_iter
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
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        # model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()


