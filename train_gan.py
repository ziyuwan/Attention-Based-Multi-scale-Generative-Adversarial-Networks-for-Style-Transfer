import time

from data import create_dataset
from models import create_gan_model
from models.gan_model import GANModel
from options.gan_options import GANOptions
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = GANOptions().parse()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training samples = %d' % dataset_size)
    gen_model, dis_model = create_gan_model(opt)
    gen_model.setup(opt)
    dis_model.setup(opt)
    gan = GANModel(gen_model, dis_model)
    visualizer = Visualizer(opt)
    total_iters = 0
    total_dis_iters = 0
    total_batch_iters = 0
    total_dis_batch_iters = 0
    cycle = dataset_size // (opt.batch_size * 10)
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        dis_model.freeze()
        for data in dataset:
            iter_start_time = time.time()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            total_batch_iters += 1
            gan.generate(data)
            if total_batch_iters % opt.display_freq == 0:
                visuals = gen_model.get_current_visuals()
                losses = gen_model.get_current_losses()
                visualizer.display_current_results(visuals, epoch, total_iters % opt.update_html_freq == 0)
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
            if total_batch_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                gen_model.save_networks(save_suffix)
        gan.use_dis = True
        dis_model.unfreeze()
        for data in dataset:
            iter_start_time = time.time()
            total_dis_iters += opt.batch_size
            epoch_iter += opt.batch_size
            total_dis_batch_iters += 1
            gan.discriminate(data)
            if total_dis_batch_iters % opt.display_freq == 0:
                losses = dis_model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp)
            if total_dis_batch_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_dis_iters))
                save_suffix = 'iter_%d' % total_dis_iters if opt.save_by_iter else 'latest'
                dis_model.save_networks(save_suffix)
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            gen_model.save_networks('latest')
            gen_model.save_networks(epoch)

            dis_model.save_networks('latest')
            dis_model.save_networks(str(epoch))
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        gen_model.update_learning_rate()
        dis_model.update_learning_rate()
