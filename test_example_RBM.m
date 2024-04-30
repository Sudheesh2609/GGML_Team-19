clear all

load 'mnist_uint8.mat'
train_x = gpuArray(double(train_x) / 255);
test_x  = gpuArray(double(test_x)  / 255);
train_y = gpuArray(double(train_y));
test_y  = gpuArray(double(test_y));

%%  ex1 train a 100 hidden unit RBM and visualize its weights
rand('state',0)
dbn.sizes = [100];
opts.numepochs =   10;
opts.batchsize = 50;
opts.momentum  =   0;
opts.alpha     =   0.1;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);
figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights