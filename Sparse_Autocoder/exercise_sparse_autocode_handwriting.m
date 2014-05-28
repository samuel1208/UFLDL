
addpath(genpath('../'));

%%% Load data
images = loadMNISTImages('../data/mnist/train-images.idx3-ubyte');
labels = loadMNISTLabels('../data/mnist/train-labels.idx1-ubyte');
[img_size, img_num] = size(images);

display_network(images(:,randi(img_num,200,1)));

visibleSize = 28*28
hiddenSize = 196
sparsityParam = 0.1
lambda = 3e-3
beta = 3
patches = images(:, 1:10000);


theta = initializeParameters(hiddenSize, visibleSize);
%[cost, grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, ...
%                                     sparsityParam, beta, patches(:,1));

%numgrad = computeNumericalGradient( @(x) sparseAutoencoderCost(x, visibleSize, ...
%                                                  hiddenSize, lambda, ...
%                                                  sparsityParam, beta, ...
%                                                  patches), theta);

% Compare numerically computed gradients with the ones obtained from backpropagation
%diff = norm(numgrad-grad)/norm(numgrad+grad);
%disp(diff);

%%======================================================================
%% STEP 4: After verifying that your implementation of
%  sparseAutoencoderCost is correct, You can start training your sparse
%  autoencoder with minFunc (L-BFGS).

%  Randomly initialize the parameters
theta = initializeParameters(hiddenSize, visibleSize);

options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, patches), ...
                              theta, options);

%%======================================================================
%% STEP 5: Visualization 

W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);

display_network(W1'); 

%print -djpeg weights.jpg   % save the visualization to a file 


