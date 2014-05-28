function patches = sampleIMAGES()
% sampleIMAGES
% Returns 10000 patches for training

load ../data/IMAGES;    % load images from disk 

patchsize = 8;  % we'll use 8x8 patches 
numpatches = 10000;

% Initialize patches with zeros.  Your code will fill in this matrix--one
% column per patch, 10000 columns. 
patches = zeros(patchsize*patchsize, numpatches);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Fill in the variable called "patches" using data 
%  from IMAGES.  
%  
%  IMAGES is a 3D array containing 10 images
%  For instance, IMAGES(:,:,6) is a 512x512 array containing the 6th image,
%  and you can type "imagesc(IMAGES(:,:,6)), colormap gray;" to visualize
%  it. (The contrast on these images look a bit off because they have
%  been preprocessed using using "whitening."  See the lecture notes for
%  more details.) As a second example, IMAGES(21:30,21:30,1) is an image
%  patch corresponding to the pixels in the block (21,21) to (30,30) of
%  Image 1

[img_h, img_w ,img_num] = size(IMAGES);

for i = 1:numpatches
    idx_img = unidrnd(img_num);
    idx_w = unidrnd(img_w-patchsize+1);
    idx_h = unidrnd(img_h-patchsize+1);
    patches(:, i) = reshape(IMAGES(idx_h:idx_h+patchsize-1, idx_w:idx_w+patchsize-1, idx_img), patchsize*patchsize,1);
end


%% ---------------------------------------------------------------
% For the autoencoder to work well we need to normalize the data
% Specifically, since the output of the network is bounded between [0,1]
% (due to the sigmoid activation function), we have to make sure 
% the range of pixel values is also bounded between [0,1]
patches = normalizeData(patches);

end

