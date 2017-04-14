function writeEdge(impath)
%% set opts for training (see edgesTrain.m)
opts=edgesTrain();                % default options (good settings)
opts.modelDir='C:\Users\asd\Desktop\Ashish\external\edges\models';          % model will be in models/forest
opts.modelFnm='modelBsds';        % model name
opts.nPos=5e5; opts.nNeg=5e5;     % decrease to speedup training
opts.useParfor=0;                 % parallelize if sufficient memory

%% set detection parameters (can set after training)
model.opts.multiscale=0;          % for top accuracy set multiscale=1
model.opts.sharpen=2;             % for top speed set sharpen=0
model.opts.nTreesEval=4;          % for top speed set nTreesEval=1
model.opts.nThreads=4;            % max number threads for evaluation
model.opts.nms=0;                 % set to true to enable nms

%% train edge detector (~20m/8Gb per tree, proportional to nPos/nNeg)
parent_dir = pwd;
model=edgesTrain(opts); % will load model if already trained
cd(parent_dir)

%%
I = imread(impath);
E =  edgesDetect(I,model);
% imshow(I), figure, imshow(E)
imwrite(E, [impath(1:end-4), '_edge.png'])
imshow([impath(1:end-4), '_edge.png'])
end