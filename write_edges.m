function write_edges(in_dir, out_dir, scale)
files = dir(fullfile(in_dir, '*.flo'));
N = numel(files);
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
% mkdir(out_dir)
%%
parfor i=1:numel(files)
    file = files(i);
    fname = file.name;
    fname = fname(1:end-8);
    display(sprintf('%s',fname))
    
    edge_path = fullfile(out_dir, sprintf('%sedge.ppm', fname));
    if exist(edge_path, 'file'), continue, end
    %%
    try
        img1 = imread(fullfile(in_dir, sprintf('%simg1.ppm', fname)));
        edges_im = edgesDetect(img1, model);
        edge_path = fullfile(out_dir, sprintf('%sedge.ppm', fname));
        imwrite(edges_im, edge_path)
    catch
        display(sprintf('%s could not be read',fname))
    end
end
end