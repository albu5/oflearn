function resize_dataset(in_dir, out_dir, scale)
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
mkdir(out_dir)
%%
for i=1:numel(files)
    if rem(i,100) == 0
       display(sprintf('%2.2f progress', 100*i/numel(files))) 
    end
    file = files(i);
    fname = file.name;
    fname = fname(1:end-8);
%     display(sprintf('%s',fname))
    
    img1_path = fullfile(out_dir, sprintf('%s0_img1.ppm', fname));
%     if exist(img1_path, 'file'), continue, end
    %%
    try
        img1 = imread(fullfile(in_dir, sprintf('%simg1.ppm', fname)));
        img2 = imread(fullfile(in_dir, sprintf('%simg2.ppm', fname)));
%         flow = readFlowFile(fullfile(in_dir, sprintf('%sflow.flo', fname)));
%         edges_im = edgesDetect(img1, model);
        missing_im = uint8(mean(abs(double(img1)-double(img2)),3));
        
        %     imshow([[imresize(img1, 1/scale), imresize(color_flow,1/scale)];[imresize(img2, 1/scale)-flo_imtranslate(imresize(img1, 1/scale), cat(3,imresize(flow(:,:,1)/scale, 1/scale), imresize(flow(:,:,2)/scale, 1/scale))), imresize(edges_im_3,1/scale)]])
        
        img1_path = fullfile(out_dir, sprintf('%s0_img1.png', fname));
        img2_path = fullfile(out_dir, sprintf('%s0_img2.png', fname));
        edge_path = fullfile(out_dir, sprintf('%s0_edge.png', fname));
        flow_path = fullfile(out_dir, sprintf('%s0_flow.flo', fname));
        miss_path = fullfile(out_dir, sprintf('%s0_miss.png', fname));
%         imwrite(imresize(img1, 1/scale), img1_path, 'Compression', 'none')
%         imwrite(imresize(img2, 1/scale), img2_path, 'Compression', 'none')
%         imwrite(imresize(edges_im, 1/scale), edge_path, 'Compression', 'none')
        imwrite(imresize(missing_im, 1/scale), miss_path, 'Compression', 'none')
        
%         writeFlowFile(cat(3, imresize(flow(:,:,1)/scale, 1/scale), imresize(flow(:,:,2)/scale, 1/scale)), flow_path);
    catch
        display(sprintf('%s could not be read',img1_path))
    end
end
end