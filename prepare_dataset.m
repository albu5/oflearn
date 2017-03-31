function prepare_dataset(in_dir, out_dir, scale)
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
parfor i=1:numel(files)
    file = files(i);
    fname = file.name;
    fname = fname(1:end-8);
    display(sprintf('%s',fname))
    
    img1_path = fullfile(out_dir, sprintf('%s0_img1.ppm', fname));
    if exist(img1_path, 'file'), continue, end
    %%
    try
        img1 = imread(fullfile(in_dir, sprintf('%simg1.ppm', fname)));
        img2 = imread(fullfile(in_dir, sprintf('%simg2.ppm', fname)));
        flow = readFlowFile(fullfile(in_dir, sprintf('%sflow.flo', fname)));
        color_flow = flowToColor(flow);
        missing_im = mean(abs(img1-img2),3);
        missing_im_3 = cat(3, missing_im, missing_im, missing_im);
        edges_im = edgesDetect(img1, model);
        edges_im_3 = uint8(255*cat(3, edges_im, edges_im, edges_im));
        
        %     imshow([[imresize(img1, 1/scale), imresize(color_flow,1/scale)];[imresize(img2, 1/scale)-flo_imtranslate(imresize(img1, 1/scale), cat(3,imresize(flow(:,:,1)/scale, 1/scale), imresize(flow(:,:,2)/scale, 1/scale))), imresize(edges_im_3,1/scale)]])
        
        img1_path = fullfile(out_dir, sprintf('%s0_img1.ppm', fname));
        img2_path = fullfile(out_dir, sprintf('%s0_img2.ppm', fname));
        edge_path = fullfile(out_dir, sprintf('%s0_edge.ppm', fname));
        miss_path = fullfile(out_dir, sprintf('%s0_miss.ppm', fname));
        flow_path = fullfile(out_dir, sprintf('%s0_flow.flo', fname));
        imwrite(imresize(img1, 1/scale), img1_path)
        imwrite(imresize(img2, 1/scale), img2_path)
        imwrite(imresize(uint8(missing_im), 1/scale), miss_path)
        imwrite(imresize(edges_im, 1/scale), edge_path)
        writeFlowFile(cat(3, imresize(flow(:,:,1)/scale, 1/scale), imresize(flow(:,:,2)/scale, 1/scale)), flow_path);
        
        
        
        
        %%
        img1_1 = fliplr(img1);
        img2_1 = fliplr(img2);
        flow_1 = fliplr(flow);
        flow_1(:,:,1) = -flow_1(:,:,1);
        missing_im_3_1 = fliplr(missing_im_3);
        edges_im_3_1 = fliplr(edges_im_3);
        
        %     imshow([[imresize(img1_1, 1/scale),...
        %         imresize(color_flow,1/scale)];...
        %         [imresize(img2_1, 1/scale)-flo_imtranslate(imresize(img1_1, 1/scale),...
        %         cat(3,imresize(flow_1(:,:,1)/scale, 1/scale), imresize(flow_1(:,:,2)/scale, 1/scale))),...
        %         imresize(edges_im_3_1,1/scale)]])
        img1_path = fullfile(out_dir, sprintf('%s1_img1.ppm', fname));
        img2_path = fullfile(out_dir, sprintf('%s1_img2.ppm', fname));
        edge_path = fullfile(out_dir, sprintf('%s1_edge.ppm', fname));
        miss_path = fullfile(out_dir, sprintf('%s1_miss.ppm', fname));
        flow_path = fullfile(out_dir, sprintf('%s1_flow.flo', fname));
        imwrite(imresize(img1_1, 1/scale), img1_path)
        imwrite(imresize(img2_1, 1/scale), img2_path)
        imwrite(imresize(uint8(missing_im_3_1(:,:,1)), 1/scale), miss_path)
        imwrite(imresize(edges_im_3_1(:,:,1), 1/scale), edge_path)
        writeFlowFile(cat(3, imresize(flow_1(:,:,1)/scale, 1/scale), imresize(flow_1(:,:,2)/scale, 1/scale)), flow_path);
        
        
        
        %%
        img1_2 = flipud(img1);
        img2_2 = flipud(img2);
        flow_2 = flipud(flow);
        flow_2(:,:,2) = -flow_2(:,:,2);
        missing_im_3_2 = flipud(missing_im_3);
        edges_im_3_2 = flipud(edges_im_3);
        
        %     imshow([[imresize(img1_2, 1/scale),...
        %         imresize(color_flow,1/scale)];...
        %         [imresize(img2_2, 1/scale)-flo_imtranslate(imresize(img1_2, 1/scale),...
        %         cat(3,imresize(flow_2(:,:,1)/scale, 1/scale), imresize(flow_2(:,:,2)/scale, 1/scale))),...
        %         imresize(edges_im_3_2,1/scale)]])
        img1_path = fullfile(out_dir, sprintf('%s2_img1.ppm', fname));
        img2_path = fullfile(out_dir, sprintf('%s2_img2.ppm', fname));
        edge_path = fullfile(out_dir, sprintf('%s2_edge.ppm', fname));
        miss_path = fullfile(out_dir, sprintf('%s2_miss.ppm', fname));
        flow_path = fullfile(out_dir, sprintf('%s2_flow.flo', fname));
        imwrite(imresize(img1_2, 1/scale), img1_path)
        imwrite(imresize(img2_2, 1/scale), img2_path)
        imwrite(imresize(uint8(missing_im_3_2(:,:,1)), 1/scale), miss_path)
        imwrite(imresize(edges_im_3_2(:,:,1), 1/scale), edge_path)
        writeFlowFile(cat(3, imresize(flow_2(:,:,1)/scale, 1/scale), imresize(flow_2(:,:,2)/scale, 1/scale)), flow_path);
        
        
        %%
        img1_3 = rot90(img1, 2);
        img2_3 = rot90(img2, 2);
        flow_3 = -rot90(flow, 2);
        missing_im_3_3 = rot90(missing_im_3, 2);
        edges_im_3_3 = rot90(edges_im_3, 2);
        
        
        %     imshow([[imresize(img1_3, 1/scale),...
        %         imresize(color_flow,1/scale)];...
        %         [imresize(img2_3, 1/scale)-flo_imtranslate(imresize(img1_3, 1/scale),...
        %         cat(3,imresize(flow_3(:,:,1)/scale, 1/scale), imresize(flow_3(:,:,2)/scale, 1/scale))),...
        %         imresize(edges_im_3_3,1/scale)]])
        img1_path = fullfile(out_dir, sprintf('%s3_img1.ppm', fname));
        img2_path = fullfile(out_dir, sprintf('%s3_img2.ppm', fname));
        edge_path = fullfile(out_dir, sprintf('%s3_edge.ppm', fname));
        miss_path = fullfile(out_dir, sprintf('%s3_miss.ppm', fname));
        flow_path = fullfile(out_dir, sprintf('%s3_flow.flo', fname));
        imwrite(imresize(img1_3, 1/scale), img1_path)
        imwrite(imresize(img2_3, 1/scale), img2_path)
        imwrite(imresize(uint8(missing_im_3_3(:,:,1)), 1/scale), miss_path)
        imwrite(imresize(edges_im_3_3(:,:,1), 1/scale), edge_path)
        writeFlowFile(cat(3, imresize(flow_3(:,:,1)/scale, 1/scale), imresize(flow_3(:,:,2)/scale, 1/scale)), flow_path);
    catch
        display(sprintf('%s could not be read',img1_path))
    end
end
end