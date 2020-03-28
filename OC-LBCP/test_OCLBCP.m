load('regular_map.mat');

path = 'sample/rgb';
despath = 'sample/descriptor';
dirlist = dir(path);


for l = 3:length(dirlist)
    folder_left = strcat('', path, '/', dirlist(l).name, '/left/');
    folder_right = strcat(path, '/', dirlist(l).name, '/right/');
    f_pathlist = dir([folder_left, '*.jpg']);
    
    for i = 1:length(f_pathlist)
        file_left = strcat(folder_left, f_pathlist(i).name);
        file_right = strcat(folder_right, f_pathlist(i).name);
        img_left = imread(file_left);
        img_right = imread(file_right);
        
        % left image
        img_left = rgb2gray(img_left);
        img_left = histeq(img_left);
        butterw_left = butterworth(img_left);
        t_lbp_left = combinationBlock(butterw_left, 7);
        lbp_left =  regular_map(t_lbp_left+1,:);
        left_im = reshape(lbp_left,[size(t_lbp_left) 3]);
        
        % right image
        img_right = rgb2gray(img_right);
        img_right = histeq(img_right);
        butterw_right = butterworth(img_right);
        t_lbp_right = combinationBlock(butterw_right, 7);
        lbp_right =  regular_map(t_lbp_right+1,:);
        right_im = reshape(lbp_right,[size(t_lbp_right) 3]);
        
        folder = strcat(despath, '/', dirlist(l).name);
        sfolder_left = strcat(despath, '/', dirlist(l).name, '/left/');
        sfolder_right = strcat(despath, '/', dirlist(l).name, '/right/');
        
        if ~exist(folder, 'dir')
            mkdir(folder);
            mkdir(sfolder_left);
            mkdir(sfolder_right);
        end
        
        des_left = strcat(sfolder_left, f_pathlist(i).name);
        imwrite(left_im, des_left);
        
        des_right = strcat(sfolder_right, f_pathlist(i).name);
        imwrite(right_im, des_right);
    end
end