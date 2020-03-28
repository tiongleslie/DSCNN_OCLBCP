function final = combinationBlock(Input, thres)
    blocks = mat2cell(Input, [10 10 10 10 10 10 10 10], ...
        [10 10 10 10 10 10 10 10]);
    patternb = cell(size(blocks));
    
    for i = 1:size(patternb,1)
        for j = 1:size(patternb,2)
            patternb{i,j} = combine_LBP_LTP(blocks{i,j}, thres);
        end
    end

    my_pattern = cell2mat(patternb);
    
    final = uint8(my_pattern);
end