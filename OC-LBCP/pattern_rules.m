function pixel = pattern_rules(input, pattern)
    % 1 means upper, 2 means lowe
    if pattern == 1
        if input == 0
            pixel = 0;
        elseif input == -1
            pixel = 0;
        elseif input == 1
            pixel = 1;
        end
    elseif pattern == 2
        if input == 0
            pixel = 0;
        elseif input == -1
            pixel = 1;
        elseif input == 1
            pixel = 0;
        end
    end
end