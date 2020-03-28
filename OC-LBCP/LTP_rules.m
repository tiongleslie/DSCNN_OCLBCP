function pixel = LTP_rules(p, c, thres)
    if (p >= (c - thres)) && (p <= (c + thres))
        pixel = 0;
    elseif (p < (c - thres))
        pixel = -1;
    elseif (p > (c + thres))
        pixel = 1;
    end
end
