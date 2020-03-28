function pattern = combine_LBP_LTP(im, thres)
    
    c_1 = uint8(zeros(size(im)));
    c_2 = uint8(zeros(size(im)));
    c_3 = uint8(zeros(size(im)));
    c_4 = uint8(zeros(size(im)));
    pattern = uint8(zeros(size(im)));
    
    [row, col] = size(im);
    
    for r = 2 : row - 1
        for c = 2 : col - 1
            centerPixel = im(r, c);
            lbp_pixel7 = LBP_rules(im(r-1, c-1), centerPixel);
            lbp_pixel6 = LBP_rules(im(r-1, c), centerPixel);
            lbp_pixel5 = LBP_rules(im(r-1, c+1), centerPixel);
            lbp_pixel4 = LBP_rules(im(r, c+1), centerPixel);
            lbp_pixel3 = LBP_rules(im(r+1, c+1), centerPixel);
            lbp_pixel2 = LBP_rules(im(r+1, c), centerPixel);
            lbp_pixel1 = LBP_rules(im(r+1, c-1), centerPixel);
            lbp_pixel0 = LBP_rules(im(r, c-1), centerPixel);
            
            ltp_pixel7 = LTP_rules(im(r-1, c-1), centerPixel, thres);
            ltp_pixel6 = LTP_rules(im(r-1, c), centerPixel, thres);
            ltp_pixel5 = LTP_rules(im(r-1, c+1) , centerPixel, thres);
            ltp_pixel4 = LTP_rules(im(r, c+1) , centerPixel, thres);
            ltp_pixel3 = LTP_rules(im(r+1, c+1) , centerPixel, thres);
            ltp_pixel2 = LTP_rules(im(r+1, c) , centerPixel, thres);
            ltp_pixel1 = LTP_rules(im(r+1, c-1) , centerPixel, thres);
            ltp_pixel0 = LTP_rules(im(r, c-1) , centerPixel, thres);
        
            u_pixel7 = pattern_rules(ltp_pixel7, 1);
            u_pixel6 = pattern_rules(ltp_pixel6, 1);
            u_pixel5 = pattern_rules(ltp_pixel5, 1);
            u_pixel4 = pattern_rules(ltp_pixel4, 1);
            u_pixel3 = pattern_rules(ltp_pixel3, 1);
            u_pixel2 = pattern_rules(ltp_pixel2, 1);
            u_pixel1 = pattern_rules(ltp_pixel1, 1);
            u_pixel0 = pattern_rules(ltp_pixel0, 1);
            
            l_pixel7 = pattern_rules(ltp_pixel7, 2);
            l_pixel6 = pattern_rules(ltp_pixel6, 2);
            l_pixel5 = pattern_rules(ltp_pixel5, 2);
            l_pixel4 = pattern_rules(ltp_pixel4, 2);
            l_pixel3 = pattern_rules(ltp_pixel3, 2);
            l_pixel2 = pattern_rules(ltp_pixel2, 2);
            l_pixel1 = pattern_rules(ltp_pixel1, 2);
            l_pixel0 = pattern_rules(ltp_pixel0, 2);
            
            c_1(r, c) = uint8(...
                lbp_pixel7 * 2^7 + u_pixel6 * 2^6 + ...
                lbp_pixel5 * 2^5 + u_pixel4 * 2^4 + ...
                lbp_pixel3 * 2^3 + u_pixel2 * 2^2 + ...
                lbp_pixel1 * 2 + u_pixel0);
            
            c_2(r, c) = uint8(...
                u_pixel7 * 2^7 + lbp_pixel6 * 2^6 + ...
                u_pixel5 * 2^5 + lbp_pixel4 * 2^4 + ...
                u_pixel3 * 2^3 + lbp_pixel2 * 2^2 + ...
                u_pixel1 * 2 + lbp_pixel0);
            
            c_3(r, c) = uint8(...
                lbp_pixel7 * 2^7 + l_pixel6 * 2^6 + ...
                lbp_pixel5 * 2^5 + l_pixel4 * 2^4 + ...
                lbp_pixel3 * 2^3 + l_pixel2 * 2^2 + ...
                lbp_pixel1 * 2 + l_pixel0);
            
            c_4(r, c) = uint8(...
                l_pixel7 * 2^7 + lbp_pixel6 * 2^6 + ...
                l_pixel5 * 2^5 + lbp_pixel4 * 2^4 + ...
                l_pixel3 * 2^3 + lbp_pixel2 * 2^2 + ...
                l_pixel1 * 2 + lbp_pixel0);
            
            pattern(r, c) = max([c_1(r, c) , c_2(r, c) , c_3(r, c) , ...
                c_4(r, c)]);
        end
    end
    
    % Average Info
    pattern(:,1) = pattern(:,2);
    pattern(:,col) = pattern(:,col-1);
    pattern(1,:) = pattern(2,:);
    pattern(row,:) = pattern(row-1,:);
end