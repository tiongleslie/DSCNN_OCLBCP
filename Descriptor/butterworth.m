function bf = butterworth(Img)
    Img = double(Img);
    D0 = 5;
    n = 2;
    
    % Fourier Transform
    Z = fft2(Img);
    Z = fftshift(Z);
    
    % Butterworth Filter
    M = size(Img, 1);
    N = size(Img, 2);
    S = zeros(M,N);
    
    % Generate Butterworth Filter
    for u = 1:M
       for v = 1:N
            d =  sqrt((u - (M/2))^2 + (v - (N/2))^2);
            if d == 0
                h = 0;
            else
                h = 1/(1+(D0/d)^(2*n));
            end
            S(u,v) = h * Z(u,v);
        end
    end
    
    S = ifftshift(S); 
    bf = ifft2(S);
    
    bf = real(bf);
    bf = uint8(bf);
end