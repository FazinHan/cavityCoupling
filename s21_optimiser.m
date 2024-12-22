function s21_optimiser(arg, min_prominence, fine_smoothing, coarse_smoothing)
    filename = sprintf('data\\yig_t_sweep_outputs\\%s.csv',arg); % change this for other configurations
    output_file = sprintf('data\\yig_t_sweep_outputs\\peaks_widths\\%s_peaks_widths.csv',arg);


    peak_aware_fitter3(filename, min_prominence, fine_smoothing, coarse_smoothing, output_file);
end

function fit_params_matrix = peak_aware_fitter3(filename, min_prominence, fine_smoothing, coarse_smoothing, output_file)

    full_data = readmatrix(filename);

    frequencies = full_data(2:end,1);
    s21_full = full_data(2:end,2:end);
    hdc = full_data(1,2:end);

    num_data = size(s21_full,2); 

    fit_params_matrix = zeros(4, num_data);

    for col=1:num_data
        smoothed_data = smooth(s21_full(:,col),fine_smoothing, 'lowess');
        [pks, locs, widths, prominence] = findpeaks(-1*smoothed_data, frequencies,'MinPeakProminence',min_prominence,'MinPeakDistance',.1,'SortStr','descend','NPeaks',2);
        % fit_params_matrix(1:2,col) = smooth(locs,coarse_smoothing, 'lowess');
        fit_params_matrix(1:2,col) = locs;
        fit_params_matrix(3:4,col) = widths;
    end


    headers = {'xc1','xc2','w1','w2'};
    output_data = [headers; num2cell(fit_params_matrix')];
    writetable(cell2table(output_data), output_file, 'WriteVariableNames', false);


end

function A = theoretical_s21(b, H)
    M=0.175;%u0Ms for YIG in T unit
    r=1.760859708e11;%gyromagnetic ratio
    
    i=sqrt(-1);%imagniary number
    
    % H=[0.08:0.0005*2:0.16];%magnetic field in T unit
    
    wp=5.26302e9*2*pi;%photon mode in rad/s
    
    %H0=0.05;%in T unit
    %He=H+H0;
    fsw=r*sqrt(H.*(H+M))/2/pi;
    wr=fsw*2*pi;%rad/s
    
    K=0.048;
    
    a=2.1e-2;% alpha DeltaH = 150
    % a = 1.4e-5; % alpha DeltaH = 0.1
    % b=0.0047;%beta
    
    for ii=1:length(wr)
    p = [1 (-(wr(ii)+wp-i*a*wr(ii)-i*b*wp)) (wr(ii)*wp-i*a*wr(ii)*wp-i*b*wr(ii)*wp-a*b*wr(ii)*wp-(1/4)*K*K*wp*wp)];
    rr = roots(p);
    A(ii,1)=rr(1);
    A(ii,2)=rr(2);
    end
    
    k=find(abs(wr-wp)==min(abs(wr-wp)));
    norm_H=H/H(k);
    
end