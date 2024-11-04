filename = 'data\py.csv'; % change this for other configurations
output_file = 'data\py_peaks_widths.csv';

function fit_params_matrix = peak_aware_fitter3(filename, output_file)

    full_data = readmatrix(filename);

    frequencies = full_data(2:end,1);
    s21_full = full_data(2:end,2:end);

    num_data = size(s21_full,2); 

    fit_params_matrix = zeros(4, num_data);

    % magnetic_field_values = full_data(1,1:end);

    for col=1:num_data
        smoothed_data = smooth(s21_full(:,col),.01, 'lowess');
        [pks, locs, widths, prominence] = findpeaks(-1*smoothed_data, frequencies,'MinPeakProminence',0.4,'MinPeakDistance',.1,'SortStr','descend','NPeaks',2);
        fit_params_matrix(1:2,col) = locs;
        fit_params_matrix(3:4,col) = widths;
    end

    % index = 45;
    % smoothed_data = smooth(s21_full(:,index),.01, 'lowess');
    % plot(frequencies,-1*s21_full(:,index));
    % hold on;
    % findpeaks(-1*smoothed_data,frequencies,'Annotate','extents','MinPeakProminence',0.2,'MinPeakDistance',.1,'SortStr','descend','NPeaks',2);
    
    % figure;
    % plot(magnetic_field_values,fit_params_matrix(1,:),'bo');
    % hold on;
    % plot(magnetic_field_values,fit_params_matrix(2,:),'ro');
    % 
    % xlabel('Magnetic Field (Oe)');
    % ylabel('Frequency (GHz)');
    % title('Peaks vs. Magnetic Field');
    % % legend('show');
    % grid on;
    % 
    % figure;
    % plot(magnetic_field_values,fit_params_matrix(3,:),'bo');
    % hold on;
    % plot(magnetic_field_values,fit_params_matrix(4,:),'bo');
    % 
    % xlabel('Magnetic Field (Oe)');
    % ylabel('Line Width');
    % title('Widths vs. Magnetic Field');
    % legend('show');
    % grid on;

    headers = {'xc1','xc2','w1','w2'};
    output_data = [headers; num2cell(fit_params_matrix')];
    writetable(cell2table(output_data), output_file, 'WriteVariableNames', false);

    % disp(size(fit_params_matrix(1,:)));
    % disp(size(magnetic_field_values));

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
    
    %plotting real part 
    % A(:,2) = omega- ; A(:,1) = omega+
    % plot(H*10000,real(A(:,2))/2/pi,'LineWidth',3,'Color','blue');hold on;plot(H*10000,real(A(:,1)/2/pi),'LineWidth',3,'Color','black');hold off;
    
    % plotting imaginary part
    % plot(H*10000,-imag(A(:,1))/2/pi,'LineWidth',3,'Color','black');hold on;plot(H*10000,-imag(A(:,2)/2/pi),'LineWidth',3,'Color','black');hold off;
end

peak_aware_fitter3(filename, output_file);

% observed_data = readmatrix('data\peaks_widths.csv');

% full_data = readmatrix(filename);

% frequencies = full_data(1:end,1) .* 2e9 * pi;

% s21_full = full_data(1:end,2:end);

% num_data = size(s21_full,2); 

% locs = observed_data(:,1:3) .* 2e9 * pi;

% sorted_omega = sort(locs,2);

% magnetic_field_values = full_data(1,1:end);
% 
% function score = inter(b, sorted_omega, num_data, magnetic_field_values)
% 
%     theo = load('real_part.mat');
%     oc1 = theo.oc1; % smallest
%     oc2 = theo.oc2;
%     oc3 = theo.oc3; % greatest
% 
%     % disp(size(sorted_omega));
%     % disp(size(theo));
% 
%     score = 0;
% 
%     for i=1:num_data
%         if abs(abs(sorted_omega(i,1)-sorted_omega(i,2))-sorted_omega(i,2))<1
%             one = (sorted_omega(i,1) - oc1(i)).^2;
%             two = (sorted_omega(i,1) - oc2(i)).^2;
%             three = (sorted_omega(i,1) - oc3(i)).^2;
%             score = score + min(one,two,three);
%         else    
%             score1 = (oc3(i) - sorted_omega(i,1)).^2;
%             score2 = (oc2(i) - sorted_omega(i,2)).^2;
%             score3 = (oc1(i) - sorted_omega(i,3)).^2;
% 
%             score = score1 + score2 + score3;
%         end
%     end
% 
%     % score = abs(score);
% 
% end

% objective = @(b)inter(b,sorted_omega*1e-10,num_data,magnetic_field_values);
% 
% options = optimset('Display','iter');
% 
% b = fminbnd(objective,0,1,options);
% 
% % disp(b);
% 
% theo = theoretical_s21(b);
% 
% figure;
% plot(magnetic_field_values,real(theo(:,1)),'black');
% hold on;
% plot(magnetic_field_values,real(theo(:,2)),'black');
% hold on;
% plot(magnetic_field_values,locs(:,1),'bo');
% hold on;
% plot(magnetic_field_values,locs(:,2),'bo');
% axis([1075 1375 3.225e10 3.5e10]);

% disp(objective(0.0047, sorted_omega));