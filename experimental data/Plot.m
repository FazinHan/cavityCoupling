% Load the MAT file
base_name = 'EMS2A90'; % adjust accordingly
mat_file_path = fullfile('E:\SNU\EMS\20250613', base_name, [base_name '.mat']);
data = load(mat_file_path);

% Extract frequency and magnitude data (likely numeric arrays)
freq = data.(['frequencies_' base_name]);         % NxM numeric array
mag_s11 = data.(['magnitudes_s11_' base_name]);   % NxM numeric array
mag_s12 = data.(['magnitudes_s12_' base_name]);   % NxM numeric array
mag_s21 = data.(['magnitudes_s21_' base_name]);   % NxM numeric array
mag_s22 = data.(['magnitudes_s22_' base_name]);   % NxM numeric array

% Reconstruct magnetic field vector exactly as in Python
current = 0:0.02:7.02; % Check length matches freq dimension 1 (number of files)
field = 1.61893 + 26.1952 * current; % length(field) == number of files

% Create meshgrid: X = field, Y = frequency (take frequency values from first file)
freq_vec = freq(1, :);  % frequency values for all points (assumed same for all files)
[X, Y] = meshgrid(field, freq_vec);

% Transpose magnitude matrices to fit meshgrid (frequency as rows, field as columns)
Z_s11 = mag_s11';  
Z_s12 = mag_s12';
Z_s21 = mag_s21';
Z_s22 = mag_s22';

figure;

% Plot S21
subplot(2,2,1);
contourf(X, Y, Z_s21, 300, 'LineColor', 'none');
colormap(jet);
colorbar;
%caxis([-50 0]);
title('S21 Magnitude (dB)');
xlabel('Magnetic Field (mT)');
ylabel('Frequency (Hz)');

% Plot S12
subplot(2,2,2);
contourf(X, Y, Z_s12, 300, 'LineColor', 'none');
colormap(jet);
colorbar;
%caxis([-50 0]);
title('S12 Magnitude (dB)');
xlabel('Magnetic Field (mT)');
ylabel('Frequency (Hz)');

% Plot S11
subplot(2,2,3);
contourf(X, Y, Z_s11, 300, 'LineColor', 'none');
colormap(jet);
colorbar;
title('S11 Magnitude (dB)');
xlabel('Magnetic Field (mT)');
ylabel('Frequency (Hz)');

% Plot S22
subplot(2,2,4);
contourf(X, Y, Z_s22, 300, 'LineColor', 'none');
colormap(jet);
colorbar;
title('S22 Magnitude (dB)');
xlabel('Magnetic Field (mT)');
ylabel('Frequency (Hz)');

sgtitle('S-Parameters Magnitude vs Magnetic Field and Frequency');
%pcolor(mag_s21');shading interp;colormap(turbo);caxis([-50 0])
%B=75; [~,i]=min(abs(field-B)); figure; plot(freq(i,:), mag_s21(i,:), '-b', freq(i,:), mag_s12(i,:), '-r'); legend('S21', 'S12');
%[~, idx] = max(max(mag_s21, [], 2)); figure; plot(freq(idx,:), mag_s21(idx,:), '-k'); title(sprintf('Max S21 = %.2f dB at %.2f mT', max(mag_s21(idx,:)), field(idx)));


