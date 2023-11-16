function [processed, ms] = preproc_matrix(csi_data, samp_rate, rx_cnt, rx_acnt, acf_window_size, acf_window_step, sub_idx, enable_norm, enable_filter, phase_cleaning)
%     samp_rate = 1000;
    half_rate = samp_rate / 2;
    uppe_orde = 6;
    uppe_stop = 60;
    if uppe_stop > half_rate
        uppe_stop = half_rate-1;
    end
    lowe_orde = 3;
    lowe_stop = 2;
    [lu,ld] = butter(uppe_orde,uppe_stop/half_rate,'low');
    [hu,hd] = butter(lowe_orde,lowe_stop/half_rate,'high');
    
    min_ts = size(csi_data,1);
    % Doppler Spectrum For Each Antenna
%     [csi_data, ~] = csi_get_all(r1_path);
    if phase_cleaning
        if rx_acnt == 1
            pair_num = 1;
        else
            pair_num = nchoosek(rx_acnt, 2);
        end
    else
        pair_num = rx_acnt;
    end
    %downsample
    % [30, 3]
    sub_sel = reshape(1:(30*rx_acnt),30,rx_acnt);
    % [sel, 3]
    sub_sel = sub_sel(sub_idx,:);
    % [1, sel*3]
    sub_sel = reshape(sub_sel,1,[]);
    sub_num = length(sub_idx);
    
    window_start = 1:acf_window_step:(min_ts-acf_window_size);
    window_num = length(window_start);
        
    ms = zeros(sub_num*pair_num,rx_cnt);
    processed = zeros(min_ts, sub_num*pair_num, rx_cnt);
    for rx_idx = 1:rx_cnt
        if phase_cleaning
            csi_data = reshape(csi_data(1:min_ts,sub_sel), [], sub_num, rx_acnt);
            for k = 1:rx_acnt
                conj_mult(:,((k-1)*sub_num+1):k*sub_num) = csi_data(:,:,k) .* conj(csi_data(:,:,mod(k,rx_acnt)+1));
            end
        else
            conj_mult = csi_data(1:min_ts,sub_sel);
        end

        if enable_norm
            conj_mult = normalize(conj_mult, 1, "norm", 1);
        end
        % MS top
        for sub_idx = 1:size(conj_mult,2)
            ms_sub = zeros(window_num,1);
            for win_idx = 1:window_num
                win_range = window_start(win_idx):window_start(win_idx)+acf_window_size-1;
                
                acf_sub = myacf(conj_mult(win_range, sub_idx), 1);
                ms_sub(win_idx) = acf_sub(2,:);
                % ms = normalize(ms,2,"norm",1);
            end
            ms(sub_idx,rx_idx) = mean(ms_sub);
        end
        if enable_filter
        % Filter Out Static Component & High Frequency Component(-filter = on)
            for jj = 1:size(conj_mult, 2)
                conj_mult(:,jj) = filtfilt(lu, ld, conj_mult(:,jj));
                conj_mult(:,jj) = filtfilt(hu, hd, conj_mult(:,jj));
            end
        end

        processed(:,:,rx_idx) = conj_mult;
    end
%     processed = reshape(processed, min_ts, sub_num*pair_num*rx_cnt);
end