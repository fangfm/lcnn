#! /usr/bin/env octave

argvs = argv();
argc = size(argvs, 1);

if argc != 2
  printf("-------------------------------\n");
  printf("Usage: [input_dir] [output_dir]\n");
  printf("-------------------------------\n");
  return
end

input_dir = argvs{1};
output_dir = argvs{2};

wavs = dir(strcat(input_dir, '/*.wav'));
input_dir = strcat(input_dir, '/');

hamm_coef = hamming(400);

for i = 1:size(wavs,1)
  f = fopen(strcat(input_dir, wavs(i).name));
  head = fread(f, 22, 'short');
  data = fread(f, 'short');
  fclose(f);

  %% disp("pre-emphasis");
  previous_data = zeros(size(data));
  previous_data(2:end) = data(1:end-1);
  data = data - 0.97*previous_data;
  
  indices = 1:160:size(data,1)-400;
  dataCell = cell(size(indices,2));
  n = 1;
  for j = indices
    frame_data = data(j:j+400-1);
    frame_data = frame_data .* hamm_coef;
    frame_data = fft(frame_data, 1728);
    frame_data = log(abs(frame_data(1:864)) + exp(-80));
    dataCell{n} = frame_data;
    n = n+1;
  end

  clear data;

  total = n-1;
  residual = 400 - mod(total, 400);
  residualCell = cell(residual);
  for j = 1:residual
    if total > 400
      residualCell{j} = dataCell{j};
    else
      residualCell{j} = dataCell{mod(j-1,total)+1};
    end
  end

  data = [dataCell{:}, residualCell{:}];
  
  name = strrep(wavs(i).name, 'wav', 'bin');
  f = fopen(strcat(output_dir, name), 'w');
  fwrite(f, reshape(data', size(data,1)*size(data,2), 1), 'float');
  fclose(f);

  clear data;
  clear dataCell;
  clear residualCell;
end
