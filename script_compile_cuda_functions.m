clear; clc;
script_dir = fileparts(matlab.desktop.editor.getActiveFilename);
mexcuda('-largeArrayDims', fullfile(script_dir,'lp_total_variation_cuda.cu'));