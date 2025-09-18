function out_arr = spatial_diff(out_arr, in_arr)
    % out_arr: (X1, X2, ..., Xp, p) shape.
    %   For each points of last dimension, each partial differention results is saved.
    % in_arr: (X1, X2, ..., Xp) shape
    
    % check the arrays are valid
    in_arr_size = size(in_arr);
    expected_out_arr_size = horzcat(in_arr_size,length(in_arr_size));
    assert(all(expected_out_arr_size == size(out_arr)), "Size of arrays are not valid")
    % Evaluate diff
    out_arr(:) = 0;
    
    % Compute partial differences along each dimension
    for dim = 1:length(in_arr_size)
        % Create index arrays for current and previous positions
        % Store result in output array at the last dimension index 'dim'
        indices_curr = repmat({':'}, 1, length(in_arr_size));
        indices_prev = repmat({':'}, 1, length(in_arr_size));
        out_indices = repmat({':'}, 1, length(in_arr_size) + 1);
        
        % Set up indices for difference calculation
        % Current: 2:end, Previous: 1:end-1 along dimension 'dim'
        indices_curr{dim} = 2:in_arr_size(dim);
        indices_prev{dim} = 1:(in_arr_size(dim)-1);
        out_indices{dim} = 2:in_arr_size(dim);  % Skip first element in each dimension
        out_indices{end} = dim;  % Store in the dim-th slice of last dimension
        
        % Calculate partial difference: p(i,j,k) - p(i-1,j,k) for x-direction, etc.
        out_arr(out_indices{:}) = in_arr(indices_curr{:}) - in_arr(indices_prev{:});
    end
end