function out_arr = spatial_diff_T(out_arr, in_arr)
    % Note: Differential operator is anti-symmetric
    % out_arr: (X1, X2, ..., Xp) shape
    % in_arr: (X1, X2, ..., Xp, p) shape.

    % check the arrays are valid
    out_arr_size = size(out_arr);
    expected_in_arr_size = horzcat(out_arr_size,length(out_arr_size));
    assert(all(expected_in_arr_size == size(in_arr)), "Size of arrays are not valid")
    % Evaluate diff
    out_arr(:) = 0;
    
    % Compute partial differences along each dimension
    for dim = 1:length(out_arr_size)
        % Create index arrays for current and previous positions
        % Store result in output array at the last dimension index 'dim'
        indices_next = repmat({':'}, 1, length(out_arr_size) + 1);
        indices_curr = repmat({':'}, 1, length(out_arr_size) + 1);
        out_indices = repmat({':'}, 1, length(out_arr_size));
        
        % Set up indices for difference calculation
        % Current: 2:end, Previous: 1:end-1 along dimension 'dim'
        indices_next{dim} = 2:out_arr_size(dim);
        indices_curr{dim} = 1:(out_arr_size(dim)-1);
        indices_next{end} = dim;
        indices_curr{end} = dim;
        out_indices{dim} = 1:(out_arr_size(dim)-1);  % Skip first element in each dimension
        
        % Calculate partial difference: p(i,j,k) - p(i+1,j,k) for x-direction, etc.
        out_arr(out_indices{:}) = out_arr(out_indices{:}) + in_arr(indices_curr{:}) - in_arr(indices_next{:});
    end
end