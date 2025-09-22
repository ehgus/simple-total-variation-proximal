function out_arr = spatial_diff_T_cpu(out_arr, in_arr)
    % Note: Differential operator is anti-symmetric
    % out_arr: (X1, X2, ..., Xp) shape
    % in_arr: (p, X1, X2, ..., Xp) shape.

    % check the arrays are valid
    out_arr_dims = size(out_arr);
    ndims_out = length(out_arr_dims);
    expected_in_arr_dims = horzcat(ndims_out, out_arr_dims);
    assert(all(expected_in_arr_dims == size(in_arr)), "Size of arrays are not valid")

    % Initialize out_arr
    out_arr(:) = sum(in_arr,1);

    % Compute partial differences along each dimension
    for dim = 1:ndims_out
        new_out_dims = [1 prod(out_arr_dims(1:dim-1)) out_arr_dims(dim) prod(out_arr_dims(dim+1:end))];
        new_in_dims = new_out_dims;
        new_in_dims(1) = ndims_out;

        out_arr = reshape(out_arr, new_out_dims);
        in_arr = reshape(in_arr, new_in_dims);
        % Take centered differentiation
        out_arr = out_arr - circshift(in_arr(dim,:,:,:), 1, 3);
    end
    out_arr = reshape(out_arr, out_arr_dims);
end