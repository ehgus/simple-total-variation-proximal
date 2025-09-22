function out_arr = spatial_diff(out_arr, in_arr)
    % out_arr: (p, X1, X2, ..., Xp) shape.
    %   For each points of last dimension, each partial differention results is saved.
    % in_arr: (X1, X2, ..., Xp) shape

    % check the arrays are valid
    in_arr_dims = size(in_arr);
    ndims_in = length(in_arr_dims);
    expected_out_arr_dims = horzcat(ndims_in, in_arr_dims);
    assert(all(expected_out_arr_dims == size(out_arr)), "Size of arrays are not valid")

    % Compute partial differences along each dimension
    for dim = 1:ndims_in
        new_in_dims = [1 prod(in_arr_dims(1:dim-1)) in_arr_dims(dim) prod(in_arr_dims(dim+1:end))];
        new_out_dims = new_in_dims;
        new_out_dims(1) = ndims_in;

        in_arr = reshape(in_arr, new_in_dims);
        out_arr = reshape(out_arr, new_out_dims);
        % Initialize staring edges zero values.
        out_arr(dim,:,1,:) = 0;
        % Take centered differentiation
        out_arr(dim,:,2:end,:) = in_arr(1,:,2:end,:) - in_arr(1,:,1:end-1,:);
    end
    out_arr = reshape(out_arr, expected_out_arr_dims);
end