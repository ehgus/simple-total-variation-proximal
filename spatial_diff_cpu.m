function out_arr = spatial_diff_cpu(out_arr, in_arr)
    % out_arr: (X1, X2, ..., Xp, p) shape.
    %   For each points of last dimension, each partial differention results is saved.
    % in_arr: (X1, X2, ..., Xp) shape

    % check the arrays are valid
    in_arr_dims = size(in_arr);
    ndims_in = length(in_arr_dims);
    expected_out_arr_dims = horzcat(in_arr_dims, ndims_in);
    assert(all(expected_out_arr_dims == size(out_arr)), "Size of arrays are not valid")

    % Initialize out_arr
    out_arr = reshape(out_arr, [], ndims_in);
    for dim = 1:ndims_in
        out_arr(:,dim) = in_arr(:);
    end

    % Compute partial differences along each dimension
    for dim = 1:ndims_in
        new_in_dims = [prod(in_arr_dims(1:dim-1)) in_arr_dims(dim) prod(in_arr_dims(dim+1:end))];
        new_out_dims = horzcat(new_in_dims, ndims_in);

        in_arr = reshape(in_arr, new_in_dims);
        out_arr = reshape(out_arr, new_out_dims);
        % Take centered differentiation
        out_arr(:,:,:,dim) = out_arr(:,:,:,dim) - circshift(in_arr, -1, 2);
    end
    out_arr = reshape(out_arr, expected_out_arr_dims);
end