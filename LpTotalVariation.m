classdef LpTotalVariation < OptRegularizer
    properties
        weight
        p
        niter
        norm_weight
        % derived parameter
        norm
        z_tmp
        z
    end
    methods
        function obj=LpTotalVariation(weight, p, niter, norm_weight)
            obj.weight=weight;
            obj.p=p;
            obj.niter=niter;
            obj.norm_weight=norm_weight;
            obj.norm=LpUnitBall(round(1/(1-1/p))); % projection of unit ball
        end
        function create_tmp_arrays(obj, x)
            z_shape = horzcat(size(x), ndims(x));
            if isempty(obj.z) || any(size(obj.z) ~= z_shape)
                obj.z = zeros(z_shape, 'like', x);
                obj.z_tmp = zeros(z_shape, 'like', x);
            else
                obj.z(:) = 0;
                obj.z_tmp(:) = 0;
            end
        end
        function y=proximal(obj, x)
            w = obj.weight;
            v = obj.norm_weight;
            % y = x +  w(∇^T)z
            % where z = argmin[v|-z|_p* + v*(w/2|(∇^T)z|^2_2+(z^T)∇x)]
            % See Benchettou, Oumaima, Abdeslem Hafid Bentbib, and Abderrahman Bouhamidi. "An accelerated tensorial double proximal gradient method for total variation regularization problem." Journal of Optimization Theory and Applications 198.1 (2023): 111-134.
            if isgpuarray(x)
                % all-in-one
                y = lp_total_variation_cuda(x, w, v, obj.niter, obj.norm.p);
            else
                % step 1: Initialize empty z
                % if x is (X,Y,Z,) shape, z0 is (X,Y,Z,3) where fourth dimension if for saving each partial differentiation
                obj.create_tmp_arrays(x);
                % step 2: Find z using iteration
                % Initial guess of y = x + w(∇^T)z = x (We set z = 0 at the initial point)
                y = x;
                for idx=1:obj.niter
                    % Evaluate diff value of v*(w/2|(∇^T)z|^2_2+(z^T)∇x) -> diff = v∇(x + w(∇^T)z) = v∇y
                    obj.z_tmp = v .* spatial_diff(obj.z_tmp, y);
                    % Apply diff value
                    obj.z_tmp = obj.z - obj.z_tmp;
                    % Apply proximal
                    obj.z = v*obj.norm.projection(obj.z_tmp ./ v);
                    % step 3: Calculate y
                    % y = x +  w(∇^T)z
                    y = spatial_diff_T(y, obj.z);
                    y = x + w .* y;
                end
            end
        end
    end
end